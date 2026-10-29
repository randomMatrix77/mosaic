#include <android/log.h>
#include <cmath>
#include <jni.h>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat.h"
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace {
constexpr int kContextSize = 2048;
constexpr int kMaxTokens = 128;
constexpr int kAllGpuLayers = -1;

enum class LoadedModelKind { None, Generation, Embedding };

void android_log_callback(enum ggml_log_level level, const char *text, void *) {
    const int priority = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR :
        level == GGML_LOG_LEVEL_WARN ? ANDROID_LOG_WARN :
        level == GGML_LOG_LEVEL_INFO ? ANDROID_LOG_INFO : ANDROID_LOG_DEBUG;
    __android_log_write(priority, "MosaicLlama", text);
}

void image_log(const char *message) {
    __android_log_write(ANDROID_LOG_ERROR, "MosaicLlama", message);
}

std::mutex g_mutex;
llama_model *g_model = nullptr;
common_chat_templates_ptr g_templates;
LoadedModelKind g_model_kind = LoadedModelKind::None;
mtmd_context *g_mtmd = nullptr;

void unload_locked() {
    g_templates.reset();
    if (g_mtmd) mtmd_free(g_mtmd);
    g_mtmd = nullptr;
    if (g_model) llama_model_free(g_model);
    g_model = nullptr;
    g_model_kind = LoadedModelKind::None;
}

std::string take(JNIEnv *env, jstring value) {
    const char *chars = env->GetStringUTFChars(value, nullptr);
    std::string copy(chars ? chars : "");
    if (chars) env->ReleaseStringUTFChars(value, chars);
    return copy;
}

jstring result(JNIEnv *env, const std::string &value) {
    return env->NewStringUTF(value.c_str());
}

bool has_gpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const ggml_backend_dev_t device = ggml_backend_dev_get(i);
        const auto type = ggml_backend_dev_type(device);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

std::string backend_summary() {
    std::ostringstream summary;
    summary << "llama.cpp backends:";
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const ggml_backend_dev_t device = ggml_backend_dev_get(i);
        summary << " [" << ggml_backend_dev_name(device) << ": "
                << ggml_backend_dev_description(device) << "]";
    }
    return summary.str();
}

void set_batch(llama_batch_ext *batch, const std::vector<llama_token> &tokens, int pos) {
    llama_batch_ext_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos token_pos = pos + static_cast<int>(i);
        llama_batch_ext_set_pos(batch, index, &token_pos);
    }
    llama_batch_ext_set_output_logits(batch, static_cast<int>(tokens.size()) - 1, true);
}

void set_embedding_batch(llama_batch_ext *batch, const std::vector<llama_token> &tokens) {
    llama_batch_ext_clear(batch);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const int index = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos token_pos = static_cast<llama_pos>(i);
        llama_batch_ext_set_pos(batch, index, &token_pos);
        llama_batch_ext_set_output_embd(batch, index, true);
    }
}

jfloatArray embedding_result(JNIEnv *env, const float *embedding, int dimensions) {
    if (!embedding || dimensions <= 0) return nullptr;
    std::vector<float> normalized(dimensions);
    double sum_squares = 0.0;
    for (int i = 0; i < dimensions; ++i) {
        if (!std::isfinite(embedding[i])) return nullptr;
        sum_squares += static_cast<double>(embedding[i]) * embedding[i];
    }
    const double norm = std::sqrt(sum_squares);
    if (!std::isfinite(norm) || norm == 0.0) return nullptr;
    for (int i = 0; i < dimensions; ++i) normalized[i] = static_cast<float>(embedding[i] / norm);
    jfloatArray array = env->NewFloatArray(dimensions);
    if (array) env->SetFloatArrayRegion(array, 0, dimensions, normalized.data());
    return array;
}

bool append_image_embedding(llama_batch_ext *batch, const mtmd_input_chunk *chunk, int &position) {
    if (!g_mtmd || !chunk || mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        image_log("image embedding failed: invalid mtmd image chunk");
        return false;
    }
    if (mtmd_encode_chunk(g_mtmd, chunk) != 0) {
        image_log("image embedding failed: mtmd projector encode failed");
        return false;
    }
    const float *encoded = mtmd_get_output_embd(g_mtmd);
    const int token_count = static_cast<int>(mtmd_input_chunk_get_n_tokens(chunk));
    const int input_dimensions = llama_model_n_embd_inp(g_model);
    if (!encoded || token_count <= 0 || input_dimensions <= 0 || token_count >= kContextSize) {
        image_log("image embedding failed: invalid projector output dimensions or token count");
        return false;
    }

    for (int i = 0; i < token_count; ++i) {
        const llama_embd row = {encoded + static_cast<size_t>(i) * input_dimensions, 1,
                                static_cast<size_t>(input_dimensions)};
        const int index = llama_batch_ext_add_embd(batch, 0, row);
        const llama_pos token_position = position++;
        llama_batch_ext_set_pos(batch, index, &token_position);
        llama_batch_ext_set_output_embd(batch, index, true);
    }
    return true;
}

std::string generate_locked(const std::vector<common_chat_msg> &messages) {
    common_chat_templates_inputs template_inputs;
    template_inputs.messages = messages;
    template_inputs.add_generation_prompt = true;
    template_inputs.enable_thinking = false; // Qwen 3.5 non-thinking mode.
    const std::string prompt = common_chat_templates_apply(
        g_templates.get(), template_inputs).prompt.str();
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    int count = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, false, true);
    if (count <= 0 || count >= kContextSize - kMaxTokens) return "ERROR: prompt is too long.";
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), tokens.data(), count, false, true) < 0) {
        return "ERROR: tokenization failed.";
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = kContextSize;
    context_params.n_batch = count;
    context_params.n_ubatch = count;
    llama_context *context = llama_init_from_model(g_model, context_params);
    if (!context) return "ERROR: unable to allocate a 2048-token context.";
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(20));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.8f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));
    llama_batch_ext *batch = llama_batch_ext_init(context);
    set_batch(batch, tokens, 0);

    std::string output;
    int position = 0;
    for (int generated = 0; generated < kMaxTokens; ++generated) {
        if (llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
            output = "ERROR: llama.cpp decode failed.";
            break;
        }
        position += static_cast<int>(tokens.size());
        const llama_token next = llama_sampler_sample(sampler, context, -1);
        if (llama_vocab_is_eog(vocab, next)) break;
        char piece[256];
        const int piece_size = llama_token_to_piece(vocab, next, piece, sizeof(piece), 0, true);
        if (piece_size < 0) { output = "ERROR: token conversion failed."; break; }
        output.append(piece, piece_size);
        llama_sampler_accept(sampler, next);
        tokens.assign(1, next);
        set_batch(batch, tokens, position);
    }
    llama_batch_ext_free(batch);
    llama_sampler_free(sampler);
    llama_free(context);
    return output.empty() ? "(no generated tokens)" : output;
}
} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_dev_mosaic_poc_NativeLlama_initialize(JNIEnv *env, jclass, jstring native_lib_dir) {
    std::lock_guard<std::mutex> lock(g_mutex);
    llama_log_set(android_log_callback, nullptr);
    mtmd_helper_log_set(android_log_callback, nullptr);
    // CPU is linked into this library: avoid filesystem discovery, which Android
    // restricts when native libraries are loaded directly from the APK.
    (void) take(env, native_lib_dir);
    llama_backend_init();
    const std::string system_info = llama_print_system_info();
    const std::string summary = backend_summary();
    __android_log_write(ANDROID_LOG_INFO, "MosaicLlama", summary.c_str());
    return result(env, system_info + "\n" + summary);
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_mosaic_poc_NativeLlama_loadModel(JNIEnv *env, jclass, jstring path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();
    const std::string model_path = take(env, path);
    llama_model_params params = llama_model_default_params();
    // -1 asks llama.cpp to offload every available layer to the Vulkan device.
    // If model allocation fails, retain a working CPU path rather than leaving
    // the application without a usable model.
    const bool gpu_available = has_gpu_device();
    params.n_gpu_layers = gpu_available ? kAllGpuLayers : 0;
    g_model = llama_model_load_from_file(model_path.c_str(), params);
    bool cpu_fallback = false;
    if (!g_model && gpu_available) {
        __android_log_write(ANDROID_LOG_WARN, "MosaicLlama",
                            "GPU model load failed; retrying Qwen on CPU.");
        params.n_gpu_layers = 0;
        g_model = llama_model_load_from_file(model_path.c_str(), params);
        cpu_fallback = true;
    }
    if (!g_model) return result(env, "ERROR: llama.cpp could not load the selected GGUF.");
    g_templates = common_chat_templates_init(g_model, "");
    g_model_kind = LoadedModelKind::Generation;
    if (gpu_available && !cpu_fallback) {
        return result(env, "OK: model loaded with GPU offload");
    }
    return result(env, gpu_available ? "OK: model loaded with CPU fallback" :
                                  "OK: model loaded on CPU (GPU unavailable)");
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_mosaic_poc_NativeLlama_loadEmbeddingModel(JNIEnv *env, jclass, jstring path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();
    const std::string model_path = take(env, path);
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 0; // CPU-only experiment.
    g_model = llama_model_load_from_file(model_path.c_str(), params);
    if (!g_model) return result(env, "ERROR: llama.cpp could not load the selected embedding GGUF.");
    g_model_kind = LoadedModelKind::Embedding;
    return result(env, "OK: embedding model loaded");
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_mosaic_poc_NativeLlama_loadEmbeddingProjector(JNIEnv *env, jclass, jstring path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || g_model_kind != LoadedModelKind::Embedding) {
        return result(env, "ERROR: load the EmbeddingGemma text GGUF first.");
    }
    if (g_mtmd) mtmd_free(g_mtmd);
    g_mtmd = nullptr;
    const std::string projector_path = take(env, path);
    mtmd_context_params params = mtmd_context_params_default();
    params.use_gpu = false;
    g_mtmd = mtmd_init_from_file(projector_path.c_str(), g_model, params);
    if (!g_mtmd) return result(env, "ERROR: llama.cpp could not load the selected multimodal projector.");
    if (!mtmd_support_vision(g_mtmd)) {
        mtmd_free(g_mtmd);
        g_mtmd = nullptr;
        return result(env, "ERROR: the selected projector does not support image input.");
    }
    return result(env, "OK: multimodal projector loaded");
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_mosaic_poc_NativeLlama_generate(JNIEnv *env, jclass, jstring message) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || g_model_kind != LoadedModelKind::Generation) {
        return result(env, "ERROR: no generation model is loaded.");
    }
    common_chat_msg chat;
    chat.role = "user";
    chat.content = take(env, message);
    return result(env, generate_locked({chat}));
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_mosaic_poc_NativeLlama_generateChat(JNIEnv *env, jclass, jstring serialized_messages) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || g_model_kind != LoadedModelKind::Generation) {
        return result(env, "ERROR: no generation model is loaded.");
    }
    try {
        const auto input = nlohmann::json::parse(take(env, serialized_messages));
        if (!input.is_array() || input.empty() || input.size() > 128) {
            return result(env, "ERROR: chat history must be a non-empty array of at most 128 messages.");
        }
        std::vector<common_chat_msg> messages;
        for (const auto &item : input) {
            if (!item.is_object() || !item.contains("role") || !item.contains("content") ||
                !item["role"].is_string() || !item["content"].is_string()) {
                return result(env, "ERROR: each chat message requires string role and content fields.");
            }
            const std::string role = item["role"].get<std::string>();
            if (role != "system" && role != "user" && role != "assistant") {
                return result(env, "ERROR: unsupported chat role.");
            }
            messages.push_back({role, item["content"].get<std::string>()});
        }
        return result(env, generate_locked(messages));
    } catch (const std::exception &) {
        return result(env, "ERROR: invalid serialized chat history.");
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_dev_mosaic_poc_NativeLlama_embedText(JNIEnv *env, jclass, jstring text) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || g_model_kind != LoadedModelKind::Embedding) return nullptr;

    const std::string input = take(env, text);
    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    const int count = -llama_tokenize(vocab, input.c_str(), input.size(), nullptr, 0, true, true);
    if (count <= 0 || count >= kContextSize) return nullptr;
    std::vector<llama_token> tokens(count);
    if (llama_tokenize(vocab, input.c_str(), input.size(), tokens.data(), count, true, true) < 0) return nullptr;

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = kContextSize;
    context_params.n_batch = count;
    context_params.n_ubatch = count; // Required for non-causal embedding models.
    context_params.embeddings = true;
    llama_context *context = llama_init_from_model(g_model, context_params);
    if (!context) return nullptr;
    llama_batch_ext *batch = llama_batch_ext_init(context);
    set_embedding_batch(batch, tokens);
    const int process_result = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch);
    const float *embedding = process_result == 0 ? llama_get_embeddings_seq(context, 0) : nullptr;
    jfloatArray output = embedding_result(env, embedding, llama_model_n_embd_out(g_model));
    llama_batch_ext_free(batch);
    llama_free(context);
    return output;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_dev_mosaic_poc_NativeLlama_embedImage(JNIEnv *env, jclass, jstring path) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || g_model_kind != LoadedModelKind::Embedding || !g_mtmd) return nullptr;
    const std::string image_path = take(env, path);
    const mtmd_helper_init_opt init_options = mtmd_helper_init_opt_default();
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_file(
        g_mtmd, image_path.c_str(), false, init_options);
    if (!wrapper.bitmap) {
        image_log("image embedding failed: image decoder could not read the supplied file");
        return nullptr;
    }

    mtmd_input_part part = {nullptr, wrapper.bitmap};
    const mtmd_input_part *parts[] = {&part};
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    const int tokenize_result = mtmd_tokenize_from_parts(g_mtmd, chunks, parts, 1, true);
    jfloatArray output = nullptr;
    if (tokenize_result == 0 && mtmd_input_chunks_size(chunks) > 0) {
        int total_tokens = 0;
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            total_tokens += static_cast<int>(mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, i)));
        }
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = kContextSize;
        context_params.n_batch = total_tokens;
        context_params.n_ubatch = total_tokens;
        context_params.embeddings = true;
        llama_context *context = llama_init_from_model(g_model, context_params);
        if (context) {
            llama_batch_ext *batch = llama_batch_ext_init(context);
            const bool old_causal = llama_get_causal_attn(context);
            // EmbeddingGemma's vision projector requires bidirectional attention. All
            // text and image rows must enter one batch so they can attend together.
            llama_set_causal_attn(context, false);
            int position = 0;
            bool assembled = total_tokens > 0 && total_tokens < kContextSize;
            for (size_t i = 0; assembled && i < mtmd_input_chunks_size(chunks); ++i) {
                const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
                if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    size_t token_count = 0;
                    const llama_token *tokens = mtmd_input_chunk_get_tokens_text(chunk, &token_count);
                    for (size_t j = 0; j < token_count; ++j) {
                        const int index = llama_batch_ext_add_token(batch, 0, tokens[j]);
                        const llama_pos token_position = position++;
                        llama_batch_ext_set_pos(batch, index, &token_position);
                        llama_batch_ext_set_output_embd(batch, index, true);
                    }
                } else if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    assembled = append_image_embedding(batch, chunk, position);
                } else {
                    assembled = false;
                }
            }
            if (assembled && llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch) == 0) {
                output = embedding_result(env, llama_get_embeddings_seq(context, 0),
                                          llama_model_n_embd_out(g_model));
            } else if (assembled) {
                image_log("image embedding failed: llama.cpp multimodal decode failed");
            }
            llama_set_causal_attn(context, old_causal);
            llama_batch_ext_free(batch);
        }
        if (!context) image_log("image embedding failed: unable to allocate llama.cpp context");
        if (context) llama_free(context);
    } else {
        image_log("image embedding failed: mtmd tokenization did not produce exactly one chunk");
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(wrapper.bitmap);
    if (wrapper.video_ctx) mtmd_helper_video_free(wrapper.video_ctx);
    return output;
}

extern "C" JNIEXPORT void JNICALL
Java_dev_mosaic_poc_NativeLlama_unloadModel(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    unload_locked();
}

extern "C" JNIEXPORT jboolean JNICALL
Java_dev_mosaic_poc_NativeLlama_isModelLoaded(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_model_kind == LoadedModelKind::Generation ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_dev_mosaic_poc_NativeLlama_isEmbeddingModelLoaded(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_model_kind == LoadedModelKind::Embedding ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_dev_mosaic_poc_NativeLlama_isEmbeddingProjectorLoaded(JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_mtmd ? JNI_TRUE : JNI_FALSE;
}
