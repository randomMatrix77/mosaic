#include <android/log.h>
#include <jni.h>

#include <atomic>
#include <algorithm>
#include <memory>
#include <mutex>
#include <ostream>
#include <streambuf>
#include <string>
#include <utility>

#include "llm/llm.hpp"

namespace {
constexpr const char* kLogTag = "vibeOS-MNN";
constexpr const char* kEop = "<eop>";

struct Session {
    std::unique_ptr<MNN::Transformer::Llm> llm;
    std::atomic_bool cancel_requested{false};
    std::mutex mutex;
};

class TokenEmitter final : public std::streambuf {
public:
    TokenEmitter(JNIEnv* env, jobject callback, jmethodID on_token)
        : env_(env), callback_(callback), on_token_(on_token) {}

    bool finished() const { return finished_; }

    void flush_pending() {
        if (!finished_ && !pending_.empty()) {
            emit(pending_);
            pending_.clear();
        }
    }

protected:
    std::streamsize xsputn(const char* text, std::streamsize count) override {
        pending_.append(text, static_cast<size_t>(count));
        process();
        return count;
    }

private:
    void process() {
        const auto marker = pending_.find(kEop);
        if (marker != std::string::npos) {
            emit(pending_.substr(0, marker));
            pending_.clear();
            finished_ = true;
            return;
        }

        // Hold a suffix that could be the beginning of the MNN EOP marker.
        const size_t retained = std::min(pending_.size(), std::string(kEop).size() - 1);
        if (pending_.size() > retained) {
            emit(pending_.substr(0, pending_.size() - retained));
            pending_.erase(0, pending_.size() - retained);
        }
    }

    void emit(const std::string& value) {
        if (value.empty() || env_->ExceptionCheck()) return;
        jstring chunk = env_->NewStringUTF(value.c_str());
        if (chunk != nullptr) {
            env_->CallVoidMethod(callback_, on_token_, chunk);
            env_->DeleteLocalRef(chunk);
        }
    }

    JNIEnv* env_;
    jobject callback_;
    jmethodID on_token_;
    std::string pending_;
    bool finished_{false};
};

std::string get_string(JNIEnv* env, jstring value) {
    if (value == nullptr) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) return {};
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jlongArray make_metrics(JNIEnv* env, const MNN::Transformer::LlmContext* context) {
    const jlong values[] = {
        context == nullptr ? 0 : context->prompt_len,
        context == nullptr ? 0 : context->gen_seq_len,
        context == nullptr ? 0 : context->prefill_us,
        context == nullptr ? 0 : context->decode_us,
        context == nullptr ? 0 : context->vision_us,
        context == nullptr ? static_cast<jlong>(MNN::Transformer::LlmStatus::INTERNAL_ERROR)
                           : static_cast<jlong>(context->status)
    };
    jlongArray result = env->NewLongArray(6);
    env->SetLongArrayRegion(result, 0, 6, values);
    return result;
}
} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_io_vibeos_intelligence_mnn_MnnRuntime_nativeLoad(
    JNIEnv* env, jobject, jstring config_path, jstring runtime_config) {
    const std::string config = get_string(env, config_path);
    const std::string overrides = get_string(env, runtime_config);
    if (config.empty()) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "A model config.json path is required.");
        return 0;
    }

    auto llm = std::unique_ptr<MNN::Transformer::Llm>(MNN::Transformer::Llm::createLLM(config));
    if (!llm) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "MNN could not create an LLM from config.json.");
        return 0;
    }
    if (!overrides.empty() && !llm->set_config(overrides)) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "MNN rejected the runtime configuration.");
        return 0;
    }
    if (!llm->load()) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "MNN failed to load the model graph or weights.");
        return 0;
    }

    auto* session = new Session{std::move(llm)};
    __android_log_print(ANDROID_LOG_INFO, kLogTag, "Loaded model config: %s", config.c_str());
    return reinterpret_cast<jlong>(session);
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_io_vibeos_intelligence_mnn_MnnRuntime_nativeGenerate(
    JNIEnv* env, jobject, jlong pointer, jstring prompt, jint max_new_tokens, jobject callback) {
    auto* session = reinterpret_cast<Session*>(pointer);
    if (session == nullptr || session->llm == nullptr) {
        env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), "No MNN model is loaded.");
        return nullptr;
    }
    if (callback == nullptr) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "A token callback is required.");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(session->mutex);
    session->cancel_requested.store(false);
    const std::string prompt_text = get_string(env, prompt);
    jclass callback_class = env->GetObjectClass(callback);
    jmethodID on_token = env->GetMethodID(callback_class, "onToken", "(Ljava/lang/String;)V");
    if (on_token == nullptr) {
        env->ThrowNew(env->FindClass("java/lang/IllegalArgumentException"), "Token callback must implement onToken(String).");
        return nullptr;
    }

    TokenEmitter emitter(env, callback, on_token);
    std::ostream output(&emitter);
    // This is the official MNN Chat stepping pattern: prefill first, then generate one token at a time.
    session->llm->response(prompt_text, &output, kEop, 0);
    int generated = 0;
    while (!session->cancel_requested.load() && !emitter.finished() && generated < max_new_tokens) {
        session->llm->generate(1);
        ++generated;
    }
    emitter.flush_pending();
    const auto* context = session->llm->getContext();
    __android_log_print(ANDROID_LOG_INFO, kLogTag,
                        "generation status=%d prompt=%d generated=%d prefill_us=%lld decode_us=%lld",
                        context == nullptr ? -1 : static_cast<int>(context->status),
                        context == nullptr ? 0 : context->prompt_len,
                        context == nullptr ? 0 : context->gen_seq_len,
                        context == nullptr ? 0 : static_cast<long long>(context->prefill_us),
                        context == nullptr ? 0 : static_cast<long long>(context->decode_us));
    return make_metrics(env, context);
}

extern "C" JNIEXPORT void JNICALL
Java_io_vibeos_intelligence_mnn_MnnRuntime_nativeCancel(JNIEnv*, jobject, jlong pointer) {
    auto* session = reinterpret_cast<Session*>(pointer);
    if (session != nullptr) session->cancel_requested.store(true);
}

extern "C" JNIEXPORT void JNICALL
Java_io_vibeos_intelligence_mnn_MnnRuntime_nativeUnload(JNIEnv*, jobject, jlong pointer) {
    auto* session = reinterpret_cast<Session*>(pointer);
    delete session;
}
