#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
android_dir="$(cd "$script_dir/.." && pwd)"
repo_root="$(cd "$android_dir/../.." && pwd)"
mnn_source_dir="${MNN_SOURCE_DIR:-$android_dir/third_party/MNN}"
android_ndk="${ANDROID_NDK:-/Users/kronos/Library/Android/sdk/ndk/27.2.12479018}"
cmake_bin="${VIBEOS_CMAKE_BIN:-/Users/kronos/Library/Android/sdk/cmake/3.22.1/bin/cmake}"
build_dir="$mnn_source_dir/project/android/build-vibeos-arm64"
output_dir="$android_dir/app/src/main/jniLibs/arm64-v8a"
expected_revision="$(tr -d '[:space:]' < "$script_dir/MNN_REVISION")"

if [[ ! -f "$mnn_source_dir/CMakeLists.txt" ]]; then
  echo "MNN source was not found at $mnn_source_dir." >&2
  echo "Clone https://github.com/alibaba/MNN there, then checkout $expected_revision." >&2
  exit 2
fi
if [[ ! -f "$android_ndk/build/cmake/android.toolchain.cmake" ]]; then
  echo "Android NDK was not found at $android_ndk." >&2
  exit 2
fi
if [[ ! -x "$cmake_bin" ]]; then
  echo "CMake was not found at $cmake_bin." >&2
  exit 2
fi

actual_revision="$(git -C "$mnn_source_dir" rev-parse HEAD)"
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "MNN revision mismatch: expected $expected_revision, got $actual_revision." >&2
  exit 2
fi

"$cmake_bin" -S "$mnn_source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$android_ndk/build/cmake/android.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE=Release \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-26 \
  -DANDROID_STL=c++_static \
  -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
  -DMNN_BUILD_FOR_ANDROID_COMMAND=ON \
  -DMNN_LOW_MEMORY=ON \
  -DMNN_CPU_WEIGHT_DEQUANT_GEMM=ON \
  -DMNN_BUILD_LLM=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
  -DMNN_ARM82=ON \
  -DMNN_USE_LOGCAT=ON \
  -DMNN_OPENCL=ON \
  -DLLM_SUPPORT_VISION=ON \
  -DMNN_BUILD_OPENCV=ON \
  -DMNN_IMGCODECS=ON \
  -DMNN_SEP_BUILD=OFF \
  -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,max-page-size=16384
"$cmake_bin" --build "$build_dir" --parallel 4

mnn_library="$(find "$build_dir" -name libMNN.so -type f -print -quit)"
if [[ -z "$mnn_library" ]]; then
  echo "MNN build completed without libMNN.so." >&2
  exit 3
fi
mkdir -p "$output_dir"
cp "$mnn_library" "$output_dir/libMNN.so"
printf 'sdk.dir=/Users/kronos/Library/Android/sdk\nmnn.source.dir=%s\n' "$mnn_source_dir" > "$android_dir/local.properties"
echo "Prepared $output_dir/libMNN.so for APK packaging."
