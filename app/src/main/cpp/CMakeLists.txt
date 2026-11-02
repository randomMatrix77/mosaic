cmake_minimum_required(VERSION 3.22.1)
project(mosaic_jni LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/llama.cpp)
# libmtmd provides the official llama.cpp multimodal projector path.
set(LLAMA_BUILD_MTMD ON CACHE BOOL "Build llama.cpp multimodal support" FORCE)

# Android's NDK supplies the Vulkan loader, headers and glslc. It doesn't ship a
# CMake package config for the header-only SPIRV-Headers dependency, so provide
# the small package definition in this source tree before configuring llama.cpp.
list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(SPIRV-Headers_DIR "${CMAKE_CURRENT_LIST_DIR}/cmake/SPIRV-Headers"
    CACHE PATH "Bundled Android NDK SPIRV-Headers package" FORCE)
if (ANDROID)
    if (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
        set(MOSAIC_NDK_HOST_TAG "darwin-x86_64")
    elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
        set(MOSAIC_NDK_HOST_TAG "linux-x86_64")
    elseif (CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
        set(MOSAIC_NDK_HOST_TAG "windows-x86_64")
    else()
        message(FATAL_ERROR "Unsupported host for Android NDK Vulkan shader compiler")
    endif()
    set(Vulkan_GLSLC_EXECUTABLE
        "${CMAKE_ANDROID_NDK}/shader-tools/${MOSAIC_NDK_HOST_TAG}/glslc"
        CACHE FILEPATH "Android NDK glslc" FORCE)

    # OpenCL isn't part of the NDK, but Samsung exposes libOpenCL.so as a
    # vendor-public library. The checked-in headers are Khronos OpenCL-Headers;
    # the small link-only library is copied from the connected test device and
    # is excluded from source control.
    set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_LIST_DIR}/OpenCL-Headers" CACHE PATH
        "Khronos OpenCL headers" FORCE)
    set(OpenCL_LIBRARY "${CMAKE_CURRENT_LIST_DIR}/prebuilt/${ANDROID_ABI}/libOpenCL.so"
        CACHE FILEPATH "Android OpenCL link library" FORCE)
endif()

add_subdirectory(${LLAMA_SRC} build-llama EXCLUDE_FROM_ALL)

if (TARGET ggml-vulkan)
    # The NDK exports the C Vulkan headers only. llama.cpp's Vulkan backend
    # also uses Khronos's header-only C++ binding (vulkan.hpp).
    target_include_directories(ggml-vulkan PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/Vulkan-Headers/include"
        "${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/spirv-tools/external/spirv-headers/include")
endif()
add_library(mosaic_jni SHARED mosaic_jni.cpp)
target_include_directories(mosaic_jni PRIVATE
    ${LLAMA_SRC}/include
    ${LLAMA_SRC}/common
    ${LLAMA_SRC}/vendor
    ${LLAMA_SRC}/tools/mtmd
    ${LLAMA_SRC}/ggml/include)
target_link_libraries(mosaic_jni PRIVATE llama llama-common mtmd android log)
if (TARGET ggml-vulkan)
    # Keep the static Vulkan registry object in the final JNI library.
    target_link_libraries(mosaic_jni PRIVATE ggml-vulkan)
endif()
if (TARGET ggml-opencl)
    # Keep the static OpenCL registry object in the final JNI library.
    target_link_libraries(mosaic_jni PRIVATE ggml-opencl)
endif()
