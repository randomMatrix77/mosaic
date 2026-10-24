cmake_minimum_required(VERSION 3.22.1)
project(vibeos_mnn LANGUAGES CXX)

if(NOT MNN_SOURCE_DIR OR NOT EXISTS "${MNN_SOURCE_DIR}/transformers/llm/engine/include/llm/llm.hpp")
    message(FATAL_ERROR "MNN_SOURCE_DIR must point at the pinned official MNN checkout. Run scripts/build-mnn.sh first.")
endif()

set(MNN_RUNTIME "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI}/libMNN.so")
if(NOT EXISTS "${MNN_RUNTIME}")
    message(FATAL_ERROR "Missing ${MNN_RUNTIME}. Run native/android/scripts/build-mnn.sh first.")
endif()

add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION "${MNN_RUNTIME}")

add_library(vibeos_mnn SHARED mnn_jni.cpp)
target_compile_features(vibeos_mnn PRIVATE cxx_std_17)
target_compile_options(vibeos_mnn PRIVATE -Wall -Wextra -Werror=return-type)
target_include_directories(vibeos_mnn PRIVATE
    "${MNN_SOURCE_DIR}/include"
    "${MNN_SOURCE_DIR}/transformers/llm/engine/include"
)
target_link_options(vibeos_mnn PRIVATE "-Wl,-z,max-page-size=16384")
target_link_libraries(vibeos_mnn android log MNN)

