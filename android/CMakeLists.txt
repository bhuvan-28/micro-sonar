cmake_minimum_required(VERSION 3.22)
project(MicroSonar VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Dependencies
# ============================================================

# Google Oboe (audio I/O) - pulled via FetchContent
include(FetchContent)

FetchContent_Declare(
    oboe
    GIT_REPOSITORY https://github.com/google/oboe.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(oboe)

# KissFFT (header-only FFT for Android)
FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG        v131
)
FetchContent_MakeAvailable(kissfft)

# ============================================================
# Micro-Sonar Native Library
# ============================================================
add_library(microsonar SHARED
    cpp/sonar_engine.cpp
    cpp/chirp_generator.cpp
    cpp/matched_filter.cpp
    cpp/feature_extractor.cpp
    cpp/jni_bridge.cpp
)

target_include_directories(microsonar PRIVATE
    cpp/
    ${kissfft_SOURCE_DIR}
)

target_link_libraries(microsonar
    oboe::oboe
    android
    log
)

# TFLite (add path to your downloaded TFLite AAR/headers)
# Uncomment once TFLite C++ API is set up:
# target_link_libraries(microsonar tensorflowlite)

