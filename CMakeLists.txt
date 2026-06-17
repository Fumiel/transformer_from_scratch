cmake_minimum_required(VERSION 3.16)
project(tiny_transformer_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tiny_transformer_cpp
    cpp/main.cpp
    cpp/tensor.cpp
    cpp/layers.cpp
    cpp/attention.cpp
    cpp/model.cpp
    cpp/weights_loader.cpp
)

target_include_directories(tiny_transformer_cpp PRIVATE cpp)

target_compile_options(tiny_transformer_cpp PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
