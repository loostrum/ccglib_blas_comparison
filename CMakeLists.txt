cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(
  ccglib_blas_comparison
  DESCRIPTION "Compare ccglib with cuBLAS/rocBLAS"
  VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

if(NOT DEFINED CCGLIB_BACKEND)
  set(CCGLIB_BACKEND "CUDA")
endif()
set(CCGLIB_BACKEND
    ${CCGLIB_BACKEND}
    CACHE STRING "GPU backend API to use")
set_property(CACHE CCGLIB_BACKEND PROPERTY STRINGS "CUDA" "HIP")
set(CUDAWRAPPERS_BACKEND ${CCGLIB_BACKEND})

# fetch cudawrappers and ccglib
include(FetchContent)
FetchContent_Declare(
  cudawrappers
  GIT_REPOSITORY https://github.com/nlesc-recruit/cudawrappers
  GIT_TAG "main")
FetchContent_MakeAvailable(cudawrappers)

FetchContent_Declare(
  ccglib
  GIT_REPOSITORY https://git.astron.nl/RD/recruit/ccglib
  GIT_TAG "main")
FetchContent_MakeAvailable(ccglib)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)


add_executable(benchmark_ccglib src/ccglib.cpp)
target_link_libraries(benchmark_ccglib PUBLIC ccglib)
target_include_directories(benchmark_ccglib PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(${CCGLIB_BACKEND} STREQUAL "HIP")
  set_source_files_properties(src/benchmark_ccglib.cpp PROPERTIES LANGUAGE HIP)
endif()
