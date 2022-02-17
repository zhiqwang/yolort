cmake_minimum_required(VERSION 3.18.0)
project(yolort_trt)

option(TENSORRT_DIR "Path to built TensorRT directory." STRING)
message(STATUS "TENSORRT_DIR: ${TENSORRT_DIR}")

find_package(OpenCV REQUIRED)

add_compile_options(-Wextra -Wall -Wno-unused -Wno-deprecated-declarations)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nvinfer cudart nvonnxparser nvinfer_plugin)
