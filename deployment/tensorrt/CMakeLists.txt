cmake_minimum_required(VERSION 3.18.0)
project(yolort_trt LANGUAGES CXX CUDA)

option(TENSORRT_DIR "Path to built TensorRT directory." STRING)
message(STATUS "TENSORRT_DIR: ${TENSORRT_DIR}")

find_package(OpenCV REQUIRED)

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	add_compile_options(-Wall -O2)
ELSE()
	add_compile_options(-Wextra -Wall -Wno-unused -Wno-deprecated-declarations)
ENDIF()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda
find_package(CUDA)
message(STATUS "CUDA_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
IF (CMAKE_SYSTEM_NAME MATCHES "Linux")
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
ELSEIF (CMAKE_SYSTEM_NAME MATCHES "Windows")
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
ELSE()
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
ENDIF()

# tensorrt
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} nvinfer cudart nvonnxparser nvinfer_plugin)
