cmake_minimum_required(VERSION 3.0.0)
project(yolort_onnx VERSION 0.1.0)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

add_executable(yolort_onnx main.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_include_directories(yolort_onnx PRIVATE "${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_DIR}/lib")

target_link_libraries(yolort_onnx ${OpenCV_LIBS})

if (WIN32)
    target_link_libraries(yolort_onnx "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    target_link_libraries(yolort_onnx "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)
