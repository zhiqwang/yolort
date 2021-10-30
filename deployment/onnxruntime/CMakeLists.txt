cmake_minimum_required(VERSION 3.18.0)
project(yolort_onnx)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

find_package(OpenCV REQUIRED)

add_executable(yolort_onnx main.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (APPLE)
  target_include_directories(yolort_onnx PRIVATE "${ONNXRUNTIME_DIR}/include/onnxruntime/core/session")
else()
  target_include_directories(yolort_onnx PRIVATE "${ONNXRUNTIME_DIR}/include")
endif()

target_link_libraries(yolort_onnx ${OpenCV_LIBS})

if (WIN32)
  target_link_libraries(yolort_onnx "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif()

if (APPLE)
  target_link_libraries(yolort_onnx "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
endif()

if (UNIX AND NOT APPLE)
  target_link_libraries(yolort_onnx "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()
