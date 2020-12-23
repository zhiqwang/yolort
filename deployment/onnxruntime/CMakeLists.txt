cmake_minimum_required(VERSION 3.14)

project(YOLOv5_ORT_Example VERSION 0.2.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCV REQUIRED)

find_path(
  ONNX_RUNTIME_SESSION_INCLUDE_DIRS
  onnxruntime_cxx_api.h
  HINTS /usr/local/include/onnxruntime/core/session/
)

find_library(
  ONNX_RUNTIME_LIB
  onnxruntime
  HINTS /usr/local/lib
)

# If the package has been found, several variables will be set
message(STATUS "ONNXRuntime library status:")
message(STATUS "    include path: ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}")
message(STATUS "    libraries: ${ONNX_RUNTIME_LIB}")

FILE(GLOB YOLO_SOURCE_FILES src/*.cpp)

add_executable(yolo_inference ${YOLO_SOURCE_FILES})

target_include_directories(
  yolo_inference
  PRIVATE
  ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}
  ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolo_inference PRIVATE ${ONNX_RUNTIME_LIB} ${OpenCV_LIBRARIES})
