cmake_minimum_required(VERSION 3.14)

project(ONNXRuntime_Example VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenCV REQUIRED)

find_package(ONNX_RUNTIME REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the ONNXRuntimeConfig.cmake file.
message(STATUS "ONNXRuntime library status:")
message(STATUS "    session path: ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}")
message(STATUS "    libraries: ${ONNX_RUNTIME_LIB}")
message(STATUS "    include path: ${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS}")

# find_path(
#   ONNX_RUNTIME_SESSION_INCLUDE_DIRS
#   onnxruntime_cxx_api.h
#   HINTS /usr/local/include/onnxruntime/core/session/
# )
# find_path(
#   ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS
#   cuda_provider_factory.h
#   HINTS /usr/local/include/onnxruntime/core/providers/cuda/
# )
# find_library(
#   ONNX_RUNTIME_LIB
#   onnxruntime
#   HINTS /usr/local/lib
# )

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
