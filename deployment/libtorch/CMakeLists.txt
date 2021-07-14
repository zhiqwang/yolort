cmake_minimum_required(VERSION 3.14)

project(yolo_inference)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# This due to some headers importing Python.h
find_package(Python3 COMPONENTS Development)

FILE(GLOB YOLO_SOURCE_FILES *.cpp)

add_executable(yolo_inference ${YOLO_SOURCE_FILES})
target_compile_features(yolo_inference PUBLIC cxx_range_for)

target_link_libraries(
  yolo_inference
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
  Python3::Python
)

# set C++14 to compile
set_property(TARGET yolo_inference PROPERTY CXX_STANDARD 14)
