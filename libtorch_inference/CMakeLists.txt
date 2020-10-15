cmake_minimum_required(VERSION 3.14)
project(yolo_inference)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

find_package(OpenCV REQUIRED)
if (NOT OpenCV_FOUND)
  message(FATAL_ERROR "OpenCV not found")
endif (NOT OpenCV_FOUND)

# This due to some headers importing Python.h
find_package(Python3 COMPONENTS Development)

include_directories(${PROJECT_SOURCE_DIR}/include)

FILE(GLOB YOLO_SOURCE_FILES src/*.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${YOLO_SOURCE_FILES})
target_compile_features(yolo_inference PUBLIC cxx_range_for)

target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
  Python3::Python
)

# set C++14 to compile
set_property(TARGET yolo_inference PROPERTY CXX_STANDARD 14)
