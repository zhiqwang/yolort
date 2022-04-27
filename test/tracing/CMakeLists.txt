cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(test_tracing)

find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)

# This due to some headers importing Python.h
find_package(Python3 COMPONENTS Development)

add_executable(${CMAKE_PROJECT_NAME} test_tracing.cpp)
target_compile_features(test_tracing PUBLIC cxx_range_for)

target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
  Python3::Python
)

# set C++14 to compile
set_property(TARGET test_tracing PROPERTY CXX_STANDARD 14)
