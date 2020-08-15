cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(demonet)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(TorchVision REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCE_FILES src/*.cpp)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  ${TORCH_LIBRARIES}
  TorchVision::TorchVision
)

# set C++14 to compile PyTorch
set_property(TARGET demonet PROPERTY CXX_STANDARD 14)
