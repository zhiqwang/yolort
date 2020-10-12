cmake_minimum_required(VERSION 3.14)
project( libtorch_inference )

find_package( Torch REQUIRED )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall" )

find_package( OpenCV REQUIRED )
if ( NOT OpenCV_FOUND )
  message( FATAL_ERROR "OpenCV not found" )
endif ( NOT OpenCV_FOUND )

include_directories( ${PROJECT_SOURCE_DIR}/include )

FILE( GLOB YOLO_SOURCE_FILES src/*.cpp )

add_executable( ${CMAKE_PROJECT_NAME} ${YOLO_SOURCE_FILES} )

target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${TORCH_LIBRARIES}
)

# set C++14 to compile
set_property(TARGET libtorch_inference PROPERTY CXX_STANDARD 14)
