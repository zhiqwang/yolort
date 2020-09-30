cmake_minimum_required(VERSION 3.14)
project(libtorch-inference)

FIND_PACKAGE( Torch REQUIRED )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")

FIND_PACKAGE( OpenCV REQUIRED )
if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

include_directories( ${PROJECT_SOURCE_DIR}/include )

FILE(GLOB YOLO_SOURCE_FILES src/*.cpp)

ADD_EXECUTABLE( ${CMAKE_PROJECT_NAME} ${YOLO_SOURCE_FILES} )

TARGET_LINK_LIBRARIES(
        ${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
)
