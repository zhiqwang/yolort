cmake_minimum_required(VERSION 2.6)
project(yolov6) # 1
add_definitions(-DAPI_EXPORTS) 
set(OpenCV_DIR "D:\\OpenCV4.1.1\\opencv\\build")  #2
set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}\\include) #3 
set(OpenCV_LIB_DIRS ${OpenCV_DIR}\\x64\\vc15\\lib) #4
set(TRT_DIR "D:\\TensorRT-8.4.0.6")  #3
set(Dirent_INCLUDE_DIRS "D:\\dirent-1.23.2\\include") #10

add_definitions(-std=c++11)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# setup CUDA
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

####
enable_language(CUDA)  # add this line, then no need to setup cuda path in vs
####
include_directories(${PROJECT_SOURCE_DIR}include)
include_directories(${TRT_DIR}\\include)

# -D_MWAITXINTRIN_H_INCLUDED for solving error: identifier "__builtin_ia32_mwaitx" is undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -D_MWAITXINTRIN_H_INCLUDED")

# setup opencv
find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIRS}) #15
include_directories(${Dirent_INCLUDE_DIRS}) #16
link_directories(${TRT_DIR}\\lib)

#ADD_LIBRARY(yolov6 SHARED main.cpp)
add_executable(yolov6 main.cpp)
target_include_directories(
    yolov6 PUBLIC ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(yolov6 nvinfer)
target_link_libraries(yolov6 nvinfer_plugin)
target_link_libraries(yolov6 nvparsers)
target_link_libraries(yolov6 nvonnxparser)
target_link_libraries(yolov6 ${CUDA_LIBRARIES})
target_link_libraries(yolov6 ${OpenCV_LIBS})
