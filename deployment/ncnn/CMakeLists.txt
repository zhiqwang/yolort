cmake_minimum_required(VERSION 3.14)

project(yolort_ncnn)

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

find_package(ncnn REQUIRED)

file(GLOB YOLO_SOURCE_FILES *.cpp)

add_executable(yolort_ncnn ${YOLO_SOURCE_FILES})

target_compile_features(yolort_ncnn PUBLIC cxx_range_for)

target_link_libraries(yolort_ncnn ncnn ${OpenCV_LIBS})
