cmake_minimum_required(VERSION 2.8)

project(resnet)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories("/usr/local/cuda-10.2/include")
link_directories("/usr/local/cuda-10.2/lib64")

include_directories("/home/pcl/TensorRT-7.1.3.4/include")
link_directories("/home/pcl/TensorRT-7.1.3.4/lib")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(resnet ${PROJECT_SOURCE_DIR}/resnet.cpp)
target_link_libraries(resnet nvinfer)
target_link_libraries(resnet cudart)
target_link_libraries(resnet ${OpenCV_LIBS})

add_definitions(-O2 -pthread)
