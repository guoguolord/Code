cmake_minimum_required(VERSION 3.10)
project(BubbleSort)

set(CMAKE_CXX_STANDARD 11)

include_directories(E:/Code/CPlusPlus/Header)  # 添加头文件路径

add_executable(BubbleSort src/main.cpp src/bubbleSort.cpp)  # 添加所有源文件