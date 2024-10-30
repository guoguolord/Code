cmake_minimum_required(VERSION 3.10)
project(ArrayStruct)

set(CMAKE_CXX_STANDARD 11)
# include_directories(E:/GitHub/Code/CPlusPlus/Header)  # 添加头文件路径
add_executable(ArrayStruct src/main.cpp src/twoArrayStruct.cpp)  # 添加所有源文件

target_include_directories(ArrayStruct PRIVATE Header)