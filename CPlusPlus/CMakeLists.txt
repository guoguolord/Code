cmake_minimum_required(VERSION 3.10)
project(HERO)

set(CMAKE_CXX_STANDARD 11)

add_executable(HERO src/main.cpp)  # 添加所有源文件
target_include_directories(HERO PRIVATE include)
