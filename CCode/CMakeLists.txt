cmake_minimum_required(VERSION 3.20)
project(CCode C)

# 设置C的标准
set(CMAKE_C_STANDARD 11)


# 寻找src下的所有文件
aux_source_directory(src SRC_FILES)
list(REMOVE_ITEM SRC_FILES ReadWave.c)

# 添加所有文件进行编译
add_executable(CCode ${SRC_FILES})

# 包含头文件目录，include目录
target_include_directories(CCode PRIVATE ./include)
