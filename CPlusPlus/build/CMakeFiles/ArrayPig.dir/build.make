# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = D:\VSCode\cmake\cmake_3.31.0\bin\cmake.exe

# The command to remove a file.
RM = D:\VSCode\cmake\cmake_3.31.0\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = E:\Code\CPlusPlus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\Code\CPlusPlus\build

# Include any dependencies generated for this target.
include CMakeFiles/ArrayPig.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ArrayPig.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ArrayPig.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ArrayPig.dir/flags.make

CMakeFiles/ArrayPig.dir/codegen:
.PHONY : CMakeFiles/ArrayPig.dir/codegen

CMakeFiles/ArrayPig.dir/src/main.cpp.obj: CMakeFiles/ArrayPig.dir/flags.make
CMakeFiles/ArrayPig.dir/src/main.cpp.obj: CMakeFiles/ArrayPig.dir/includes_CXX.rsp
CMakeFiles/ArrayPig.dir/src/main.cpp.obj: E:/Code/CPlusPlus/src/main.cpp
CMakeFiles/ArrayPig.dir/src/main.cpp.obj: CMakeFiles/ArrayPig.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=E:\Code\CPlusPlus\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ArrayPig.dir/src/main.cpp.obj"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ArrayPig.dir/src/main.cpp.obj -MF CMakeFiles\ArrayPig.dir\src\main.cpp.obj.d -o CMakeFiles\ArrayPig.dir\src\main.cpp.obj -c E:\Code\CPlusPlus\src\main.cpp

CMakeFiles/ArrayPig.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ArrayPig.dir/src/main.cpp.i"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\Code\CPlusPlus\src\main.cpp > CMakeFiles\ArrayPig.dir\src\main.cpp.i

CMakeFiles/ArrayPig.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ArrayPig.dir/src/main.cpp.s"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\Code\CPlusPlus\src\main.cpp -o CMakeFiles\ArrayPig.dir\src\main.cpp.s

CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.obj: CMakeFiles/ArrayPig.dir/flags.make
CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.obj: CMakeFiles/ArrayPig.dir/includes_CXX.rsp
CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.obj: E:/Code/CPlusPlus/src/arrayPig.cpp
CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.obj: CMakeFiles/ArrayPig.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=E:\Code\CPlusPlus\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.obj"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.obj -MF CMakeFiles\ArrayPig.dir\src\arrayPig.cpp.obj.d -o CMakeFiles\ArrayPig.dir\src\arrayPig.cpp.obj -c E:\Code\CPlusPlus\src\arrayPig.cpp

CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.i"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\Code\CPlusPlus\src\arrayPig.cpp > CMakeFiles\ArrayPig.dir\src\arrayPig.cpp.i

CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.s"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\Code\CPlusPlus\src\arrayPig.cpp -o CMakeFiles\ArrayPig.dir\src\arrayPig.cpp.s

# Object files for target ArrayPig
ArrayPig_OBJECTS = \
"CMakeFiles/ArrayPig.dir/src/main.cpp.obj" \
"CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.obj"

# External object files for target ArrayPig
ArrayPig_EXTERNAL_OBJECTS =

ArrayPig.exe: CMakeFiles/ArrayPig.dir/src/main.cpp.obj
ArrayPig.exe: CMakeFiles/ArrayPig.dir/src/arrayPig.cpp.obj
ArrayPig.exe: CMakeFiles/ArrayPig.dir/build.make
ArrayPig.exe: CMakeFiles/ArrayPig.dir/linkLibs.rsp
ArrayPig.exe: CMakeFiles/ArrayPig.dir/objects1.rsp
ArrayPig.exe: CMakeFiles/ArrayPig.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=E:\Code\CPlusPlus\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ArrayPig.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\ArrayPig.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ArrayPig.dir/build: ArrayPig.exe
.PHONY : CMakeFiles/ArrayPig.dir/build

CMakeFiles/ArrayPig.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\ArrayPig.dir\cmake_clean.cmake
.PHONY : CMakeFiles/ArrayPig.dir/clean

CMakeFiles/ArrayPig.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\Code\CPlusPlus E:\Code\CPlusPlus E:\Code\CPlusPlus\build E:\Code\CPlusPlus\build E:\Code\CPlusPlus\build\CMakeFiles\ArrayPig.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/ArrayPig.dir/depend
