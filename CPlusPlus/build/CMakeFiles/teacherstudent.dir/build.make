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
CMAKE_SOURCE_DIR = E:\GitHub\Code\CPlusPlus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = E:\GitHub\Code\CPlusPlus\build

# Include any dependencies generated for this target.
include CMakeFiles/teacherstudent.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/teacherstudent.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/teacherstudent.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/teacherstudent.dir/flags.make

CMakeFiles/teacherstudent.dir/codegen:
.PHONY : CMakeFiles/teacherstudent.dir/codegen

CMakeFiles/teacherstudent.dir/src/main.cpp.obj: CMakeFiles/teacherstudent.dir/flags.make
CMakeFiles/teacherstudent.dir/src/main.cpp.obj: CMakeFiles/teacherstudent.dir/includes_CXX.rsp
CMakeFiles/teacherstudent.dir/src/main.cpp.obj: E:/GitHub/Code/CPlusPlus/src/main.cpp
CMakeFiles/teacherstudent.dir/src/main.cpp.obj: CMakeFiles/teacherstudent.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=E:\GitHub\Code\CPlusPlus\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/teacherstudent.dir/src/main.cpp.obj"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/teacherstudent.dir/src/main.cpp.obj -MF CMakeFiles\teacherstudent.dir\src\main.cpp.obj.d -o CMakeFiles\teacherstudent.dir\src\main.cpp.obj -c E:\GitHub\Code\CPlusPlus\src\main.cpp

CMakeFiles/teacherstudent.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/teacherstudent.dir/src/main.cpp.i"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\GitHub\Code\CPlusPlus\src\main.cpp > CMakeFiles\teacherstudent.dir\src\main.cpp.i

CMakeFiles/teacherstudent.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/teacherstudent.dir/src/main.cpp.s"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\GitHub\Code\CPlusPlus\src\main.cpp -o CMakeFiles\teacherstudent.dir\src\main.cpp.s

CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.obj: CMakeFiles/teacherstudent.dir/flags.make
CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.obj: CMakeFiles/teacherstudent.dir/includes_CXX.rsp
CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.obj: E:/GitHub/Code/CPlusPlus/src/teacherStudent.cpp
CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.obj: CMakeFiles/teacherstudent.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=E:\GitHub\Code\CPlusPlus\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.obj"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.obj -MF CMakeFiles\teacherstudent.dir\src\teacherStudent.cpp.obj.d -o CMakeFiles\teacherstudent.dir\src\teacherStudent.cpp.obj -c E:\GitHub\Code\CPlusPlus\src\teacherStudent.cpp

CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.i"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E E:\GitHub\Code\CPlusPlus\src\teacherStudent.cpp > CMakeFiles\teacherstudent.dir\src\teacherStudent.cpp.i

CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.s"
	D:\VSCode\mingw64\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S E:\GitHub\Code\CPlusPlus\src\teacherStudent.cpp -o CMakeFiles\teacherstudent.dir\src\teacherStudent.cpp.s

# Object files for target teacherstudent
teacherstudent_OBJECTS = \
"CMakeFiles/teacherstudent.dir/src/main.cpp.obj" \
"CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.obj"

# External object files for target teacherstudent
teacherstudent_EXTERNAL_OBJECTS =

teacherstudent.exe: CMakeFiles/teacherstudent.dir/src/main.cpp.obj
teacherstudent.exe: CMakeFiles/teacherstudent.dir/src/teacherStudent.cpp.obj
teacherstudent.exe: CMakeFiles/teacherstudent.dir/build.make
teacherstudent.exe: CMakeFiles/teacherstudent.dir/linkLibs.rsp
teacherstudent.exe: CMakeFiles/teacherstudent.dir/objects1.rsp
teacherstudent.exe: CMakeFiles/teacherstudent.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=E:\GitHub\Code\CPlusPlus\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable teacherstudent.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\teacherstudent.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/teacherstudent.dir/build: teacherstudent.exe
.PHONY : CMakeFiles/teacherstudent.dir/build

CMakeFiles/teacherstudent.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\teacherstudent.dir\cmake_clean.cmake
.PHONY : CMakeFiles/teacherstudent.dir/clean

CMakeFiles/teacherstudent.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" E:\GitHub\Code\CPlusPlus E:\GitHub\Code\CPlusPlus E:\GitHub\Code\CPlusPlus\build E:\GitHub\Code\CPlusPlus\build E:\GitHub\Code\CPlusPlus\build\CMakeFiles\teacherstudent.dir\DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/teacherstudent.dir/depend

