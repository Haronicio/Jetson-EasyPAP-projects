# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake-3.27.4/bin/cmake

# The command to remove a file.
RM = /opt/cmake-3.27.4/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src

# Include any dependencies generated for this target.
include CMakeFiles/murb-implem-lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/murb-implem-lib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/murb-implem-lib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/murb-implem-lib.dir/flags.make

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.o: CMakeFiles/murb-implem-lib.dir/flags.make
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.o: murb/implem/SimulationNBodyGPUOpenCL.cpp
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.o: CMakeFiles/murb-implem-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.o -MF CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.o.d -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.o -c /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyGPUOpenCL.cpp

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyGPUOpenCL.cpp > CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.i

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyGPUOpenCL.cpp -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.s

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.o: CMakeFiles/murb-implem-lib.dir/flags.make
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.o: murb/implem/SimulationNBodyHetero.cpp
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.o: CMakeFiles/murb-implem-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.o -MF CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.o.d -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.o -c /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyHetero.cpp

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyHetero.cpp > CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.i

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyHetero.cpp -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.s

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.o: CMakeFiles/murb-implem-lib.dir/flags.make
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.o: murb/implem/SimulationNBodyNaive.cpp
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.o: CMakeFiles/murb-implem-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.o -MF CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.o.d -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.o -c /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyNaive.cpp

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyNaive.cpp > CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.i

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyNaive.cpp -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.s

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.o: CMakeFiles/murb-implem-lib.dir/flags.make
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.o: murb/implem/SimulationNBodyOpenMP.cpp
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.o: CMakeFiles/murb-implem-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.o -MF CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.o.d -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.o -c /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyOpenMP.cpp

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyOpenMP.cpp > CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.i

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyOpenMP.cpp -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.s

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.o: CMakeFiles/murb-implem-lib.dir/flags.make
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.o: murb/implem/SimulationNBodyOptim.cpp
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.o: CMakeFiles/murb-implem-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.o -MF CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.o.d -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.o -c /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyOptim.cpp

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyOptim.cpp > CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.i

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodyOptim.cpp -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.s

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.o: CMakeFiles/murb-implem-lib.dir/flags.make
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.o: murb/implem/SimulationNBodySIMD.cpp
CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.o: CMakeFiles/murb-implem-lib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.o -MF CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.o.d -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.o -c /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodySIMD.cpp

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodySIMD.cpp > CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.i

CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/murb/implem/SimulationNBodySIMD.cpp -o CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.s

murb-implem-lib: CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyGPUOpenCL.cpp.o
murb-implem-lib: CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyHetero.cpp.o
murb-implem-lib: CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyNaive.cpp.o
murb-implem-lib: CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOpenMP.cpp.o
murb-implem-lib: CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodyOptim.cpp.o
murb-implem-lib: CMakeFiles/murb-implem-lib.dir/murb/implem/SimulationNBodySIMD.cpp.o
murb-implem-lib: CMakeFiles/murb-implem-lib.dir/build.make
.PHONY : murb-implem-lib

# Rule to build all files generated by this target.
CMakeFiles/murb-implem-lib.dir/build: murb-implem-lib
.PHONY : CMakeFiles/murb-implem-lib.dir/build

CMakeFiles/murb-implem-lib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/murb-implem-lib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/murb-implem-lib.dir/clean

CMakeFiles/murb-implem-lib.dir/depend:
	cd /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src /home/sesi/Documents/progpar/project/Move-U-r-Body/murb-se/src/CMakeFiles/murb-implem-lib.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/murb-implem-lib.dir/depend

