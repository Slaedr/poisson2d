cmake_minimum_required(VERSION 2.8)

# Pass -DPROF=1 to profile
# Pass -DSSE=1 to compile with SSE 4.2 instructions; ignored when compiling for KNC.
# Pass -DAVX=1 to compile with AVX instructions

project (poisson)

set (CMAKE_CXX_FLAGS "-std=c++14")
set (CMAKE_CXX_FLAGS_DEBUG " ")
set (CMAKE_CXX_FLAGS_RELEASE " ")

# set compile options
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopt-info-loop-inline-vec-optimized-missed=optimizations.info")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -qopt-report=2")
endif()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")
message(STATUS "Compiling for x86_64")
if(SSE)
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -msse4.2")
	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
		message(STATUS "Compiling with SSE4.2")
	endif()
elseif(AVX)
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx")
	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
		message(STATUS "Compiling with AVX")
	endif()
elseif(KNL)
	set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=knl")
	if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
		message(STATUS "Compiling with AVX")
	endif()
endif()

if(PROF)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
	message(STATUS "Profile build")
endif()

# Eigen
#include_directories($ENV{EIGEN_DIR})

# ---------------------------------------------------------------------------- #

# flags and stuff

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -ggdb -DDEBUG=1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -UDEBUG -DNDEBUG -Wall")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode")
else()
	message(STATUS "Release mode")
endif()

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# libraries to be compiled
add_library(poissonlib poisson.cpp gradientschemes.cpp afsolver.cpp aoutput_struct.cpp structmesh2d.cpp)

# for the final executable(s)

add_executable(poisson poisson_main.cpp)
target_link_libraries(poisson poissonlib)

# Tags

set_source_files_properties(tags PROPERTIES GENERATED true)
add_custom_target ( tags
    COMMAND ctags -R --c++-kinds=+pf --extra=+q . 
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} 
  )

find_program(CTAGS_PATH ctags)
if(CTAGS_PATH)
    message(STATUS "Found ctags: ${CTAGS_PATH}")
    add_dependencies (poisson tags)
else()
	message(STATUS "ctags not found! Tags will not be generated.")
endif(CTAGS_PATH)

