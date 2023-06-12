cmake_minimum_required(VERSION 3.5)

project(test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
file(GLOB HEADER_FILES ../sadl/*.h )

include_directories(..)
add_executable(test   test.cpp ${HEADER_FILES})

if( UNIX OR MINGW )
 set(CMAKE_CXX_FLAGS "-ffast-math -Wall -fstrict-aliasing") 
 set_target_properties(test  PROPERTIES COMPILE_FLAGS "-mavx2  -mavx512f -mavx512bw" )
endif()


if( MSVC )
  set_property(test PROPERTIES COMPILE_FLAGS "/arch:AVX2")
endif()
