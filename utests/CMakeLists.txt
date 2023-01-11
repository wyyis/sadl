cmake_minimum_required(VERSION 3.5)

project(test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -std=c++17 -ffast-math -Wall -fstrict-aliasing -mavx2")
file(GLOB HEADER_FILES ../sadl/*.h )

include_directories(..)
add_executable(test   test.cpp ${HEADER_FILES})


