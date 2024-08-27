cmake_minimum_required(VERSION 3.21)
project(FinalAssignment)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(FinalAssignment
    src/main.cpp
    src/Graph.cpp
    src/Algorithms.cpp
        include/Graph.h
        include/Algorithms.h)
