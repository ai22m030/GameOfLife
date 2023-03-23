cmake_minimum_required(VERSION 3.23)
project(GameOfLife)

set(CMAKE_CXX_STANDARD 20)

add_executable(GameOfLife Timing.cpp main.cpp)
