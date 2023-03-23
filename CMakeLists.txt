cmake_minimum_required(VERSION 3.23)
project(GameOfLife)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")

add_executable(GameOfLife Timing.cpp main.cpp)

find_package(OpenMP QUIET)

target_include_directories(GameOfLife PUBLIC OpenMP::OpenMP_CXX_INCLUDE_DIRS)

if (OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
endif ()