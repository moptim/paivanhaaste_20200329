cmake_minimum_required(VERSION 3.10)

project(ph)

add_executable(ph main.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(glfw3 3.2 REQUIRED)
target_link_libraries(ph glfw)

find_package(GLEW REQUIRED)
target_link_libraries(ph GLEW::GLEW)

find_package(OpenGL REQUIRED)
target_link_libraries(ph OpenGL::GL)
