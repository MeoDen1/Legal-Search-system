cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(database_engine)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Python_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin/python")
set(pybind11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")
find_package(Torch REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

file(GLOB SOURCES "src/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/binding.cpp")

# Executable main
add_executable(database_engine_bin main.cpp ${SOURCES})
target_include_directories(database_engine_bin PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(database_engine_bin PRIVATE "${TORCH_LIBRARIES}" yaml-cpp)

# Python module
pybind11_add_module(database_engine src/binding.cpp ${SOURCES})
target_include_directories(database_engine PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(database_engine PRIVATE "${TORCH_LIBRARIES}" yaml-cpp)
