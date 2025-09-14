cmake_minimum_required(VERSION 4.0)
project(example_project)

find_package(Torch REQUIRED)

# Option to enable/disable NNUE evaluation
option(NNUE_EVAL "Use NNUE evaluation" OFF)

if(NNUE_EVAL)
    add_definitions(-DNNUE_EVAL=1)
    message(STATUS "NNUE evaluation enabled")
else()
    add_definitions(-DNNUE_EVAL=0)
    message(STATUS "NNUE evaluation disabled")
endif()

# 1. Use C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Disable compiler-specific extensions

# 3. Include directories
include_directories(src)

# 4. Add source files
file(GLOB_RECURSE SOURCES src/game/*.cpp)

# 6. Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
# 7. Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -ffast-math")
# 8. Set optimization level
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

add_executable(uci ${SOURCES} "src/uci.cpp")
target_link_libraries(uci "${TORCH_LIBRARIES}")

# add_executable(make_tables ${SOURCES} "src/make_tables.cpp")
# add_executable(perft ${SOURCES} "src/perft.cpp")

# To build, run the following commands in the terminal:
# mkdir build
# cd build; cmake .. -DCMAKE_BUILD_TYPE; make; cd ..

# To run the executable:
# ./build/uci