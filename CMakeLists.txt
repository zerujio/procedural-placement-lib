cmake_minimum_required(VERSION 3.20)
project(ProceduralPlacementLib)

set(ASSETS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/assets)

add_subdirectory(lib)
add_subdirectory(src)
add_subdirectory(example)
add_subdirectory(test)