add_executable(01-basic-placement 01-basic-placement.cpp)
target_link_libraries(01-basic-placement procedural-placement-lib simple-renderer glfw stb_image)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/heightmap.png ${CMAKE_CURRENT_BINARY_DIR}/heightmap.png SYMBOLIC)