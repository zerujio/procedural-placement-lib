add_executable(01-basic-placement 01-basic-placement.cpp)
target_link_libraries(01-basic-placement procedural-placement-lib glutils glfw stb_image)

file(CREATE_LINK ${ASSETS_DIRECTORY} ${CMAKE_CURRENT_BINARY_DIR}/assets SYMBOLIC)