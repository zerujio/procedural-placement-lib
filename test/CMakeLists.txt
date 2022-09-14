include(CTest)

file(CREATE_LINK ${ASSETS_DIRECTORY} ${CMAKE_CURRENT_BINARY_DIR}/assets SYMBOLIC)

add_executable(Tests tests.cpp)
target_link_libraries(Tests procedural-placement-lib glad glfw stb_image)
add_test(NAME Tests
        COMMAND Tests)