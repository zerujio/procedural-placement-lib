include(CTest)

file(CREATE_LINK ${ASSETS_DIRECTORY} ${CMAKE_CURRENT_BINARY_DIR}/assets SYMBOLIC)

add_library(catch STATIC catch.hpp catch_main.cpp)
target_compile_definitions(catch PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)

add_executable(Tests tests.cpp)
target_link_libraries(Tests catch procedural-placement-lib glad glfw stb_image)
add_test(NAME Tests
        COMMAND Tests)