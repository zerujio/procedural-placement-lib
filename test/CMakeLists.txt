include(CTest)

file(CREATE_LINK ${ASSETS_DIRECTORY} ${CMAKE_CURRENT_BINARY_DIR}/assets SYMBOLIC)

add_library(catch STATIC catch.hpp catch_main.cpp)
target_compile_definitions(catch PUBLIC CATCH_CONFIG_ENABLE_BENCHMARKING)

add_executable(Tests tests.cpp)
target_link_libraries(Tests catch procedural-placement-lib glad glfw stb_image)

option(PLACEMENT_BENCHMARK_MULTITHREAD "Use a multi-thread implementation for comparison." ON)
if (PLACEMENT_BENCHMARK_MULTITHREAD)
    target_link_libraries(Tests tbb)
    target_compile_definitions(Tests PRIVATE PLACEMENT_BENCHMARK_MULTITHREAD)
endif()

add_test(NAME Tests
        COMMAND Tests)