add_library(procedural-placement-lib STATIC
        placement.cpp
        generation_kernel.cpp
        reduction_kernel.cpp placement_pipeline.cpp placement_pipeline.hpp)

target_include_directories(procedural-placement-lib
        PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(procedural-placement-lib
        PUBLIC glm
        PRIVATE glutils)