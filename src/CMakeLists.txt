add_library(procedural-placement-lib STATIC
        gl_context.cpp
        generation_kernel.cpp
        reduction_kernel.cpp
        placement_pipeline.cpp
        compute_kernel.cpp
        placement_pipeline_kernel.cpp
        disk_distribution_generator.cpp)

target_include_directories(procedural-placement-lib
        PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(procedural-placement-lib
        PUBLIC glm glutils)