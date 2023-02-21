add_library(procedural-placement-lib STATIC
        gl_context.cpp
        placement_result.cpp
        placement_pipeline.cpp
        disk_distribution_generator.cpp
        #reduction_kernel.cpp
        #generation_kernel.cpp
        #placement_pipeline_kernel.cpp
        kernels/compute_kernel.cpp
        kernels/generation_kernel.cpp
        kernels/evaluation_kernel.cpp
        kernels/indexation_kernel.cpp
        kernels/copy_kernel.cpp)

target_include_directories(procedural-placement-lib
        PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(procedural-placement-lib
        PUBLIC glm glutils)