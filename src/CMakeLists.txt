add_library(procedural-placement-lib STATIC
        gl_context.cpp
        generation_kernel.cpp
        reduction_kernel.cpp
        placement_pipeline.cpp
        compute_kernel.cpp
        placement_pipeline_kernel.cpp
        disk_distribution_generator.cpp
        kernels/generation_kernel.cpp ../include/placement/kernel/generation_kernel.hpp kernels/evaluation_kernel.cpp ../include/placement/kernel/evaluation_kernel.hpp kernels/indexation_kernel.cpp ../include/placement/kernel/indexation_kernel.hpp kernels/copy_kernel.cpp ../include/placement/kernel/copy_kernel.hpp)

target_include_directories(procedural-placement-lib
        PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(procedural-placement-lib
        PUBLIC glm glutils)