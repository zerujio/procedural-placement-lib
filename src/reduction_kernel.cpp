#include "placement/reduction_kernel.hpp"

#include "glutils/buffer.hpp"
#include "gl_context.hpp"

#include <sstream>

namespace placement {

     const std::string ReductionKernel::source_string = (std::stringstream() << R"glsl(
#version 430 core

layout(local_size_x = )glsl" << ReductionKernel::work_group_size << R"glsl() in;

struct Candidate
{
    vec3 position;
    uint index;
};

layout(std430, binding=)glsl" << ReductionKernel::default_ssb_binding << R"glsl()
restrict coherent
buffer )glsl" << ReductionKernel::s_ssb_name << R"glsl(
{
    Candidate candidate_buffer[];
};

void main()
{
    const uint invocation_index = gl_GlobalInvocationID.x * 2;
    const Candidate c0 = candidate_buffer[invocation_index];
    const Candidate c1 = candidate_buffer[invocation_index + 1];

    for (uint group_size = 1; group_size < candidate_buffer.length(); group_size <<= 1)
    {
        const uint group_index = (gl_GlobalInvocationID.x / group_size) * 2 + 1;
        const uint base_index = group_index * group_size;
        const uint write_index = base_index + gl_GlobalInvocationID.x % group_size;
        const uint read_index = base_index - 1;

        if (write_index < candidate_buffer.length())
            candidate_buffer[write_index].index += candidate_buffer[read_index].index;

        memoryBarrierBuffer();
    }

    if (c0.index == 1)
    {
        const uint c_index = candidate_buffer[invocation_index].index - 1;
        candidate_buffer[c_index].position = c0.position;
    }

    if (c1.index == 1)
    {
        const uint c_index = candidate_buffer[invocation_index + 1].index - 1;
        candidate_buffer[c_index].position = c1.position;
    }
}
)glsl").str();

    using namespace glutils;

    ReductionKernel::ReductionKernel() : PlacementPipelineKernel(source_string) {}

    auto ReductionKernel::operator() (std::size_t num_work_groups) const -> std::size_t
    {
        useProgram();
        gl.DispatchCompute(num_work_groups, 1, 1);
        return (work_group_size * num_work_groups - 1) * sizeof (GLuint);
    }

} // placement