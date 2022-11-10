#include "placement/reduction_kernel.hpp"

#include "glutils/buffer.hpp"
#include "gl_context.hpp"

#include <sstream>

namespace placement {

     const std::string ReductionKernel::source_string = (std::stringstream() << R"glsl(
#version 430 core

layout(local_size_x = )glsl" << ReductionKernel::work_group_size << R"glsl() in;

layout(std430, binding=)glsl" << ReductionKernel::default_position_ssb_binding << R"glsl()
restrict coherent
buffer )glsl" << ReductionKernel::s_position_ssb_name << R"glsl(
{
    vec3 position_buffer[];
};

layout(std430, binding=)glsl" << ReductionKernel::default_index_ssb_binding << R"glsl()
restrict coherent
buffer )glsl" << ReductionKernel::s_index_ssb_name << R"glsl(
{
    uint index_buffer[];
};

struct Candidate
{
    vec3 position;
    uint index;
};

void main()
{
    const uint invocation_index = gl_GlobalInvocationID.x * 2;
    const Candidate c0 = {position_buffer[invocation_index], index_buffer[invocation_index]};
    const Candidate c1 = {position_buffer[invocation_index + 1], index_buffer[invocation_index + 1]};

    for (uint group_size = 1; group_size < index_buffer.length(); group_size <<= 1)
    {
        memoryBarrierBuffer();

        const uint group_index = (gl_GlobalInvocationID.x / group_size) * 2 + 1;
        const uint base_index = group_index * group_size;
        const uint write_index = base_index + gl_GlobalInvocationID.x % group_size;
        const uint read_index = base_index - 1;

        if (write_index < index_buffer.length())
            index_buffer[write_index] += index_buffer[read_index];
    }

    if (c0.index == 1)
    {
        const uint c_index = index_buffer[invocation_index] - 1;
        position_buffer[c_index] = c0.position;
    }

    if (c1.index == 1)
    {
        const uint c_index = index_buffer[invocation_index + 1] - 1;
        position_buffer[c_index] = c1.position;
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