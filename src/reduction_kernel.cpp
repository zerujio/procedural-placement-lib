#include "placement/reduction_kernel.hpp"

#include "glutils/buffer.hpp"
#include "gl_context.hpp"

namespace placement {

     const std::string ReductionKernel::source_string = std::string() + R"glsl(
#version 430 core

layout(local_size_x = 16) in;

layout(std430, binding=0)
restrict coherent
buffer )glsl" + ReductionKernel::s_position_ssb_name + R"glsl(
{
    vec3 position_buffer[];
};

layout(std430, binding=1)
restrict coherent
buffer )glsl" + ReductionKernel::s_index_ssb_name + R"glsl(
{
    uint index_buffer[];
};

struct Candidate
{
    vec3 position;
    uint is_valid;
};

void main()
{
    const uint invocation_index = gl_GlobalInvocationID.x * 2;
    const Candidate c0 = {position_buffer[invocation_index], index_buffer[invocation_index]};
    const Candidate c1 = {position_buffer[invocation_index + 1], index_buffer[invocation_index + 1]};

    for (uint group_size = 1; group_size < index_buffer.length(); group_size <<= 1)
    {
        const uint group_index = (gl_GlobalInvocationID.x / group_size) * 2 + 1;
        const uint base_index = group_index * group_size;
        const uint write_index = base_index + gl_GlobalInvocationID.x % group_size;
        const uint read_index = base_index - 1;

        if (write_index < index_buffer.length())
            index_buffer[write_index] += index_buffer[read_index];

        memoryBarrierBuffer();
    }

    if (c0.is_valid == 1)
    {
        const uint c_index = index_buffer[invocation_index];
        position_buffer[c_index] = c0.position;
    }

    if (c1.is_valid == 1)
    {
        const uint c_index = index_buffer[invocation_index + 1];
        position_buffer[c_index] = c1.position;
    }
}
)glsl";

    using namespace glutils;

    ReductionKernel::ReductionKernel() : PlacementPipelineKernel(source_string) {}

    std::size_t ReductionKernel::operator()(glutils::BufferOffset position_buffer, glutils::BufferOffset index_buffer,
                                            std::size_t count) const
    {
        bindPositionBuffer(position_buffer, count);
        bindIndexBuffer(index_buffer, count);

        index_buffer.offset += static_cast<GLintptr>(dispatchCompute(count));

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        uint reduced_count = 0;
        index_buffer.read(sizeof(GLuint), &reduced_count);

        return reduced_count;
    }

    auto ReductionKernel::dispatchCompute(std::size_t count) const -> std::size_t
    {
        const unsigned int num_work_groups = count / 32 + (count % 32 != 0); // one work group for every two 4x4 blocks
        useProgram();
        gl.DispatchCompute(num_work_groups, 1, 1);
        return (count - 1) * sizeof (GLuint);
    }

} // placement