#include "indexation_kernel.hpp"

static constexpr auto source_string = R"gl(
#version 450 core

layout(local_size_x = 32) in;

uniform uint u_class_index;

struct Candidate
{
    vec3 position;
    uint class_index;
};

layout(std430) restrict readonly
buffer CandidateBuffer
{
    Candidate array[];
} b_candidate;

// read from candidate buffer with bounds checking
uint readClassIndex(uint index)
{
    return index < b_candidate.array.length() ? b_candidate.array[index] : 0;
}

layout(std430) restrict
buffer CountBuffer
{
    uint total;
    uint by_class[];
} b_count;

layout(std430) restrict writeonly
buffer IndexBuffer
{
    uint array[];
} b_index;

/// write to index array with bounds checking
void writeIndex(uint array_index, uint value)
{
    if (array_index < b_index.array.length())
b_index.array[array_index] = value;
}

shared uint s_index_array[2 * gl_WorkGroupSize.x];
shared uint s_index_offset;

void main()
{
    const uint local_array_index_0 = gl_LocalInvocationID.x;
    const uint local_array_index_1 = local_array_index_0 + gl_WorkGroupSize.x;

    const uint global_array_index_0 = gl_WorkGroupID.x * 2 * gl_WorkGroupSize.x + local_array_index_0;
    const uint global_array_index_1 = global_array_index_0 + gl_WorkGroupSize.x;

    const uint class_index_0 = readClassIndex(global_array_index_0);
    const uint class_index_1 = readClassIndex(global_array_index_1);

    s_index_array[local_array_index_0] = class_index_0 == u_class_index;
    s_index_array[local_array_index_1] = class_index_1 == u_class_index;

    barrier();
    memoryBarrierShared();

    for (uint group_size = 1; group_size < 2 * gl_workgroupsize.x; group_size <<= 1)
    {
        const uint group_index = (gl_LocalInvocationID.x / group_size) * 2 + 1;
        const uint base_index = group_index * group_size;
        const uint write_index = base_index + gl_LocalInvocationID.x % group_size;
        const uint read_index = base_index - 1;

        s_index_array[write_index] += s_index_array[read_index];

        barrier();
        memoryBarrierShared();
    }

    if (gl_LocalInvocationID.x == gl_WorkGroupSize.x - 1)
    {
        const uint local_sum = s_index_array[2 * gl_WorkGroupSize.x - 1];
        atomicAdd(b_count.by_class[class_index], local_sum);
        s_index_offset = atomicAdd(b_count.total, local_sum);
    }

    barrier();
    memoryBarrierShared();

    if (class_index_0 == u_class_index)
        writeIndex(global_array_index_0, s_index_array[local_array_index_0] + s_index_offset - 1);

    if (class_index_1 == u_class_index)
        writeIndex(global_array_index_1, s_index_array[local_array_index_1] + s_index_offset - 1);
}
)gl";

namespace placement {
IndexationKernel::IndexationKernel() : ComputeKernel(source_string),
                                       m_class_index(*this, "u_class_index"),
                                       m_candidate_buffer(*this, "CandidateBuffer"),
                                       m_count_buffer(*this, "CountBuffer"),
                                       m_index_buffer(*this, "IndexBuffer")
{}
} // placement