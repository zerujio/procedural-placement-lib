#include "placement/kernel/indexation_kernel.hpp"

static constexpr auto source_string = R"gl(
#version 450 core

#define INVALID_INDEX 0xFFffFFff

layout(local_size_x = 32) in;

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
    return index < b_candidate.array.length() ? b_candidate.array[index].class_index : INVALID_INDEX;
}

layout(std430) restrict
buffer CountBuffer
{
    uint array[];
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

void initLocalIndexArray(uvec2 array_index, uvec2 value)
{
    s_index_array[array_index.x] = value.x;
    s_index_array[array_index.y] = value.y;
}

void addUpLocalIndexArray()
{
    for (uint group_size = 1; group_size < 2 * gl_WorkGroupSize.x; group_size <<= 1)
    {
        const uint group_index = (gl_LocalInvocationID.x / group_size) * 2 + 1;
        const uint base_index = group_index * group_size;
        const uint write_index = base_index + gl_LocalInvocationID.x % group_size;
        const uint read_index = base_index - 1;

        s_index_array[write_index] += s_index_array[read_index];

        barrier();
        memoryBarrierShared();
    }
}

uint atomicAddToClassCount(uint class_index)
{
    const uint local_sum = s_index_array[2 * gl_WorkGroupSize.x - 1];
    return atomicAdd(b_count.array[class_index], local_sum);
}

void main()
{
    const uvec2 local_index = {gl_LocalInvocationID.x, gl_LocalInvocationID.x + gl_WorkGroupSize.x};
    const uvec2 global_index = uvec2(gl_WorkGroupID.x * 2 * gl_WorkGroupSize.x) + local_index;
    const uvec2 class_index = {readClassIndex(global_index.x), readClassIndex(global_index.y)};

    uvec2 result_value = uvec2(INVALID_INDEX);

    for (uint i = 0; i < b_count.array.length(); i++)
    {
        initLocalIndexArray(local_index, uvec2(equal(class_index, uvec2(i))));

        barrier();
        memoryBarrierShared();

        addUpLocalIndexArray();

        if (gl_LocalInvocationIndex == 0)
        s_index_offset = atomicAddToClassCount(i);

        barrier();
        memoryBarrierShared();

        result_value.x = class_index.x == i ? s_index_array[local_index.x] + s_index_offset - 1 : result_value.x;
        result_value.y = class_index.y == i ? s_index_array[local_index.y] + s_index_offset - 1 : result_value.y;
    }

    writeIndex(global_index.x, result_value.x);
    writeIndex(global_index.y, result_value.y);
}
)gl";

namespace placement {
IndexationKernel::IndexationKernel() : ComputeKernel(source_string) {}
} // placement