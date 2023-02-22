#include "placement/kernel/copy_kernel.hpp"

static constexpr auto source_string = R"gl(
#version 430 core

#define NULL_CLASS_INDEX 0xFFffFFff

layout(local_size_x = 64) in;

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

layout(std430) restrict readonly
buffer IndexBuffer
{
        uint array[];
} b_index;

layout(std430) restrict writeonly
buffer OutputBuffer
{
        Candidate array[];
} b_output;

layout(std430) restrict readonly
buffer CountBuffer
{
    uint array[];
} b_count;

void main()
{
    const uint candidate_index = gl_GlobalInvocationID.x;
    if (candidate_index >= b_candidate.array.length())
        return;

    const Candidate candidate = b_candidate.array[candidate_index];
    if (candidate.class_index == NULL_CLASS_INDEX)
        return;

    const uint copy_index = b_index.array[candidate_index];

    uint index_offset = 0;
    for (uint class_index = 0; class_index < candidate.class_index; class_index++)
        index_offset += b_count.array[class_index];

    b_output.array[copy_index + index_offset] = candidate;
}
)gl";

namespace placement {
CopyKernel::CopyKernel() : ComputeKernel(source_string) {}
} // placement