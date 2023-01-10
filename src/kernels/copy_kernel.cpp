#include "copy_kernel.hpp"

static constexpr auto source_string = R"gl(
#version 430 core

#define NULL_CLASS_INDEX 0xffffffff

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

void main()
{
    const uint candidate_index = gl_GlobalInvocationID.x;
    if (candidate_index < b_candidate.array.length())
        return;

    const Candidate candidate = b_candidate.array[candidate_index];
    if (candidate.class_index == NULL_CLASS_INDEX)
        return;

    const uint copy_index = b_index.array[candidate_index];
    b_output.array[copy_index] = candidate;
}
)gl";

namespace placement {
CopyKernel::CopyKernel() : ComputeKernel(source_string),
                           m_candidate_buffer(*this, "CandidateBuffer"),
                           m_index_buffer(*this, "IndexBuffer"),
                           m_output_buffer(*this, "OutputBuffer")
{

}
} // placement