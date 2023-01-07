#include "placement/reduction_kernel.hpp"

#include "glutils/buffer.hpp"
#include "gl_context.hpp"

#include <sstream>

namespace placement {

     const std::string IndexAssignmentKernel::s_source_string = (std::stringstream() << R"glsl(
#version 450 core

layout(local_size_x = )glsl" << IndexAssignmentKernel::s_work_group_size << R"glsl() in;

struct Candidate
{
    vec3 position;
    bool valid;
};

layout(std430, binding = 0) restrict readonly
buffer )glsl" << ReductionKernel::s_candidate_ssb_name << R"glsl(
{
    Candidate candidate_array[];
};

layout(std430, binding = 1) restrict
buffer )glsl" << ReductionKernel::s_index_ssb_name << R"glsl(
{
    uint index_sum;
    uint index_array[];
};

shared uint index_cache[2 * gl_WorkGroupSize.x];
shared uint index_offset;

// read validity flag from candidate with bounds checking
uint readValidity(uint index)
{
    return index < candidate_array.length() ? uint(candidate_array[index].valid) : 0;
}

// write index to index buffer with bounds checking
void writeIndex(uint array_index, uint value)
{
    if (array_index < index_array.length())
        index_array[array_index] = value;
}

void main()
{
    const uint local_array_index0 = gl_LocalInvocationID.x;
    const uint local_array_index1 = local_array_index0 + gl_WorkGroupSize.x;

    const uint global_array_index0 = gl_WorkGroupID.x * 2 * gl_WorkGroupSize.x + local_array_index0;
    const uint global_array_index1 = global_array_index0 + gl_WorkGroupSize.x;

    const uint valid0 = readValidity(global_array_index0);
    const uint valid1 = readValidity(global_array_index1);

    index_cache[local_array_index0] = valid0;
    index_cache[local_array_index1] = valid1;

    barrier();
    memoryBarrierShared();

    for (uint group_size = 1; group_size < 2 * gl_WorkGroupSize.x; group_size <<= 1)
    {
        const uint group_index = (gl_LocalInvocationID.x / group_size) * 2 + 1;
        const uint base_index = group_index * group_size;
        const uint write_index = base_index + gl_LocalInvocationID.x % group_size;
        const uint read_index = base_index - 1;

        index_cache[write_index] += index_cache[read_index];

        barrier();
        memoryBarrierShared();
    }

    if (gl_LocalInvocationIndex.x == gl_WorkGroupSize.x - 1)
        index_offset = atomicAdd(index_sum, index_cache[2 * gl_WorkGroupSize.x - 1]);

    barrier();
    memoryBarrierShared();

    writeIndex(global_array_index0, (index_offset + index_cache[local_array_index0]) * valid0 - 1);
    writeIndex(global_array_index1, (index_offset + index_cache[local_array_index1]) * valid1 - 1);

}
)glsl").str();

    using namespace GL;

    IndexAssignmentKernel::IndexAssignmentKernel() : ReductionKernel(s_source_string) {}

    void IndexAssignmentKernel::dispatchCompute(std::size_t candidate_count) const
    {
        useProgram();
        gl.DispatchCompute(m_calculateNumWorkGroups(candidate_count), 1, 1);
    }

    auto IndexAssignmentKernel::m_calculateNumWorkGroups(const std::size_t element_count) -> std::size_t
    {
        // every invocation reduces two elements
        constexpr auto elements_per_work_group = 2 * s_work_group_size;
        // elements / elements per work group, rounded up
        return element_count / elements_per_work_group + (element_count % elements_per_work_group != 0);
    }

    // IndexedCopyKernel

    const std::string IndexedCopyKernel::s_source_string = (std::stringstream() << R"glsl(
#version 430 core

#define INVALID_INDEX -1u

layout(local_size_x=)glsl" << IndexedCopyKernel::s_workgroup_size << R"glsl() in;

layout(std430, binding = 0) restrict readonly
buffer )glsl" << IndexedCopyKernel::s_candidate_ssb_name << R"glsl(
{
    vec3 candidate_array[];
};

layout(std430, binding = 1) restrict readonly
buffer )glsl" << IndexedCopyKernel::s_index_ssb_name << R"glsl(
{
    uint index_sum;
    uint index_array[];
};

layout(std430, binding = 1) restrict writeonly
buffer )glsl" << IndexedCopyKernel::s_position_ssb_name << R"glsl(
{
    vec3 position_array[];
};

void main()
{
    if (gl_GlobalInvocationID.x < candidate_array.length())
    {
        const uint array_index = index_array[gl_GlobalInvocationID.x];
        if (array_index != INVALID_INDEX)
            position_array[array_index] = candidate_array[gl_GlobalInvocationID.x];
    }
}
)glsl").str();

    IndexedCopyKernel::IndexedCopyKernel() : ReductionKernel(s_source_string) {}

    void IndexedCopyKernel::dispatchCompute(std::size_t candidate_count) const
    {
        const GLuint num_workgroups = candidate_count / s_workgroup_size + (candidate_count % s_workgroup_size != 0);
        useProgram();
        gl.DispatchCompute(num_workgroups, 1, 1);
    }
} // placement