#include "placement/reduction_kernel.hpp"

#include "glutils/buffer.hpp"
#include "gl_context.hpp"

#include <sstream>

/*
 *  TODO: solve the inter-workgroup ordering problem.
 *
 *  Possible solution 1:
 *  Extract the loop from the shader into multiple compute dispatches. i.e.:
 *
 *      for (std::size_t group_size = 1; group_size < candidate_count; group_size <<= 1)
 *      {
 *          setUniform(group_size_uniform_location, group_size);
 *          gl.DispatchCompute(...);
 *          m_ensureOutputVisibility();
 *      }
 *
 * This is the simplest solution, but may incur in performance penalties.
 *
 * Possible solution 2:
 * Only perform reduction within a single work group, move final copy to separate shader. To reduce more items, dispatch
 * multiple times, with each dispatch covering more per work group.
 *
 *      uniform uint base_group_size; // 1, 2, 4, 8, 16, ...
 *      shared uint index_sum[2 * gl_WorkGroupSize.x];
 *
 *      uint localToGlobalIndex(uint local_index) { return gl_WorkGroupID.x * gl_WorkGroupSize.x + local_index; }
 *      uint groupOffsetFromGlobalIndex(uint global_group_index) { return global_group_index * base_group_size; }
 *      uint groupOffsetFromLocalIndex(uint local_index) { return groupOffsetFromGlobalIndex(localToGlobalIndex(local_index)); }
 *
 *      void addToIndexBuffer(uint local_group_index, int element_index, value)
 *      {
 *          index_buffer[groupOffsetFromLocalIndex(local_group_index) + element_index] += value;
 *      }
 *
 *      uint readFromIndexBuffer(uint local_group_index, int element_index)
 *      {
 *          return index_buffer[groupOffsetFromLocalIndex(local_group_index) + element_index];
 *      }
 *
 *      void main()
 *      {
 *          // se copia el último elemento de cada batch/grupo a memoria local compartida
 *          index_sum[gl_LocalInvocationIndex.x] = readFromIndexBuffer(gl_LocalInvocationID.x + 1, - 1);
 *          index_sum [gl_LocalInvocationIndex.x * 2] = readFromIndexBuffer(gl_LocalInvocationID.x * 2 + 1, - 1);
 *
 *          memoryBarrierShared();
 *          barrier();
 *
 *          // el algoritmo de reducción clásico, pero solo sobre los elementos del work group
 *          for (uint local_group_size = 1; local_group_size < gl_WorkGroupSize.x; local_group_size <<= 1)
 *          {
 *              const uint write_index = ...;
 *              const uint read_index = ...;
 *
 *              index_sum[write_index] += index_sum[read_index];
 *
 *              memoryBarrierShared();
 *              barrier();
 *          }
 *
 *          // se suma el offset correspondiente al grupo completo
 *          for (uint local_group_index = gl_LocalInvocationID.x;
 *              factor <= gl_LocalInvocationID.x * 2;
 *              local_group_index *= 2)
 *          {
 *              for (uint i = 0; i < base_group_size; i++)
 *              {
 *                  addToIndexBuffer(local_group_index, i, index_sum[local_group_index]);
 *              }
 *          }
 *      }
 *
 * Bonus: this allows the use of shared memory!.
 */

namespace placement {

     const std::string ReductionKernel::s_source_string = (std::stringstream() << R"glsl(
#version 430 core

layout(local_size_x = )glsl" << ReductionKernel::s_work_group_size << R"glsl() in;

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

Candidate readCandidate(uint index)
{
    if (index < position_buffer.length())
        return Candidate(position_buffer[index], index_buffer[index]);
    else
        return Candidate(vec3(0.0f), 0);
}

void main()
{
    const uint invocation_index = gl_GlobalInvocationID.x * 2;
    const Candidate c0 = readCandidate(invocation_index);
    const Candidate c1 = readCandidate(invocation_index + 1);

    for (uint group_size = 1; group_size < index_buffer.length(); group_size <<= 1)
    {
        barrier();
        memoryBarrierBuffer();

        const uint group_index = (gl_GlobalInvocationID.x / group_size) * 2 + 1;
        const uint base_index = group_index * group_size;
        const uint write_index = base_index + gl_GlobalInvocationID.x % group_size;
        const uint read_index = base_index - 1;

        if (write_index < index_buffer.length())
            // no need to check read_index. Per its definition, it is bound by 0 < read_index < write_index.
            index_buffer[write_index] += index_buffer[read_index];
    }

    memoryBarrierBuffer();
    barrier();

    if (c0.index == 1)
    {
        const uint reduced_index = index_buffer[invocation_index] - 1;
        position_buffer[reduced_index] = c0.position;
    }

    if (c1.index == 1)
    {
        const uint reduced_index = index_buffer[invocation_index + 1] - 1;
        position_buffer[reduced_index] = c1.position;
    }
}
)glsl").str();

    using namespace glutils;

    ReductionKernel::ReductionKernel() : PlacementPipelineKernel(s_source_string) {}

    auto ReductionKernel::dispatchCompute(std::size_t candidate_count) const -> std::size_t
    {
        if (candidate_count == 0)
            return 0;

        useProgram();
        gl.DispatchCompute(m_calculateNumWorkGroups(candidate_count), 1, 1);
        m_ensureOutputVisibility();
        return (candidate_count - 1) * sizeof(GLuint);
    }

    auto ReductionKernel::m_calculateNumWorkGroups(const std::size_t element_count) -> std::size_t
    {
        // every invocation reduces two elements
        constexpr auto elements_per_work_group = 2 * s_work_group_size;
        // elements / elements per work group, rounded up
        return element_count / elements_per_work_group + (element_count % elements_per_work_group != 0);
    }

    void ReductionKernel::setAuxiliaryBufferBindingIndices(glm::uvec2 indices)
    {
        m_forward_kernel.setInputBindingIndex(indices.x);
        m_backward_kernel.setInputBindingIndex(indices.x);
        m_forward_kernel.setOutputBindingIndex(indices.y);
        m_backward_kernel.setOutputBindingIndex(indices.y);
    }

    auto ReductionKernel::getAuxiliaryBufferBindingIndices() const -> glm::uvec2
    {
        return {m_forward_kernel.getInputBindingIndex(), m_backward_kernel.getOutputBindingIndex()};
    }

} // placement