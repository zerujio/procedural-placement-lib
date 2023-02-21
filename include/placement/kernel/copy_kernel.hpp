#ifndef PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP

#include "compute_kernel.hpp"

namespace placement {

class CopyKernel final : public ComputeKernel
{
public:
    static constexpr glm::uvec3 work_group_size{64, 1, 1};
    static constexpr uint glsl_version{430};

    CopyKernel();

    void setCandidateBufferBindingIndex(uint index)
    { m_setShaderStorageBlockBinding(m_candidate_buffer, index); }

    void setCountBufferBindingIndex(uint index)
    { m_setShaderStorageBlockBinding(m_count_buffer, index); }

    void setIndexBufferBindingIndex(uint index)
    { m_setShaderStorageBlockBinding(m_index_buffer, index); }

    void setOutputBufferBindingIndex(uint index)
    { m_setShaderStorageBlockBinding(m_output_buffer, index); }

    [[nodiscard]]
    static constexpr glm::uvec3 calculateNumWorkGroups(uint candidate_count)
    { return {1u + candidate_count / work_group_size.x, 1, 1}; }

private:
    ShaderStorageBlockIndex m_candidate_buffer{m_getShaderStorageBlockIndex("CandidateBuffer")};
    ShaderStorageBlockIndex m_count_buffer{m_getShaderStorageBlockIndex("CountBuffer")};
    ShaderStorageBlockIndex m_index_buffer{m_getShaderStorageBlockIndex("IndexBuffer")};
    ShaderStorageBlockIndex m_output_buffer{m_getShaderStorageBlockIndex("OutputBuffer")};
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP
