#ifndef PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP

#include "placement/compute_kernel.hpp"

namespace placement {

class IndexationKernel final : public ComputeKernel
{
public:
    static constexpr glm::uvec3 work_group_size {32, 1, 1};
    static constexpr uint glsl_version {450};

    IndexationKernel();

    void setCandidateBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_candidate_buffer, index); }

    void setCountBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_count_buffer, index); }

    void setIndexBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_index_buffer, index); }

    [[nodiscard]]
    static constexpr GL::GLsizeiptr getCountBufferMemoryRequirement(uint candidate_count)
    {
        return (1 + candidate_count) * static_cast<GL::GLsizeiptr>(sizeof(uint));
    }

    [[nodiscard]]
    static constexpr GL::GLsizeiptr getIndexBufferMemoryRequirement(uint candidate_count)
    {
        return candidate_count * static_cast<GL::GLsizeiptr>(sizeof(uint));
    }

    [[nodiscard]]
    static constexpr glm::uvec3 calculateNumWorkGroups(uint candidate_count)
    {
        return {1 + candidate_count / (2 * work_group_size.x), 1, 1};
    }

private:
    ShaderStorageBlockIndex m_candidate_buffer {m_getShaderStorageBlockIndex("CandidateBuffer")};
    ShaderStorageBlockIndex m_count_buffer {m_getShaderStorageBlockIndex("CountBuffer")};
    ShaderStorageBlockIndex m_index_buffer {m_getShaderStorageBlockIndex("IndexBuffer")};
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP
