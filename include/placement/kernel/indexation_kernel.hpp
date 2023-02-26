#ifndef PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP

#include "compute_kernel.hpp"

namespace placement {

class IndexationKernel final : public ComputeShaderProgram
{
public:
    static constexpr glm::uvec3 work_group_size {32, 1, 1};
    static constexpr uint glsl_version {450};

    IndexationKernel();

    void setCandidateBufferBindingIndex(uint index) { setShaderStorageBlockBindingIndex(m_candidate_buffer, index); }

    void setCountBufferBindingIndex(uint index) { setShaderStorageBlockBindingIndex(m_count_buffer, index); }

    void setIndexBufferBindingIndex(uint index) { setShaderStorageBlockBindingIndex(m_index_buffer, index); }

    [[nodiscard]]
    static constexpr GLsizeiptr getCountBufferMemoryRequirement(uint class_count)
    {
        return class_count * static_cast<GLsizeiptr>(sizeof(uint));
    }

    [[nodiscard]]
    static constexpr GLsizeiptr getIndexBufferMemoryRequirement(uint candidate_count)
    {
        return candidate_count * static_cast<GLsizeiptr>(sizeof(uint));
    }

    [[nodiscard]]
    static constexpr glm::uvec3 calculateNumWorkGroups(uint candidate_count)
    {
        return {1 + candidate_count / (2 * work_group_size.x), 1, 1};
    }

private:
    ShaderStorageBlockIndex m_candidate_buffer {getShaderStorageBlockIndex("CandidateBuffer")};
    ShaderStorageBlockIndex m_count_buffer {getShaderStorageBlockIndex("CountBuffer")};
    ShaderStorageBlockIndex m_index_buffer {getShaderStorageBlockIndex("IndexBuffer")};
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP
