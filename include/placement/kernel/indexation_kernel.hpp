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

    void setClassIndex(uint index) { m_setUniform(m_class_index, index); }

    void setCandidateBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_candidate_buffer, index); }

    void setCountBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_count_buffer, index); }

    [[nodiscard]]
    static constexpr GL::GLsizeiptr getCountBufferMemoryRequirement(GL::GLintptr number_of_classes)
    {
        return (1 + number_of_classes) * sizeof(uint);
    }

    void setIndexBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_index_buffer, index); }

    [[nodiscard]]
    static constexpr GL::GLsizeiptr getIndexBufferMemoryRequirement(GL::GLintptr number_of_candidates)
    {
        return number_of_candidates * sizeof(uint);
    }

private:
    UniformLocation m_class_index {m_getUniformLocation("u_class_index")};
    ShaderStorageBlockIndex m_candidate_buffer {m_getShaderStorageBlockIndex("CandidateBuffer")};
    ShaderStorageBlockIndex m_count_buffer {m_getShaderStorageBlockIndex("CountBuffer")};
    ShaderStorageBlockIndex m_index_buffer {m_getShaderStorageBlockIndex("IndexBuffer")};
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP