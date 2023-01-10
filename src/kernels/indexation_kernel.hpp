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

    [[nodiscard]]
    uint getCandidateBufferBindingIndex() const { return m_candidate_buffer.getBindingIndex(); }
    void setCandidateBufferBindingIndex(uint index) { m_candidate_buffer.setBindingIndex(*this, index); }

    [[nodiscard]]
    uint getCountBufferBindingIndex() const {return m_count_buffer.getBindingIndex(); }
    void setCountBufferBindingIndex(uint index) { m_count_buffer.setBindingIndex(*this, index); }
    [[nodiscard]]
    static constexpr GL::GLsizeiptr getCountBufferMemoryRequirement(GL::GLintptr number_of_classes)
    {
        return (1 + number_of_classes) * sizeof(uint);
    }

    [[nodiscard]]
    uint getIndexBufferBindingIndex() const { return m_index_buffer.getBindingIndex(); }
    void setIndexBufferBindingIndex(uint index) { m_index_buffer.setBindingIndex(*this, index); }
    [[nodiscard]]
    static constexpr GL::GLsizeiptr getIndexBufferMemoryRequirement(GL::GLintptr number_of_candidates)
    {
        return number_of_candidates * sizeof(uint);
    }

private:
    UniformLocation m_class_index;
    ShaderStorageBlock m_candidate_buffer;
    ShaderStorageBlock m_count_buffer;
    ShaderStorageBlock m_index_buffer;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP
