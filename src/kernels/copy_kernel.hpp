#ifndef PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP

#include "placement/compute_kernel.hpp"

namespace placement {

class CopyKernel final : public ComputeKernel
{
public:
    static constexpr glm::uvec3 work_group_size {64, 1, 1};
    static constexpr uint glsl_version {430};

    CopyKernel();

    [[nodiscard]]
    uint getCandidateBufferBindingIndex() const { return m_candidate_buffer.getBindingIndex(); }
    void setCandidateBufferBindingIndex(uint index) { m_candidate_buffer.setBindingIndex(*this, index); }

    [[nodiscard]]
    uint getIndexBufferBindingIndex() const { return m_index_buffer.getBindingIndex(); }
    void setIndexBufferBindingIndex(uint index) { m_index_buffer.setBindingIndex(*this, index); }

    [[nodiscard]]
    uint getOutputBufferBindingIndex() const { return m_output_buffer.getBindingIndex(); }
    void setOutputBufferBindingIndex(uint index) { m_output_buffer.setBindingIndex(*this, index); }

private:
    ShaderStorageBlock m_candidate_buffer;
    ShaderStorageBlock m_index_buffer;
    ShaderStorageBlock m_output_buffer;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP
