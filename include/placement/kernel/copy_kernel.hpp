#ifndef PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP

#include "compute_kernel.hpp"

namespace placement {

class CopyKernel final
{
public:
    static constexpr glm::uvec3 work_group_size{64, 1, 1};
    static constexpr uint glsl_version{430};

    CopyKernel();

    void operator() (uint num_work_groups, GLuint candidate_buffer_binding_index, GLuint count_buffer_binding_index,
            GLuint index_buffer_binding_index, GLuint output_buffer_binding_index);

    [[nodiscard]]
    static constexpr uint calculateNumWorkGroups(uint candidate_count)
    { return 1u + candidate_count / work_group_size.x; }

private:
    ComputeShaderProgram m_program;
    using CS = ComputeShaderProgram;
    CS::ShaderStorageBlock m_candidate_buffer;
    CS::ShaderStorageBlock m_count_buffer;
    CS::ShaderStorageBlock m_index_buffer;
    CS::ShaderStorageBlock m_output_buffer;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_COPY_KERNEL_HPP
