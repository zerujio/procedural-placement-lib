#ifndef PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP

#include "compute_kernel.hpp"

namespace placement {

class IndexationKernel final
{
public:
    static constexpr glm::uvec3 work_group_size{32, 1, 1};
    static constexpr uint glsl_version{450};

    IndexationKernel();

    void operator()(uint num_work_groups, uint candidate_buffer_binding_index, uint count_buffer_binding_index,
                    uint index_buffer_binding_index);

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
    static constexpr uint calculateNumWorkGroups(uint candidate_count)
    {
        return 1 + candidate_count / (2 * work_group_size.x);
    }

private:
    ComputeShaderProgram m_program;

    using CS = ComputeShaderProgram;

    CS::ShaderStorageBlock m_candidate_buffer;
    CS::ShaderStorageBlock m_count_buffer;
    CS::ShaderStorageBlock m_index_buffer;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_INDEXATION_KERNEL_HPP
