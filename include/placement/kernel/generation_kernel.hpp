#ifndef PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP

#include "compute_kernel.hpp"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"

namespace placement {

//TODO: rewrite other kerenels.

class GenerationKernel final
{
public:
    static constexpr glm::uvec3 work_group_size{8, 8, 1};

    GenerationKernel();

    /// Dispatch the compute kernel with the specified arguments.
    void operator()(glm::uvec2 num_work_groups, float footprint, glm::vec3 world_scale, glm::vec2 group_scale,
                    glm::uvec2 group_offset, GLuint heightmap_texture_unit, GLuint candidate_buffer_binding_index,
                    GLuint density_buffer_binding_index, GLuint world_uv_buffer_binding_index);

    template<typename ArrayLike>
    void setWorkGroupPattern(const ArrayLike &values)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setWorkGroupPatternColumn(i, values[i * work_group_size.y]);
    }

    template<typename NestedArrayLike>
    void setWorkGroupPatternColumns(const NestedArrayLike &columns)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setWorkGroupPatternColumn(i, columns[i]);
    }

    template<typename ArrayLike>
    void setWorkGroupPatternColumn(uint column_index, const ArrayLike &column_values)
    {
        m_program.setUniform(m_work_group_pattern[column_index], column_values);
    }

    /// How much space does the pattern specified with setWorkGroupPattern() occupies.
    void setWorkGroupPatternBoundaries(glm::vec2 boundaries)
    {
        m_program.setUniform(m_work_group_scale, boundaries);
    }

    [[nodiscard]]
    glm::vec2 getWorkGroupPatternBoundaries() const
    {
        return m_work_group_scale.getCachedValue();
    }

    [[nodiscard]]
    static constexpr GLsizeiptr getCandidateBufferSizeRequirement(glm::uvec3 num_work_groups)
    {
        return s_calculateBufferSize(num_work_groups, sizeof(glm::vec4));
    }

    [[nodiscard]]
    static constexpr GLsizeiptr getWorldUVBufferSizeRequirement(glm::uvec3 num_work_groups)
    {
        return s_calculateBufferSize(num_work_groups, sizeof(glm::vec2));
    }

    [[nodiscard]]
    static constexpr GLsizeiptr getDensityBufferMemoryRequirement(glm::uvec3 num_work_groups)
    {
        return s_calculateBufferSize(num_work_groups, sizeof(float));
    }

private:
    [[nodiscard]]
    static constexpr GLsizeiptr s_calculateBufferSize(glm::uvec3 num_work_groups, GLsizeiptr element_size)
    {
        const auto num_invocations = num_work_groups * work_group_size;
        return num_invocations.x * num_invocations.y * element_size;
    }

    template<typename T>
    using TypedUniform = ComputeShaderProgram::TypedUniform<T>;

    template<typename T>
    using CachedUniform = ComputeShaderProgram::CachedUniform<T>;

    using ShaderStorageBlock = ComputeShaderProgram::ShaderStorageBlock;

    ComputeShaderProgram m_program;
    TypedUniform<glm::vec2> m_footprint;
    TypedUniform<glm::vec3> m_world_scale;
    CachedUniform<glm::vec2> m_work_group_scale;
    TypedUniform<glm::uvec2> m_work_group_offset;
    TypedUniform<glm::vec2[work_group_size.x][work_group_size.y]> m_work_group_pattern;
    CachedUniform<int> m_heightmap_tex;
    ShaderStorageBlock m_candidate_buf;
    ShaderStorageBlock m_world_uv_buf;
    ShaderStorageBlock m_density_buf;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
