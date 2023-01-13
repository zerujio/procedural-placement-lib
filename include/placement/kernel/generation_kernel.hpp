#ifndef PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP

#include "placement/compute_kernel.hpp"

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"

namespace placement {

class NewGenerationKernel final : public ComputeKernel
{
public:
    static constexpr glm::uvec3 work_group_size {8, 8, 1};

    NewGenerationKernel();

    void setFootprint(float footprint) { m_setUniform(m_footprint, footprint); }

    void setWorldScale(glm::vec3 scale) { m_setUniform(m_world_scale, scale); }

    void setWorkGroupScale(glm::vec2 scale) { m_setUniform(m_work_group_scale, scale); }

    void setWorkGroupOffset(glm::uvec2 offset) { m_setUniform(m_work_group_offset, offset); }

    template<typename ArrayLike>
    void setWorkGroupPattern(const ArrayLike& values)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setWorkGroupPatternColumn(i, values[i * work_group_size.y]);
    }

    template<typename NestedArrayLike>
    void setWorkGroupPatternColumns(const NestedArrayLike& columns)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setWorkGroupPatternColumn(i, columns[i]);
    }

    template<typename ArrayLike>
    void setWorkGroupPatternColumn(uint column_index, const ArrayLike& column_values)
    {
        setWorkGroupPatternColumn(column_index, std::data(column_values));
    }

    void setWorkGroupPatternColumn(uint column_index, const glm::vec2* column_values)
    {
        m_setUniform(m_work_group_pattern.value + column_index * work_group_size.y, work_group_size.y, column_values);
    }

    void setHeightmapTextureUnit(GL::GLuint texture_unit) { m_setUniform<int>(m_heightmap_tex, texture_unit); }

    void setCandidateBufferBindingIndex(GL::GLuint index) { m_setShaderStorageBlockBinding(m_candidate_buf, index); }

    [[nodiscard]]
    static constexpr GL::GLsizeiptr getCandidateBufferSizeRequirement(glm::uvec3 num_work_groups)
    {
        return s_calculateBufferSize(num_work_groups, sizeof(glm::vec4));
    }

    void setWorldUVBufferBindingIndex(GL::GLuint index) { m_setShaderStorageBlockBinding(m_world_uv_buf, index); }

    [[nodiscard]]
    static constexpr GL::GLsizeiptr getWorldUVBufferSizeRequirement(glm::uvec3 num_work_groups)
    {
        return s_calculateBufferSize(num_work_groups, sizeof(glm::vec2));
    }

    void setDensityBufferBindingIndex(GL::GLuint index) { m_setShaderStorageBlockBinding(m_density_buf, index); }

    [[nodiscard]]
    static constexpr GL::GLsizeiptr getDensityBufferMemoryRequirement(glm::uvec3 num_work_groups)
    {
        return s_calculateBufferSize(num_work_groups, sizeof(float));
    }

private:
    [[nodiscard]]
    static constexpr GL::GLsizeiptr s_calculateBufferSize(glm::uvec3 num_work_groups, GL::GLsizeiptr element_size)
    {
        const auto num_invocations = num_work_groups * work_group_size;
        return num_invocations.x * num_invocations.y * element_size;
    }

    UniformLocation m_footprint { m_getUniformLocation("u_footprint") };
    UniformLocation m_world_scale { m_getUniformLocation("u_world_scale") };
    UniformLocation m_work_group_scale { m_getUniformLocation("u_work_group_scale") };
    UniformLocation m_work_group_offset { m_getUniformLocation("u_work_group_offset") };
    UniformLocation m_work_group_pattern { m_getUniformLocation("u_work_group_pattern") };
    UniformLocation m_heightmap_tex { m_getUniformLocation("u_heightmap") };
    ShaderStorageBlockIndex m_candidate_buf { m_getShaderStorageBlockIndex("CandidateBuffer") };
    ShaderStorageBlockIndex m_world_uv_buf { m_getShaderStorageBlockIndex("WorldUVBuffer") };
    ShaderStorageBlockIndex m_density_buf { m_getShaderStorageBlockIndex("DensityBuffer") };
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
