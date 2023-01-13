#ifndef PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP

#include "placement/compute_kernel.hpp"

namespace placement {

class EvaluationKernel final : public ComputeKernel
{
public:
    static constexpr glm::uvec3 work_group_size {8, 8, 1};

    EvaluationKernel();

    void setClassIndex(uint index) { m_setUniform(m_class_index, index); }

    template<typename ArrayLike>
    void setDitheringMatrix(const ArrayLike& values)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setWorkGroupPatternColumn(i, values[i * work_group_size.y]);
    }

    template<typename NestedArrayLike>
    void SetDitheringMatrixColumns(const NestedArrayLike& columns)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setWorkGroupPatternColumn(i, columns[i]);
    }

    template<typename ArrayLike>
    void setDitheringMatrixColumn(uint column_index, const ArrayLike& column_values)
    {
        setDitheringMatrixColumn(column_index, std::data(column_values));
    }

    void setDitheringMatrixColumn(uint column_index, const glm::vec2* column_values)
    {
        m_setUniform(m_dithering_matrix.value + column_index * work_group_size.y, work_group_size.y, column_values);
    }

    void setDensityMapTextureUnit(uint index) { m_setUniform<int>(m_density_map, index); }

    void setCandidateBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_candidate_buffer, index); }

    void setWorldUVBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_world_uv_buffer, index); }

    void setDensityBufferBindingIndex(uint index) { m_setShaderStorageBlockBinding(m_density_buffer, index); }

private:
    UniformLocation m_density_map {m_getUniformLocation("u_density_map")};
    UniformLocation m_class_index {m_getUniformLocation("u_class_index")};
    UniformLocation m_dithering_matrix {m_getUniformLocation("u_dithering_matrix")};
    ShaderStorageBlockIndex m_candidate_buffer {m_getShaderStorageBlockIndex("CandidateBuffer")};
    ShaderStorageBlockIndex m_world_uv_buffer {m_getShaderStorageBlockIndex("WorldUVBuffer")};
    ShaderStorageBlockIndex m_density_buffer {m_getShaderStorageBlockIndex("DensityBuffer")};
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP
