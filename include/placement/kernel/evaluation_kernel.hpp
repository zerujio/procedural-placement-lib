#ifndef PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP

#include "compute_kernel.hpp"

#include <array>

namespace placement {

class EvaluationKernel final : public ComputeShaderProgram
{
public:
    static constexpr glm::uvec3 work_group_size {8, 8, 1};
    static const std::array<std::array<float, work_group_size.y>, work_group_size.x> default_dithering_matrix;

    EvaluationKernel();

    void setClassIndex(uint index) { setUniform(m_class_index, index); }

    void setLowerBound(glm::vec2 lower_bound) { setUniform(m_lower_bound, lower_bound); }
    void setUpperBound(glm::vec2 upper_bound) { setUniform(m_upper_bound, upper_bound); }

    template<typename ArrayLike>
    void setDitheringMatrix(const ArrayLike& values)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setWorkGroupPatternColumn(i, values[i * work_group_size.y]);
    }

    template<typename NestedArrayLike>
    void setDitheringMatrixColumns(const NestedArrayLike& columns)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setDitheringMatrixColumn(i, columns[i]);
    }

    template<typename ArrayLike>
    void setDitheringMatrixColumn(uint column_index, const ArrayLike& column_values)
    {
        setDitheringMatrixColumn(column_index, std::data(column_values));
    }

    void setDitheringMatrixColumn(uint column_index, const float* column_values)
    {
        setUniform(m_dithering_matrix.value + column_index * work_group_size.y, work_group_size.y, column_values);
    }

    auto densityMapTextureUnit() { return UniformAccessor<&EvaluationKernel::m_density_map>(this); }

    void setDensityMapTextureUnit(uint index) { setUniform<int>(m_density_map, index); }

    void setCandidateBufferBindingIndex(uint index) { setShaderStorageBlockBindingIndex(m_candidate_buffer, index); }

    void setWorldUVBufferBindingIndex(uint index) { setShaderStorageBlockBindingIndex(m_world_uv_buffer, index); }

    void setDensityBufferBindingIndex(uint index) { setShaderStorageBlockBindingIndex(m_density_buffer, index); }

private:
    Uniform<int> m_density_map {getUniformLocation("u_density_map")};
    UniformLocation m_class_index {getUniformLocation("u_class_index")};
    UniformLocation m_dithering_matrix {getUniformLocation("u_dithering_matrix[0][0]")};
    UniformLocation m_lower_bound {getUniformLocation("u_lower_bound")};
    UniformLocation m_upper_bound {getUniformLocation("u_upper_bound")};
    ShaderStorageBlockIndex m_candidate_buffer {getShaderStorageBlockIndex("CandidateBuffer")};
    ShaderStorageBlockIndex m_world_uv_buffer {getShaderStorageBlockIndex("WorldUVBuffer")};
    ShaderStorageBlockIndex m_density_buffer {getShaderStorageBlockIndex("DensityBuffer")};
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP
