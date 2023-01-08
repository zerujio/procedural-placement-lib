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
        m_setUniform(m_dithering_matrix.get() + column_index * work_group_size.y, work_group_size.y, column_values);
    }

    [[nodiscard]]
    uint getDensityMapTextureUnit() const { return m_density_map.getTextureUnit(); }
    void setDensityMapTextureUnit(uint index) { m_density_map.setTextureUnit(*this, index); }

    [[nodiscard]]
    uint getCandidateBufferBindingIndex() const { return m_candidate_buffer.getBindingIndex(); }
    void setCandidateBufferBindingIndex(uint index) { m_candidate_buffer.setBindingIndex(*this, index); }

    [[nodiscard]]
    uint getWorldUVBufferBindingIndex() const { return m_world_uv_buffer.getBindingIndex(); }
    void setWorldUVBufferBindingIndex(uint index) { m_world_uv_buffer.setBindingIndex(*this, index); }

    [[nodiscard]]
    uint getDensityBufferBindingIndex() const {return m_density_buffer.getBindingIndex(); }
    void setDensityBufferBindingIndex(uint index) { m_density_buffer.setBindingIndex(*this, index); }

private:
    TextureSampler m_density_map;
    UniformLocation m_class_index;
    UniformLocation m_dithering_matrix;
    ShaderStorageBlock m_candidate_buffer;
    ShaderStorageBlock m_world_uv_buffer;
    ShaderStorageBlock m_density_buffer;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP
