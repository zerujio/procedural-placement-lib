#ifndef PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP

#include "compute_kernel.hpp"

#include <array>

namespace placement {

class EvaluationKernel final
{
public:
    static constexpr glm::uvec3 work_group_size{8, 8, 1};
    static const std::array<std::array<float, work_group_size.y>, work_group_size.x> default_dithering_matrix;

    EvaluationKernel();

    void operator()(glm::uvec2 num_work_groups, uint class_index, glm::vec2 lower_bound, glm::vec2 upper_bound,
                    GLuint density_map_texture_unit, GLuint candidate_buffer_binding_index,
                    GLuint world_uv_buffer_binding_index, GLuint density_buffer_binding_index);

    template<typename ArrayLike>
    void setDitheringMatrix(const ArrayLike &values)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setWorkGroupPatternColumn(i, values[i * work_group_size.y]);
    }

    template<typename NestedArrayLike>
    void setDitheringMatrixColumns(const NestedArrayLike &columns)
    {
        for (uint i = 0; i < work_group_size.x; i++)
            setDitheringMatrixColumn(i, columns[i]);
    }

    template<typename ArrayLike>
    void setDitheringMatrixColumn(uint column_index, const ArrayLike &column_values)
    {
        m_program.setUniform(m_dithering_matrix[column_index], column_values);
    }

private:
    ComputeShaderProgram m_program;

    using CS = ComputeShaderProgram;

    CS::TypedUniform <uint> m_class_index;
    CS::TypedUniform <glm::vec2> m_lower_bound;
    CS::TypedUniform <glm::vec2> m_upper_bound;
    CS::TypedUniform<float[work_group_size.x][work_group_size.y]> m_dithering_matrix;
    CS::CachedUniform<int> m_density_map;
    CS::ShaderStorageBlock m_candidate_buffer;
    CS::ShaderStorageBlock m_world_uv_buffer;
    CS::ShaderStorageBlock m_density_buffer;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_EVALUATION_KERNEL_HPP
