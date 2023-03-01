#include "placement/kernel/evaluation_kernel.hpp"

static constexpr auto source_string = R"gl(
#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;

uniform sampler2D u_density_map;
uniform float u_density_map_scale;
uniform uint u_class_index;
uniform float u_dithering_matrix [gl_WorkGroupSize.x][gl_WorkGroupSize.y];
uniform vec2 u_lower_bound;
uniform vec2 u_upper_bound;

struct Candidate {
    vec3 position;
    uint class_index;
};

layout(std430) restrict
buffer CandidateBuffer
{
    Candidate[gl_WorkGroupSize.x][gl_WorkGroupSize.y] candidate_array[];
};

layout(std430) restrict readonly
buffer WorldUVBuffer
{
    vec2[gl_WorkGroupSize.x][gl_WorkGroupSize.y] world_uv_array[];
};

layout(std430) restrict
buffer DensityBuffer
{
    float[gl_WorkGroupSize.x][gl_WorkGroupSize.y] density_array[];
};

void main()
{
    const uint array_index = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;

    const vec2 world_uv = world_uv_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y];

    const uvec2 threshold_matrix_index = (gl_LocalInvocationID.xy + uvec2(world_uv * gl_WorkGroupSize.xy)) % gl_WorkGroupSize.xy;
    const float threshold = u_dithering_matrix[threshold_matrix_index.x][threshold_matrix_index.y];

    const float density = density_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y]
                        + texture(u_density_map, world_uv).x * u_density_map_scale;

    density_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = density;

    const vec2 position2d = candidate_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y].position.xy;
    const bool above_lower_bound = all(greaterThanEqual(position2d, u_lower_bound));
    const bool below_upper_bound = all(lessThan(position2d, u_upper_bound));

    const uint current_layer_index = candidate_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y].class_index;

    if (u_class_index < current_layer_index && density > threshold && above_lower_bound && below_upper_bound)
        candidate_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y].class_index = u_class_index;
}
)gl";

namespace placement {

EvaluationKernel::EvaluationKernel()
        : m_program(source_string),
          m_class_index(m_program.getUniformLocation("u_class_index")),
          m_lower_bound(m_program.getUniformLocation("u_lower_bound")),
          m_upper_bound(m_program.getUniformLocation("u_upper_bound")),
          m_dithering_matrix(m_program.getUniformLocation("u_dithering_matrix[0][0]")),
          m_density_map_scale(m_program.getUniformLocation("u_density_map_scale")),
          m_density_map(m_program.getUniformLocation("u_density_map")),
          m_candidate_buffer(m_program.getShaderStorageBlockIndex("CandidateBuffer")),
          m_world_uv_buffer(m_program.getShaderStorageBlockIndex("WorldUVBuffer")),
          m_density_buffer(m_program.getShaderStorageBlockIndex("DensityBuffer"))
{
    setDitheringMatrixColumns(default_dithering_matrix);
}

constexpr auto wg_size = EvaluationKernel::work_group_size;

using Matrix = std::array<std::array<float, wg_size.y>, wg_size.x>;

constexpr Matrix makeDefaultDitheringMatrix()
{
    Matrix m{
            0, 32, 8, 40, 2, 34, 10, 42,
            48, 16, 56, 24, 50, 18, 58, 26,
            12, 44, 4, 36, 14, 46, 6, 38,
            60, 28, 52, 20, 62, 30, 54, 22,
            3, 35, 11, 43, 1, 33, 9, 41,
            51, 19, 59, 27, 49, 17, 57, 25,
            15, 47, 7, 39, 13, 45, 5, 37,
            63, 31, 55, 23, 61, 29, 53, 21
    };

    for (auto &col: m)
        for (auto &value: col)
            value /= 64.f;

    return m;
}

const Matrix EvaluationKernel::default_dithering_matrix{makeDefaultDitheringMatrix()};

void
EvaluationKernel::operator()(glm::uvec2 num_work_groups, uint class_index, glm::vec2 lower_bound, glm::vec2 upper_bound,
                             GLuint density_map_texture_unit, float density_map_scale, GLuint candidate_buffer_binding_index,
                             GLuint world_uv_buffer_binding_index, GLuint density_buffer_binding_index)
{
    // uniforms
    m_program.setUniform(m_class_index, class_index);
    m_program.setUniform(m_lower_bound, lower_bound);
    m_program.setUniform(m_upper_bound, upper_bound);
    m_program.setUniform(m_density_map_scale, density_map_scale);

    // textures
    m_program.setUniform(m_density_map, static_cast<GLint>(density_map_texture_unit));

    // shader storage buffer bindings
    m_program.setShaderStorageBlockBindingIndex(m_candidate_buffer, candidate_buffer_binding_index);
    m_program.setShaderStorageBlockBindingIndex(m_world_uv_buffer, world_uv_buffer_binding_index);
    m_program.setShaderStorageBlockBindingIndex(m_density_buffer, density_buffer_binding_index);

    m_program.dispatch({num_work_groups, 1});
}

} // placement