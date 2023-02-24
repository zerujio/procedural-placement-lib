#include "placement/kernel/evaluation_kernel.hpp"

static constexpr auto source_string = R"gl(
#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;

uniform sampler2D u_density_map;
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

    const float threshold = u_dithering_matrix[gl_LocalInvocationID.x][gl_LocalInvocationID.y];
    const float density = density_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y]
                        + texture(u_density_map, world_uv).x;

    density_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = density;

    const vec2 position2d = candidate_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y].position.xy;
    const bool above_lower_bound = all(greaterThanEqual(position2d, u_lower_bound));
    const bool below_upper_bound = all(lessThan(position2d, u_upper_bound));

    if (density > threshold && above_lower_bound && below_upper_bound)
        candidate_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y].class_index = u_class_index;
}
)gl";

namespace placement {

EvaluationKernel::EvaluationKernel() : ComputeKernel(source_string)
{
    setDitheringMatrixColumns(default_dithering_matrix);
}

constexpr auto wg_size = EvaluationKernel::work_group_size;

using Matrix = std::array<std::array<float, wg_size.y>, wg_size.x>;

constexpr Matrix makeDefaultDitheringMatrix()
{
    Matrix m {
            0,  32, 8,  40, 2,  34, 10, 42,
            48, 16, 56, 24, 50, 18, 58, 26,
            12, 44, 4,  36, 14, 46, 6,  38,
            60, 28, 52, 20, 62, 30, 54, 22,
            3,  35, 11, 43, 1,  33, 9,  41,
            51, 19, 59, 27, 49, 17, 57, 25,
            15, 47, 7,  39, 13, 45, 5,  37,
            63, 31, 55, 23, 61, 29, 53, 21
    };

    for (auto& col : m)
        for (auto& value : col)
            value /= 64.f;

    return m;
}

const Matrix EvaluationKernel::default_dithering_matrix {makeDefaultDitheringMatrix()};

} // placement