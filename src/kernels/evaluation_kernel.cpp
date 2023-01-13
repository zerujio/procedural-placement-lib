#include "placement/kernel/evaluation_kernel.hpp"

static constexpr auto source_string = R"gl(
#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;

uniform sampler2D u_density_map;
uniform uint u_class_index;
uniform float u_dithering_matrix [gl_WorkGroupSize.x][gl_WorkGroupSize.y];

struct Candidate {
    vec3 position;
    uint class_index;
};

layout(std430) restrict writeonly
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
                        + texture(u_density_map, world_uv);

    if (density > threshold)
        candidate_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y].class_index = u_class_index;

    density_array[array_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = density;
}
)gl";

namespace placement {

EvaluationKernel::EvaluationKernel() : ComputeKernel(source_string) {}

} // placement