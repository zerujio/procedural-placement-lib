#include "generation_kernel.hpp"

static constexpr auto source_string = R"gl(
#version 450 core

layout(local_size_x = 8, local_size_y = 8) in;

uniform float u_footprint;
uniform vec3 u_world_scale;
uniform vec2 u_work_group_scale;
uniform uvec2 u_work_group_offset;
uniform vec2 u_work_group_pattern[gl_WorkGroupSize.x][gl_WorkGroupSize.y];

uniform sampler2D u_heightmap;

struct Candidate
{
    vec3 position;
    uint class_index;
};

layout(std430) restrict writeonly
buffer CandidateBuffer
{
    Candidate[gl_WorkGroupSize.x][gl_WorkGroupSize.y] candidate_array[];
};

layout(std430) restrict writeonly
buffer WorldUVBuffer
{
    vec2[gl_WorkGroupSize.x][gl_WorkGroupSize.y] world_uv_array[];
};

layout(std430) restrict writeonly
{
    float[gl_WorkGroupSize.x][gl_WorkGroupSize.y] density_array[];
};

void main()
{
    const uint array_index = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;

    const uvec2 grid_index = gl_WorkGroupID.xy + u_work_group_offset;
    const vec2 h_position = footprint * (u_work_group_pattern[gl_LocalInvocationID.x][gl_LocalInvocationID.y]
                                         + grid_index * u_work_group_scale);

    const vec2 world_uv = h_position / world_scale.xy;
    world_uv_array[array_index][gl_WorkGroupID.x][gl_WorkGroupID.y] = world_uv;

    const float height = texture(u_heightmap, world_uv) * world_scale.z;
    candidate_array[array_index][gl_WorkGroupID.x][gl_WorkGroupID.y] = Candidate(vec3(h_position, height), 0);

    density_array[array_index][gl_WorkGroupID.x][gl_WorkGroupID.y] = 0.0f;
}
)gl";

namespace placement {

GenerationKernel::GenerationKernel() : ComputeKernel(source_string),
                                       m_footprint(*this, "u_footprint"),
                                       m_world_scale(*this, "u_world_scale"),
                                       m_work_group_scale(*this, "u_work_group_scale"),
                                       m_work_group_offset(*this, "u_work_group_offset"),
                                       m_work_group_pattern(*this, "u_work_group_pattern"),
                                       m_heightmap_tex(*this, "u_heightmap"),
                                       m_candidate_buf(*this, "CandidateBuffer"),
                                       m_world_uv_buf(*this, "WorldUVBuffer"),
                                       m_density_buf(*this, "DensityBuffer")
{}

} // placement