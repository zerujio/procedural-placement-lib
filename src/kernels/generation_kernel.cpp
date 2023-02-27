#include "placement/kernel/generation_kernel.hpp"

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
buffer DensityBuffer
{
    float[gl_WorkGroupSize.x][gl_WorkGroupSize.y] density_array[];
};

void main()
{
    const uint array_index = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;

    const uvec2 grid_index = gl_WorkGroupID.xy + u_work_group_offset;
    const vec2 h_position = u_footprint * (u_work_group_pattern[gl_LocalInvocationID.x][gl_LocalInvocationID.y]
                                         + grid_index * u_work_group_scale);

    const vec2 world_uv = h_position / u_world_scale.xy;
    world_uv_array[array_index][gl_WorkGroupID.x][gl_WorkGroupID.y] = world_uv;

    const float height = texture(u_heightmap, world_uv).x * u_world_scale.z;
    candidate_array[array_index][gl_WorkGroupID.x][gl_WorkGroupID.y] = Candidate(vec3(h_position, height), 0);

    density_array[array_index][gl_WorkGroupID.x][gl_WorkGroupID.y] = 0.0f;
}
)gl";

namespace placement {

GenerationKernel::GenerationKernel()
    :   m_program(source_string),
        m_footprint(m_program.getUniformLocation("u_footprint")),
        m_world_scale(m_program.getUniformLocation("u_world_scale")),
        m_work_group_scale(m_program.getUniformLocation("u_work_group_scale")),
        m_work_group_offset(m_program.getUniformLocation("u_work_group_offset")),
        m_work_group_pattern(m_program.getUniformLocation("u_work_group_pattern[0][0]")),
        m_heightmap_tex(m_program.getUniformLocation("u_heightmap")),
        m_candidate_buf(m_program.getShaderStorageBlockIndex("CandidateBuffer")),
        m_world_uv_buf(m_program.getShaderStorageBlockIndex("WorldUVBuffer")),
        m_density_buf(m_program.getShaderStorageBlockIndex("DensityBuffer"))
{}

void GenerationKernel::operator()(glm::uvec2 num_work_groups, glm::uvec2 group_offset, float footprint,
                                  glm::vec3 world_scale, GLuint heightmap_texture_unit,
                                  GLuint candidate_buffer_binding_index, GLuint density_buffer_binding_index,
                                  GLuint world_uv_buffer_binding_index)
{
    // uniforms
    m_program.setUniform(m_work_group_offset, group_offset);
    m_program.setUniform(m_footprint, footprint);
    m_program.setUniform(m_world_scale, world_scale);

    // textures
    m_program.setUniform(m_heightmap_tex, static_cast<GLint>(heightmap_texture_unit));

    // ssbo bindings
    m_program.setShaderStorageBlockBindingIndex(m_candidate_buf, candidate_buffer_binding_index);
    m_program.setShaderStorageBlockBindingIndex(m_density_buf, density_buffer_binding_index);
    m_program.setShaderStorageBlockBindingIndex(m_world_uv_buf, world_uv_buffer_binding_index);

    m_program.dispatch({num_work_groups, 1});
}

} // placement