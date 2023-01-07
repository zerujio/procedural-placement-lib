#include "placement/generation_kernel.hpp"
#include "gl_context.hpp"

#include "glutils/glsl_syntax.hpp"

#include <string>
#include <sstream>

namespace placement {

using namespace GL;

static const Definition lower_bound_def {
    .layout{.location=0},
    .storage = StorageQualifier::uniform,
    .type = Type::vec2_,
    .name = "u_lower_bound"
};

static const Definition upper_bound_def {
    .layout{.location=1},
    .storage = StorageQualifier::uniform,
    .type = Type::vec2_,
    .name = "u_upper_bound"
};

static const Definition world_scale_def {
    .layout{.location=2},
    .storage = StorageQualifier::uniform,
    .type = Type::vec3_,
    .name = "u_world_scale"
};

static const Definition norm_factor_def {
    .layout{.location=3},
    .storage = StorageQualifier::uniform,
    .type = Type::vec2_,
    .name = "u_norm_factor"
};

static const Definition work_group_offset_def {
    .layout{.location=4},
    .storage = StorageQualifier::uniform,
    .type = Type::uvec2_,
    .name = "u_work_group_offset"
};

static const Definition position_stencil_scale_def {
    .layout{.location=5},
    .storage = StorageQualifier::uniform,
    .type = Type::vec2_,
    .name = "u_position_stencil_scale"
};

static const Definition position_stencil_def {
    .layout{.location=6},
    .storage = StorageQualifier::uniform,
    .type = Type::vec2_,
    .name = "u_position_stencil[gl_WorkGroupSize.x][gl_WorkGroupSize.y]"
};

static const Definition height_tex_def {
    .storage = StorageQualifier::uniform,
    .type = Type::sampler2D,
    .name = "u_heightmap"
};

static const Definition density_tex_def {
    .storage = StorageQualifier::uniform,
    .type = Type::sampler2D,
    .name = "u_densitymap"
};

const std::string GenerationKernel::s_source_string = (std::ostringstream()
        << "#version 450 core\n"
        << "layout(local_size_x = " << GenerationKernel::work_group_size.x
        << ", local_size_y = " << GenerationKernel::work_group_size.y << ") in;"
        << "\n" << lower_bound_def
        << "\n" << upper_bound_def
        << "\n" << world_scale_def
        << "\n" << norm_factor_def
        << "\n" << work_group_offset_def
        << "\n" << position_stencil_scale_def
        << "\n" << position_stencil_def
        << "\n" << height_tex_def
        << "\n" << density_tex_def
        << R"glsl(

struct Candidate
{
    vec3 position;
    bool valid;
};

layout(std430, binding = 0) restrict writeonly
buffer )glsl" << PlacementPipelineKernel::s_candidate_ssb_name << R"glsl(
{
    Candidate candidate_array[][gl_WorkGroupSize.x][gl_WorkGroupSize.y];
};

const float dithering_matrix[gl_WorkGroupSize.x][gl_WorkGroupSize.y] =
{
    { 0, 32,  8, 40,  2, 34, 10, 42},
    {48, 16, 56, 24, 50, 18, 58, 26},
    {12, 44,  4, 36, 14, 46,  6, 38},
    {60, 28, 52, 20, 62, 30, 54, 22},
    { 3, 35, 11, 43,  1, 33,  9, 41},
    {51, 19, 59, 27, 49, 17, 57, 25},
    {15, 47,  7, 39, 13, 45,  5, 37},
    {63, 31, 55, 23, 61, 29, 53, 21}
};

void main()
{
    const uint group_index = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;

    // position
    const vec2 tex_coord = ((u_work_group_offset + gl_WorkGroupID.xy) * gl_WorkGroupSize.xy * u_position_stencil_scale
                              + u_position_stencil[gl_LocalInvocationID.x][gl_LocalInvocationID.y])
                            * u_norm_factor;

    const float norm_height = texture(u_heightmap, tex_coord).x;
    const vec3 position = vec3(tex_coord.x, norm_height, tex_coord.y) * u_world_scale;

    // validity
    const float density_value = texture(u_densitymap, tex_coord).x;
    const uvec2 matrix_index = (gl_LocalInvocationID.xy + gl_WorkGroupID.xy) % gl_WorkGroupSize.xy;
    const float threshold_value = dithering_matrix[matrix_index.x][matrix_index.y] / 64.0f;
    const bool is_valid = all(greaterThanEqual(position.xz, u_lower_bound))
                        && all(lessThan(position.xz, u_upper_bound))
                        && density_value > threshold_value;

    // write results
    candidate_array[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = Candidate(position, is_valid);
}
)glsl").str();

GenerationKernel::GenerationKernel() :
        PlacementPipelineKernel(s_source_string),
        m_heightmap(*this, height_tex_def.name),
        m_densitymap(*this, density_tex_def.name)
{
    m_setUniform(position_stencil_scale_def.layout.location, s_work_group_scale);
}

GL::GLsizeiptr GenerationKernel::setArgs(const glm::vec3 &world_scale, float footprint, glm::vec2 lower_bound,
                                         glm::vec2 upper_bound)
{
    // world scale
    m_setUniform(world_scale_def.layout.location, world_scale);

    // bounds
    m_setUniform(lower_bound_def.layout.location, lower_bound);
    m_setUniform(upper_bound_def.layout.location, upper_bound);

    // normalization factor
    {
        const auto norm_factor = footprint / glm::vec2(world_scale.x, world_scale.z);
        m_setUniform(norm_factor_def.layout.location, norm_factor);
    }

    const glm::vec2 work_group_footprint = glm::vec2(work_group_size) * s_work_group_scale * footprint;

    // work group offset
    {
        const glm::uvec2 offset {lower_bound / work_group_footprint};
        m_setUniform(work_group_offset_def.layout.location, offset);
    }

    m_num_work_groups = glm::uvec2((upper_bound - lower_bound) / work_group_footprint) + 1u;

    const auto num_candidates = m_num_work_groups * work_group_size;
    return num_candidates.x * num_candidates.y;
}

void GenerationKernel::dispatchCompute() const
{
    useProgram();
    gl.DispatchCompute(m_num_work_groups.x, m_num_work_groups.y, 1);
}

void GenerationKernel::setPositionStencil(
        const std::array<std::array<glm::vec2, work_group_size.y>, work_group_size.x> &positions)
{
    auto location = position_stencil_def.layout.location;

    if (location < 0)
        return;

    for (const auto& sub_array : positions)
    {
        m_setUniform(location, work_group_size.y, sub_array.data());
        location += work_group_size.y;
    }
}

} // placement