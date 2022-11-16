#include "placement/generation_kernel.hpp"
#include "gl_context.hpp"

#include "glutils/glsl_syntax.hpp"

#include <string>
#include <sstream>

namespace placement {

    using namespace glutils;

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

    static const Definition norm_footprint_def {
        .layout{.location=3},
        .storage = StorageQualifier::uniform,
        .type = Type::vec2_,
        .name = "u_norm_footprint"
    };

    static const Definition grid_offset_def {
        .layout{.location=4},
        .storage = StorageQualifier::uniform,
        .type = Type::uvec2_,
        .name = "u_grid_offset"
    };

    static const Definition height_tex_def {
        .layout{.binding=GenerationKernel::s_default_heightmap_tex_unit},
        .storage = StorageQualifier::uniform,
        .type = Type::sampler2D,
        .name = "u_heightmap"
    };

    static const Definition density_tex_def {
        .layout{.binding=GenerationKernel::s_default_densitymap_tex_unit},
        .storage = StorageQualifier::uniform,
        .type = Type::sampler2D,
        .name = "u_densitymap"
    };

    const std::string GenerationKernel::s_source_string {
        (std::ostringstream()
                << "#version 450 core\n"
                << "layout(local_size_x = " << GenerationKernel::s_work_group_size.x
                << ", local_size_y = " << GenerationKernel::s_work_group_size.y << ") in;\n"
        << lower_bound_def << "\n"
        << upper_bound_def << "\n"
        << world_scale_def << "\n"
        << norm_footprint_def << "\n"
        << grid_offset_def << "\n"
        << height_tex_def << "\n"
        << density_tex_def << "\n"
        << R"glsl(

layout(std430, binding=)glsl" << PlacementPipelineKernel::default_position_ssb_binding << R"glsl()
restrict writeonly
buffer )glsl" << PlacementPipelineKernel::s_position_ssb_name << R"glsl(
{
    vec3 position_buffer[][gl_WorkGroupSize.x][gl_WorkGroupSize.y];
};

layout(std430, binding=)glsl" << PlacementPipelineKernel::default_index_ssb_binding << R"glsl()
restrict writeonly
buffer )glsl" << PlacementPipelineKernel::s_index_ssb_name << R"glsl(
{
    uint index_buffer[][gl_WorkGroupSize.x][gl_WorkGroupSize.y];
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
    const vec2 tex_coord = (gl_GlobalInvocationID.xy + u_grid_offset) * u_norm_footprint * 2.0f;
    const float norm_height = texture(u_heightmap, tex_coord).x;
    const vec3 position = vec3(tex_coord.x, norm_height, tex_coord.y) * u_world_scale;

    // validity
    const float density_value = texture(u_densitymap, tex_coord).x;
    const float threshold_value = dithering_matrix[gl_LocalInvocationID.x][gl_LocalInvocationID.y] / 64.0f;
    const bool is_valid = all(greaterThanEqual(position.xz, u_lower_bound))
                        && all(lessThan(position.xz, u_upper_bound))
                        && density_value > threshold_value;

    // write results
    position_buffer[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = position;
    index_buffer[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = uint(is_valid);
}
)glsl").str()
    };

    GenerationKernel::GenerationKernel() :
            PlacementPipelineKernel(s_source_string),
            m_heightmap_loc(*this, height_tex_def.name),
            m_densitymap_loc(*this, density_tex_def.name)
    {}

    std::size_t GenerationKernel::setArgs(const glm::vec3 &world_scale, float footprint, glm::vec2 lower_bound,
                                   glm::vec2 upper_bound)
    {
        // footprint
        {
            const auto norm_footprint = glm::vec2(footprint) / glm::vec2(world_scale.x, world_scale.z);
            setUniform(norm_footprint_def.layout.location, norm_footprint);
        }

        // world scale
        setUniform(world_scale_def.layout.location, world_scale);

        // bounds
        setUniform(lower_bound_def.layout.location, lower_bound);
        setUniform(upper_bound_def.layout.location, upper_bound);

        // grid offset
        {
            const glm::uvec2 grid_offset {lower_bound / (2.0f * footprint)};
            setUniform(grid_offset_def.layout.location, grid_offset);
        }

        m_num_work_groups = m_calculateNumWorkGroups(footprint, lower_bound, upper_bound);

        return calculateCandidateCount();
    }

    void GenerationKernel::dispatchCompute() const
    {
        useProgram();
        gl.DispatchCompute(m_num_work_groups.x, m_num_work_groups.y, 1);
        m_ensureOutputVisibility();
    }

    auto GenerationKernel::m_calculateNumWorkGroups(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound)
    -> glm::uvec2
    {
        const glm::uvec2 min_invocations {(upper_bound - lower_bound) / (2.0f * footprint)};
        return min_invocations / s_work_group_size + 1u;
    }


} // placement