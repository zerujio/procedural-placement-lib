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

    const std::string GenerationKernel::source_string {
        (std::ostringstream()
        << "#version 450 core\n"
           "layout(local_size_x = 4, local_size_y = 4) in;\n"
        << lower_bound_def << "\n"
        << upper_bound_def << "\n"
        << world_scale_def << "\n"
        << norm_footprint_def << "\n"
        << grid_offset_def << "\n"
        << height_tex_def << "\n"
        << density_tex_def << "\n"
        << R"glsl(
layout(std430, binding=0)
restrict writeonly
buffer )glsl" << PlacementPipelineKernel::s_position_ssb_name << R"glsl(
{
    vec3 output_buffer[][4][4];
};

layout(std430, binding=1)
restrict writeonly
buffer )glsl" << PlacementPipelineKernel::s_index_ssb_name << R"glsl(
{
    uint index_buffer[][4][4];
};

const mat4 dithering_matrix =
mat4(
    0,  12, 3,  15,
    8,  4,  11, 7,
    2,  14, 1,  13,
    10, 6,  9,  5
) / 16;

void main()
{
    const uint group_index = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_WorkGroupSize.x;

    // position
    const vec2 tex_coord = (gl_GlobalInvocationID.xy + u_grid_offset) * u_norm_footprint * 2.0f;
    const float norm_height = texture(u_heightmap, tex_coord).x;
    const vec3 position = vec3(tex_coord.x, norm_height, tex_coord.y) * u_world_scale;

    // validity
    const float density_value = texture(u_densitymap, tex_coord).x;
    const float threshold_value = dithering_matrix[gl_LocalInvocationID.x][gl_LocalInvocationID.y];
    const bool is_valid = all(greaterThanEqual(position.xz, u_lower_bound))
                        && all(lessThan(position.xz, u_upper_bound))
                        && density_value > threshold_value;

    // write results
    output_buffer[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = position;
    index_buffer[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = uint(is_valid);
}
)glsl").str()
    };

    GenerationKernel::GenerationKernel() :
        PlacementPipelineKernel(source_string),
        m_heightmap_tex(*this, height_tex_def.name),
        m_densitymap_tex(*this, density_tex_def.name)
    {}

    auto GenerationKernel::dispatchCompute(float footprint, glm::vec3 world_scale, glm::vec2 lower_bound,
                                           glm::vec2 upper_bound) const -> glm::uvec2
    {
        useProgram();

        // footprint
        {
            const auto norm_footprint = glm::vec2(footprint) / glm::vec2(world_scale.x, world_scale.z);
            gl.Uniform2f(norm_footprint_def.layout.location, norm_footprint.x, norm_footprint.y);
        }

        // world scale
        gl.Uniform3f(world_scale_def.layout.location, world_scale.x, world_scale.y, world_scale.z);

        // bounds
        gl.Uniform2f(lower_bound_def.layout.location, lower_bound.x, lower_bound.y);
        gl.Uniform2f(upper_bound_def.layout.location, upper_bound.x, upper_bound.y);

        // grid offset
        {
            const glm::uvec2 grid_offset {lower_bound / (2.0f * footprint)};
            gl.Uniform2ui(grid_offset_def.layout.location, grid_offset.x, grid_offset.y);
        }

        const auto wg = computeNumWorkGroups(footprint, lower_bound, upper_bound);
        gl.DispatchCompute(wg.x, wg.y, 1);

        return wg;
    }


} // placement