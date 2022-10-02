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
        .layout{.binding=GenerationKernel::heightmap_tex_unit},
        .storage = StorageQualifier::uniform,
        .type = Type::sampler2D,
        .name = "u_heightmap"
    };

    static const Definition density_tex_def {
        .layout{.binding=GenerationKernel::densitymap_tex_unit},
        .storage = StorageQualifier::uniform,
        .type = Type::sampler2D,
        .name = "u_densitymap"
    };

    static const std::string source_string {
        (std::ostringstream()
        << "#version 450 core\n"
           "layout(local_size_x = 4, local_size_y = 4) in;\n"
        << upper_bound_def << "\n"
        << lower_bound_def << "\n"
        << world_scale_def << "\n"
        << norm_footprint_def << "\n"
        << grid_offset_def << "\n"
        << height_tex_def << "\n"
        << density_tex_def << "\n"
        << R"glsl(
struct Candidate
{
    vec3 position;
    uint index;
};

layout(binding=)glsl" << GenerationKernel::output_buffer_binding << R"glsl()
restrict writeonly
buffer OutputBuffer
{
    Candidate output_buffer[][4][4];
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
    //output_buffer[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = Candidate(position, uint(is_valid));
    output_buffer[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] =
        Candidate(vec3(u_lower_bound, u_norm_footprint), uint(is_valid));
}
)glsl").str()
    };

    GenerationKernel::GenerationKernel()
    {
        Guard<Shader> shader {Shader::Type::compute};
        auto c_str = source_string.c_str();
        shader->setSource(1, &c_str);
        shader->compile();
        if (shader->getParameter(Shader::Parameter::compile_status) != GL_TRUE)
        {
            const int len = shader->getParameter(Shader::Parameter::info_log_length);
            std::string log;
            log.resize(len);
            shader->getInfoLog(len, nullptr, log.data());
            throw std::runtime_error(log);
        }

        m_program->attachShader(*shader);
        m_program->link();
        if (m_program->getParameter(Program::Parameter::link_status) != GL_TRUE)
        {
            const int len = m_program->getParameter(Program::Parameter::info_log_length);
            std::string log;
            log.resize(len);
            m_program->getInfoLog(len, nullptr, log.data());
            throw std::runtime_error(log);
        }
    }

    void GenerationKernel::dispatchCompute(float footprint, glm::vec3 world_scale, glm::vec2 lower_bound,
                                           glm::vec2 upper_bound) const
    {
        m_program->use();

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

        const auto wg = getNumWorkGroups(footprint, lower_bound, upper_bound);
        gl.DispatchCompute(wg.x, wg.y, 1);
    }


} // placement