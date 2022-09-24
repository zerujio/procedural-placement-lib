#include "placement.hpp"

#include "glutils/program.hpp"
#include "glutils/guard.hpp"
#include "glutils/buffer.hpp"
#include "glutils/glsl_syntax.hpp"
#include "glutils/gl.hpp"

#include <glm/vector_relational.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <stdexcept>
#include <sstream>

namespace placement {

    using namespace glutils;

    constexpr Definition height_map_def
    {
        .layout {.binding = 0},
        .storage = StorageQualifier::uniform,
        .type = Type::sampler2D,
        .name = "u_height_map"
    };

    constexpr Definition density_map_def
    {
        .layout {.binding = 1},
        .storage = StorageQualifier::uniform,
        .type = Type::sampler2D,
        .name = "u_density_map"
    };

    constexpr Definition world_size_def
    {
        .layout {.location = 0},
        .storage = StorageQualifier::uniform,
        .type = Type::vec3_,
        .name = "u_world_scale"
    };

    constexpr Definition index_offset_def
    {
        .layout {.location = 1},
        .storage = StorageQualifier::uniform,
        .type = Type::uvec2_,
        .name = "u_index_offset"
    };

    constexpr Definition footprint_def
    {
        .layout {.location = 2},
        .storage = StorageQualifier::uniform,
        .type = Type::vec2_,
        .name = "u_norm_footprint"
    };

    constexpr Definition lower_bound_def
    {
        .layout {.location = 3},
        .storage = StorageQualifier::uniform,
        .type = Type::vec2_,
        .name = "u_norm_lower_bound"
    };

    constexpr Definition upper_bound_def
    {
        .layout {.location = 4},
        .storage = StorageQualifier::uniform,
        .type = Type::vec2_,
        .name = "u_norm_upper_bound"
    };

    constexpr int position_buffer_binding = 0;
    constexpr int index_buffer_binding = 1;
    static auto compute_shader_main_src =
R"gl(
layout(binding = 0, std430) restrict writeonly
buffer PositionBuffer
{
    vec3 position_buffer[][4][4];
};

layout(binding = 1, std430) restrict writeonly
buffer IndexBuffer
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
    const vec2 tex_coord = (gl_GlobalInvocationID.xy + u_index_offset) * u_norm_footprint * 2.0f;
    const float normalized_height = texture(u_height_map, tex_coord).x;
    const vec3 position = vec3(tex_coord.x, normalized_height, tex_coord.y) * u_world_scale;

    // validity
    const float density_value = texture(u_density_map, tex_coord).x;
    const float threshold_value = dithering_matrix[gl_LocalInvocationID.x][gl_LocalInvocationID.y];
    const bool is_valid =
            all(greaterThanEqual(tex_coord, u_norm_lower_bound))
            && all(lessThan(tex_coord, u_norm_upper_bound))
            && density_value > threshold_value;

    // write results
    position_buffer[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = position;
    index_buffer[group_index][gl_LocalInvocationID.x][gl_LocalInvocationID.y] = uint(is_valid);
}
)gl";
/*
def reduce(a):
    group_size = 1
    while group_size < len(a):
        for i in range(len(a)):
            group_index = i // group_size
            if group_index % 2:
                last_of_prev_group = group_index * group_size - 1
                a[i] += a[last_of_prev_group]
        group_size <<= 1
    return a
*/

    static auto getComputeShaderSource() -> const std::string&
    {
        static std::string source_string;
        if (source_string.empty())
        {
            std::ostringstream oss;
            oss << "#version 440 core\n\nlayout(local_size_x = 4, local_size_y = 4) in;\n\n"
                << height_map_def       << '\n'
                << density_map_def      << '\n'
                << world_size_def       << '\n'
                << index_offset_def     << '\n'
                << footprint_def        << '\n'
                << lower_bound_def      << '\n'
                << upper_bound_def      << '\n'
                << compute_shader_main_src;
            source_string = oss.str();
        }
        return source_string;
    }

    static auto getComputeProgram() -> Program
    {
        static Program program;
        if (!gl.IsProgram(program.getName()))
        {
            Guard<Shader> shader {Shader::Type::compute};
            auto c_str = getComputeShaderSource().c_str();
            shader->setSource(1, &c_str);
            shader->compile();
            if (shader->getParameter(Shader::Parameter::compile_status) != GL_TRUE)
            {
                const int len = shader->getParameter(Shader::Parameter::info_log_length);
                std::string log;
                log.resize(len);
                shader->getInfoLog(len, nullptr, log.data());
                throw std::runtime_error("compute shader compilation failed:\n" + std::move(log));
            }

            program = Program::create();
            program.attachShader(*shader);
            program.link();
            if (program.getParameter(Program::Parameter::link_status) != GL_TRUE)
            {
                const int len = program.getParameter(Program::Parameter::info_log_length);
                std::string log;
                log.resize(len);
                program.getInfoLog(len, nullptr, log.data());
                throw std::runtime_error("compute shader linking failed:\n" + std::move(log));
            }
        }
        return program;
    }

    void loadGL(GLloader gl_loader)
    {
        if (!glutils::loadGLContext(gl_loader))
            throw std::runtime_error("failed to load OpenGL procedures");
    }

    std::vector<glm::vec3> computePlacement(const WorldData& world_data, float footprint,
                                            glm::vec2 lower_bound, glm::vec2 upper_bound)
    {
        if (! glm::all(glm::lessThanEqual(lower_bound, upper_bound)))
            return {};

        auto program = getComputeProgram();
        program.use();

        // bind height map
        gl.BindTextureUnit(height_map_def.layout.binding, world_data.height_texture);

        // bind density map
        gl.BindTextureUnit(density_map_def.layout.binding, world_data.density_texture);

        // world scale
        gl.Uniform3f(world_size_def.layout.location, world_data.scale.x, world_data.scale.y, world_data.scale.z);
        const glm::vec2 h_world_scale {world_data.scale.x, world_data.scale.z};

        // normalized footprint
        {
            const auto norm_footprint = glm::vec2(footprint) / h_world_scale;
            gl.Uniform2f(footprint_def.layout.location, norm_footprint.x, norm_footprint.y);
        }

        // normalized bounds
        {
            const auto norm_lower_bound = lower_bound / h_world_scale;
            const auto norm_upper_bound = upper_bound / h_world_scale;
            gl.Uniform2f(lower_bound_def.layout.location, norm_lower_bound.x, norm_lower_bound.y);
            gl.Uniform2f(upper_bound_def.layout.location, norm_upper_bound.x, norm_upper_bound.y);
        }

        // index offset
        {
            const glm::uvec2 index_offset {lower_bound / (2 * footprint)};
            gl.Uniform2ui(index_offset_def.layout.location, index_offset.x, index_offset.y);
        }

        // calculate workgroup count
        const glm::uvec2 index_count {(upper_bound - lower_bound) / (2 * footprint)};
        const glm::uvec2 workgroup_count = index_count / 4u + 1u;

        // allocate and bind output buffer
        Guard<Buffer> position_buffer;
        position_buffer->allocateImmutable(sizeof(glm::vec3) * workgroup_count.x * workgroup_count.y * 16,
                                         Buffer::StorageFlags::dynamic_storage | Buffer::StorageFlags::map_read);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, position_buffer_binding, position_buffer->getName());

        Guard<Buffer> index_buffer;
        index_buffer->allocateImmutable(sizeof(uint) * workgroup_count.x * workgroup_count.y * 16,
                                        Buffer::StorageFlags::dynamic_storage | Buffer::StorageFlags::map_read);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, index_buffer_binding, index_buffer->getName());

        // dispatch compute
        gl.DispatchCompute(workgroup_count.x, workgroup_count.y, 1);

        // read generated placement data
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        const auto* positions = static_cast<const glm::vec3 *>(position_buffer->map(Buffer::AccessMode::read_only));
        const auto* indices = static_cast<const uint*>(index_buffer->map(Buffer::AccessMode::read_only));
        if (!positions or !indices)
        {
            throw std::runtime_error("output buffer mapping failed");
        }

        std::vector<glm::vec3> valid_positions;
        valid_positions.reserve(index_count.x * index_count.y);

        for (int i = 0; i < workgroup_count.x * workgroup_count.y * 16; i++)
        {
            const auto point = positions[i];
            const uint valid = indices[i];
            if (valid)
                valid_positions.emplace_back(point);
        }

        position_buffer->unmap();
        index_buffer->unmap();

        return valid_positions;
    }
} // placement