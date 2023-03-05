#include "glutils/vertex_array.hpp"
#include "glutils/buffer.hpp"

#include "simple-renderer/glsl_definitions.hpp"
#include "simple-renderer/renderer.hpp"

#include "placement/placement.hpp"
#include "placement/density_map.hpp"

#include "example-common.hpp"

#include "stb_image.h"

#include "imgui.h"

#include "glm/vec3.hpp"

#include <array>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <vector>

struct STBImageError : public std::runtime_error
{
    STBImageError() : std::runtime_error(stbi_failure_reason())
    {}
};

class ImageFile
{
public:
    ImageFile(const char *filename, int desired_channels = 0) :
            m_data(stbi_load(filename, &m_size.x, &m_size.y, &m_channels, desired_channels), stbi_image_free)
    {
        if (!m_data)
            throw STBImageError();
    }

    [[nodiscard]] int getChannels() const
    { return m_channels; }

    [[nodiscard]] glm::ivec2 getSize() const
    { return m_size; }

    [[nodiscard]] const stbi_uc *getData() const
    { return m_data.get(); }

private:
    int m_channels{0};
    glm::ivec2 m_size{0, 0};
    std::unique_ptr<stbi_uc[], decltype(&stbi_image_free)> m_data;
};

unsigned int loadTexture(const char *filename)
{
    ImageFile image_file{filename};

    using GL::gl;

    unsigned int texture;
    gl.GenTextures(1, &texture);

    gl.BindTexture(GL_TEXTURE_2D, texture);

    {
        constexpr std::array<GLenum, 4> formats{GL_RED, GL_RG, GL_RGB, GL_RGBA};
        const GLenum format = formats[image_file.getChannels() - 1];

        gl.TexImage2D(GL_TEXTURE_2D, 0, format, image_file.getSize().x, image_file.getSize().y, 0, format,
                      GL_UNSIGNED_BYTE, image_file.getData());
    }

    gl.GenerateMipmap(GL_TEXTURE_2D);

    return texture;
}

std::map<std::string, GLuint> loadTexturesFromDirectory(const std::string &directory)
{
    std::map<std::string, GLuint> textures;

    for (const auto &entry: std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file()) continue;

        try
        { textures.emplace(entry.path().filename(), loadTexture(entry.path().c_str())); }

        catch (std::exception &e)
        { std::cout << "couldn't load " << entry.path() << ": " << e.what() << "\n"; }
    }

    return textures;
}

std::pair<simple::Mesh, simple::ShaderProgram> makeAxes()
{
    using namespace simple;

    const std::array<glm::vec3, 6> positions{glm::vec3(0.0f), {1, 0, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}, {0, 0, 1}};
    const std::array<glm::vec3, 6> normals{glm::vec3{1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}};

    Mesh mesh{ positions, normals, {}};

    mesh.setDrawMode(DrawMode::lines);

    ShaderProgram program
            {
                    "out vec3 vertex_color;"
                    "void main() "
                    "{"
                    "   gl_Position = proj_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0f);"
                    "   vertex_color = vertex_normal;"
                    "}",
                    "in vec3 vertex_color;"
                    "void main() { frag_color = vec4(vertex_color, 1.0f); }"
            };

    return {std::move(mesh), std::move(program)};

}

const std::vector<glm::vec3> &getCubePositions()
{
    static const std::vector<glm::vec3> vertex_positions
            {
                    // face 0
                    {.5,  .5,  -.5},
                    {.5,  -.5, -.5},
                    {-.5, .5,  -.5},
                    {-.5, -.5, -.5},
                    // face 1
                    {.5,  .5,  .5},
                    {.5,  -.5, .5},
                    {-.5, .5,  .5},
                    {-.5, -.5, .5},
                    // face 2
                    {.5,  .5,  .5},
                    {.5,  .5,  -.5},
                    {-.5, .5,  .5},
                    {-.5, .5,  -.5},
                    // face 3
                    {.5,  -.5, .5},
                    {.5,  -.5, -.5},
                    {-.5, -.5, .5},
                    {-.5, -.5, -.5},
                    // face 4
                    {.5,  .5,  .5},
                    {.5,  .5,  -.5},
                    {.5,  -.5, .5},
                    {.5,  -.5, -.5},
                    // face 5
                    {-.5, .5,  .5},
                    {-.5, .5,  -.5},
                    {-.5, -.5, .5},
                    {-.5, -.5, -.5},
            };

    return vertex_positions;
}

const std::vector<glm::vec3> &getCubeNormals()
{
    static const std::vector<glm::vec3> vertex_normals
            {
                    {.0,  .0,  -1.},
                    {.0,  .0,  -1.},
                    {.0,  .0,  -1.},
                    {.0,  .0,  -1.},
                    {.0,  .0,  1.},
                    {.0,  .0,  1.},
                    {.0,  .0,  1.},
                    {.0,  .0,  1.},
                    {0.,  1.,  0.},
                    {0.,  1.,  0.},
                    {0.,  1.,  0.},
                    {0.,  1.,  0.},
                    {0.,  -1., 0.},
                    {0.,  -1., 0.},
                    {0.,  -1., 0.},
                    {0.,  -1., 0.},
                    {1.,  0.,  0.},
                    {1.,  0.,  0.},
                    {1.,  0.,  0.},
                    {1.,  0.,  0.},
                    {-1., 0.,  0.},
                    {-1., 0.,  0.},
                    {-1., 0.,  0.},
                    {-1., 0.,  0.},
            };
    return vertex_normals;
}

const std::vector<glm::vec2> &getCubeUVs()
{
    static const std::vector<glm::vec2> vertex_uvs
            {
                    {1., 0.},
                    {1., 1.},
                    {0., 0.},
                    {0., 1.},
                    {1., 0.},
                    {1., 1.},
                    {0., 0.},
                    {0., 1.},
                    {1., 0.},
                    {1., 1.},
                    {0., 0.},
                    {0., 1.},
                    {1., 0.},
                    {1., 1.},
                    {0., 0.},
                    {0., 1.},
                    {1., 0.},
                    {1., 1.},
                    {0., 0.},
                    {0., 1.},
                    {1., 0.},
                    {1., 1.},
                    {0., 0.},
                    {0., 1.},
            };
    return vertex_uvs;
}

const std::vector<unsigned int> &getCubeIndices()
{
    static const std::vector<unsigned int> indices
            {
                    1, 3, 0, 0, 3, 2,    // face 0
                    6, 5, 4, 5, 6, 7,    // face 1
                    9, 10, 8, 10, 9, 11,  // face 2
                    14, 13, 12, 13, 14, 15, // face 3
                    19, 16, 18, 16, 19, 17, // face 4
                    22, 21, 23, 21, 22, 20  // face 5
            };
    return indices;
}

simple::Mesh makeCubeMesh()
{
    return simple::Mesh(getCubePositions(), getCubeNormals(), getCubeUVs(), getCubeIndices());
}

std::vector<glm::vec3> generateCirclePositions(unsigned int num_vertices)
{
    std::vector<glm::vec3> vertices;
    vertices.reserve(num_vertices);
    for (int i = 0; i < num_vertices; i++)
    {
        const float angle = static_cast<float>(i) * 2.f * glm::pi<float>() / static_cast<float>(num_vertices);
        vertices.emplace_back(glm::cos(angle), glm::sin(angle), 0.f);
    }
    return vertices;
}

SimpleInstancedMesh::SimpleInstancedMesh(const std::vector<glm::vec3> &vertices,
                                         const std::vector<glm::vec3> &vertex_normals,
                                         const std::vector<glm::vec2> &vertex_texcoords,
                                         const std::vector<unsigned int> &indices) :
        m_vertex_count(vertices.size()),
        m_index_count(indices.size()),
        m_instance_count(0)
{
    if (vertices.empty())
        throw std::runtime_error("vertices can't be empty");

    {
        const std::size_t vertex_data_size = vertices.size() * sizeof(glm::vec3);
        const std::size_t index_data_size = indices.size() * sizeof(unsigned int);

        std::byte init_data[vertex_data_size + index_data_size];
        memcpy(init_data, vertices.data(), vertex_data_size);
        memcpy(init_data + vertex_data_size, indices.data(), index_data_size);

        m_main_buffer.allocateImmutable(vertex_data_size + index_data_size, GL::BufferHandle::StorageFlags::none,
                                         init_data);
    }

    m_vertex_array.bindVertexBuffer(s_main_buffer_binding, m_main_buffer, 0, sizeof(glm::vec3));

    const auto position_location = simple::vertex_position_def.layout.location;
    m_vertex_array.bindAttribute(position_location, s_main_buffer_binding);
    m_vertex_array.setAttribFormat(position_location,
                                   GL::VertexAttributeLength::_3,
                                   GL::VertexAttributeBaseType::_float,
                                   false, 0);
    m_vertex_array.enableAttribute(position_location);

    if (!indices.empty())
        m_vertex_array.bindElementBuffer(m_main_buffer);

    m_vertex_array.bindVertexBuffer(s_instance_buffer_binding, m_instance_buffer, 0, sizeof(glm::vec4));
    m_vertex_array.setBindingDivisor(s_instance_buffer_binding, 1);

    m_vertex_array.bindAttribute(instance_attr_location, s_instance_buffer_binding);
    m_vertex_array.setAttribFormat(instance_attr_location,
                                   GL::VertexAttributeLength::_3,
                                   GL::VertexAttributeBaseType::_float,
                                   false, 0);
    m_vertex_array.enableAttribute(instance_attr_location);

    m_vertex_array.bindAttribute(instance_attr_location + 1, s_instance_buffer_binding);
    m_vertex_array.setAttribIFormat(instance_attr_location + 1,
                                    GL::VertexAttributeLength::_1,
                                    GL::VertexAttributeBaseType::_uint,
                                    sizeof(glm::vec3));
    m_vertex_array.enableAttribute(instance_attr_location + 1);
}

void SimpleInstancedMesh::updateInstanceData(const placement::Result &result)
{
    constexpr GLsizeiptr instance_alignment = sizeof(glm::vec4);

    m_instance_count = result.getElementArrayLength();
    m_instance_buffer.allocate(m_instance_count * instance_alignment, GL::Buffer::Usage::static_draw);

    result.copyAll(m_instance_buffer);
}

void SimpleInstancedMesh::collectDrawCommands(const CommandCollector &collector) const
{
    if (m_index_count)
        collector.emplace(simple::DrawElementsInstancedCommand(m_draw_mode,
                                                               m_index_count,
                                                               simple::IndexType::unsigned_int,
                                                               m_vertex_count * sizeof(glm::vec3),
                                                               m_instance_count),
                          m_vertex_array);
    else
        collector.emplace(simple::DrawArraysInstancedCommand(m_draw_mode, 0, m_vertex_count, m_instance_count),
                          m_vertex_array);
}

[[nodiscard]]
std::string loadTextFileToString(const std::string& filename)
{
    return (std::stringstream() << std::ifstream(filename).rdbuf()).str();
}

simple::ShaderProgram loadShaderProgram(const std::string& vertex_shader_file_path,
                                        const std::string& fragment_shader_file_path)
{
    return {loadTextFileToString(vertex_shader_file_path), loadTextFileToString(fragment_shader_file_path)};
}

placement::ComputeShaderProgram loadComputeShaderProgram(const std::string& compute_shader_file_path)
{
    return placement::ComputeShaderProgram{loadTextFileToString(compute_shader_file_path)};
}