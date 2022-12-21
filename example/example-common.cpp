#include "example-common.hpp"

#include "stb_image.h"

#include "glm/vec3.hpp"

#include <array>

struct STBImageError : public std::runtime_error
{
    STBImageError() : std::runtime_error(stbi_failure_reason()) {}
};

class ImageFile
{
public:
    ImageFile(const char* filename, int desired_channels = 0) :
    m_data(stbi_load(filename, &m_size.x, &m_size.y, &m_channels, desired_channels), stbi_image_free)
    {
        if (!m_data)
            throw STBImageError();
    }

    [[nodiscard]] int getChannels() const {return m_channels;}
    [[nodiscard]] glm::ivec2 getSize() const {return m_size;}
    [[nodiscard]] const stbi_uc* getData() const {return m_data.get();}

private:
    int m_channels {0};
    glm::ivec2 m_size {0, 0};
    std::unique_ptr<stbi_uc[], decltype(&stbi_image_free)> m_data;
};

unsigned int loadTexture(const char* filename)
{
    ImageFile image_file {filename};

    using glutils::gl;

    unsigned int texture;
    gl.GenTextures(1, &texture);

    gl.BindTexture(GL_TEXTURE_2D, texture);

    {
        constexpr std::array<GLenum, 4> formats {GL_RED, GL_RG, GL_RGB, GL_RGBA};
        const GLenum format = formats[image_file.getChannels() - 1];

        gl.TexImage2D(GL_TEXTURE_2D, 0, format, image_file.getSize().x, image_file.getSize().y, 0, format,
                      GL_UNSIGNED_BYTE, image_file.getData());
    }

    gl.GenerateMipmap(GL_TEXTURE_2D);

    return texture;
}

std::pair<simple::Mesh, simple::ShaderProgram> makeAxes()
{
    using namespace simple;

    Mesh mesh
    {
            {
                    glm::vec3(0.0f), {1, 0, 0},
                    glm::vec3(0.0f), {0, 1, 0},
                    glm::vec3(0.0f), {0, 0, 1}
            },
            {
                    {1, 0, 0}, {1, 0, 0},
                    {0, 1, 0}, {0, 1, 0},
                    {0, 0, 1}, {0, 0, 1}
            }
    };
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

const std::vector<glm::vec3>& getCubePositions()
{
    static const std::vector<glm::vec3> vertex_positions
            {
                    // face 0
                    { .5, .5,-.5},
                    { .5,-.5,-.5},
                    {-.5, .5,-.5},
                    {-.5,-.5,-.5},
                    // face 1
                    { .5, .5, .5},
                    { .5,-.5, .5},
                    {-.5, .5, .5},
                    {-.5,-.5, .5},
                    // face 2
                    { .5, .5, .5},
                    { .5, .5,-.5},
                    {-.5, .5, .5},
                    {-.5, .5,-.5},
                    // face 3
                    { .5,-.5, .5},
                    { .5,-.5,-.5},
                    {-.5,-.5, .5},
                    {-.5,-.5,-.5},
                    // face 4
                    { .5, .5, .5},
                    { .5, .5,-.5},
                    { .5,-.5, .5},
                    { .5,-.5,-.5},
                    // face 5
                    {-.5, .5, .5},
                    {-.5, .5,-.5},
                    {-.5,-.5, .5},
                    {-.5,-.5,-.5},
            };

    return vertex_positions;
}

const std::vector<glm::vec3>& getCubeNormals()
{
    static const std::vector<glm::vec3> vertex_normals
            {
                    { .0, .0,-1.},
                    { .0, .0,-1.},
                    { .0, .0,-1.},
                    { .0, .0,-1.},
                    { .0, .0, 1.},
                    { .0, .0, 1.},
                    { .0, .0, 1.},
                    { .0, .0, 1.},
                    { 0., 1., 0.},
                    { 0., 1., 0.},
                    { 0., 1., 0.},
                    { 0., 1., 0.},
                    { 0.,-1., 0.},
                    { 0.,-1., 0.},
                    { 0.,-1., 0.},
                    { 0.,-1., 0.},
                    { 1., 0., 0.},
                    { 1., 0., 0.},
                    { 1., 0., 0.},
                    { 1., 0., 0.},
                    {-1., 0., 0.},
                    {-1., 0., 0.},
                    {-1., 0., 0.},
                    {-1., 0., 0.},
            };
    return vertex_normals;
}

const std::vector<glm::vec2>& getCubeUVs()
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

const std::vector<unsigned int>& getCubeIndices()
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
        vertices.emplace_back(glm::cos(angle), 0.f, glm::sin(angle));
    }
    return vertices;
}
