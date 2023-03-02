/// This header declares various utilities used in the examples.

#ifndef PROCEDURALPLACEMENTLIB_EXAMPLE_COMMON_HPP
#define PROCEDURALPLACEMENTLIB_EXAMPLE_COMMON_HPP

#include "common/glfw_wrapper.hpp"
#include "common/camera_controller.hpp"
#include "common/imgui_wrapper.hpp"
#include "common/load_obj.hpp"

#include "simple-renderer/mesh.hpp"
#include "simple-renderer/shader_program.hpp"
#include "placement/placement_result.hpp"

#include <utility>

class SimpleInstancedMesh : public simple::Drawable
{
public:
    explicit SimpleInstancedMesh(const std::vector<glm::vec3>& vertices, const std::vector<unsigned int>& indices = {});

    ///
    void updateInstanceData(const placement::Result& result);

    void collectDrawCommands(const CommandCollector &collector) const override;

    [[nodiscard]] simple::DrawMode getDrawMode() const {return m_draw_mode;}
    void setDrawMode(simple::DrawMode draw_mode) {m_draw_mode = draw_mode;}

    /// Vertex attribute index for instanced data.
    static constexpr unsigned int instance_attr_location = 3; // position is location 0, normal are 1, uvs are 2

private:
    static constexpr unsigned int s_main_buffer_binding = 0;
    static constexpr unsigned int s_instance_buffer_binding = s_main_buffer_binding + 1;

    GL::Buffer m_main_buffer;
    GL::Buffer m_instance_buffer;
    GL::VertexArray m_vertex_array;
    uint32_t m_vertex_count;
    uint32_t m_index_count;
    uint32_t m_instance_count;
    simple::DrawMode m_draw_mode = simple::DrawMode::triangles;
};

/// loads a texture from a file and create an OpenGL texture object from it.
GLuint loadTexture(const char* filename);
std::map<std::string, GLuint> loadTexturesFromDirectory(const std::string &directory);

[[nodiscard]]
simple::ShaderProgram loadShaderProgram(const std::string& vertex_shader_file_path,
                                        const std::string& fragment_shader_file_path);

[[nodiscard]]
placement::ComputeShaderProgram loadComputeShaderProgram(const std::string& compute_shader_file_path);

std::pair<simple::Mesh, simple::ShaderProgram> makeAxes();

const std::vector<glm::vec3>& getCubePositions();
const std::vector<glm::vec3>& getCubeNormals();
const std::vector<glm::vec2>& getCubeUVs();
const std::vector<unsigned int>& getCubeIndices();
simple::Mesh makeCubeMesh();

std::vector<glm::vec3> generateCirclePositions(unsigned int num_vertices);

#endif //PROCEDURALPLACEMENTLIB_EXAMPLE_COMMON_HPP
