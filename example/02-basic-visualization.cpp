/*
 * 02 - Basic visualization
 * This example shows how the generated position data can be copied directly to another GPU buffer
 * and then rendered.
 */

#include "example-common.hpp"

#include "placement/placement.hpp"

#include "simple-renderer/renderer.hpp"
#include "glutils/guard.hpp"

#include <memory>

class ResultMesh : public simple::Drawable
{
public:
    explicit ResultMesh(placement::Result result) : m_result(std::move(result))
    {
        const auto &result_buffer = m_result.getBuffer();
        m_vertex_array.bindVertexBuffer(0, result_buffer.gl_object, result_buffer.getElementRange().offset,
                                        sizeof(glm::vec4));
        m_vertex_array.bindAttribute(0 /*attrib index*/, 0 /*buffer index*/);
        m_vertex_array.setAttribFormat(0 /*attrib index*/, GL::VertexArrayHandle::AttribSize::_3,
                                       GL::VertexArrayHandle::AttribType::_float, false, 0);
        m_vertex_array.enableAttribute(0);
    }

    void collectDrawCommands(const CommandCollector &collector) const override
    {
        collector.emplace(simple::DrawArraysCommand(simple::DrawMode::points, 0, m_result.getElementArrayLength()),
                          m_vertex_array);
    }

private:
    GL::VertexArray m_vertex_array;
    placement::Result m_result;
};

int main()
{
    GLFW::InitGuard glfw_init_guard;
    GLFW::Window window {"02 - Basic Visualization"};

    //placement
    GLuint densitymap = loadTexture("assets/heightmap.png"); // deliberately using heightmap as densitymap
    GLuint heightmap = loadTexture("assets/black.png");

    placement::PlacementPipeline pipeline;
    pipeline.setRandomSeed(89581751);

    const placement::WorldData world_data {/*scale=*/{1.f, 1.f, -1.f}, heightmap};
    const placement::LayerData layer_data {/*footprint=*/0.001f, /*densitymaps=*/{{/*texture=*/densitymap}}};

    auto future_results = pipeline.computePlacement(world_data, layer_data, glm::vec2(0.0f), glm::vec2(1.0f));

    // rendering
    simple::Renderer renderer;

    simple::Camera camera;

    simple::ShaderProgram program ("void main() {gl_Position = vec4(vertex_position * vec3(2.f, 2.f, 1.f) - vec3(1.f, 1.f, 0.f), 1.0f);}",
                                   "void main() {frag_color = vec4(1.0f);}");

    // render directly from the result buffer
    ResultMesh mesh (future_results.readResult());

    while (!glfwWindowShouldClose(window.get()))
    {
        renderer.draw(mesh, program, glm::mat4(1.0f));

        renderer.finishFrame(camera);

        glfwSwapBuffers(window.get());
        glfwPollEvents();
    }
}
