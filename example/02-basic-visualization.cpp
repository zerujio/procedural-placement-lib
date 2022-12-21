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

int main()
{
    GLFWInitGuard glfw_init_guard;
    Window window {"02 - Basic Visualization"};

    //placement
    GLuint densitymap = loadTexture("assets/heightmap.png"); // deliberately using heightmap as densitymap
    GLuint heightmap = loadTexture("assets/black.png");

    placement::PlacementPipeline pipeline;
    pipeline.setDensityTexture(densitymap);
    pipeline.setHeightTexture(heightmap);
    pipeline.setWorldScale({1.f, 1.f, -1.f});
    pipeline.setRandomSeed(89581751);

    pipeline.computePlacement(0.001f, glm::vec2(0.0f), glm::vec2(1.0f));

    // copy results between gpu buffers
    using namespace glutils;

    Buffer buffer = Buffer::create();
    buffer.allocate(pipeline.getResultsSize() * sizeof(glm::vec4), Buffer::Usage::dynamic_draw);

    pipeline.copyResultsToGPUBuffer(buffer.getName());

    VertexArray vertex_array = VertexArray::create();
    vertex_array.bindVertexBuffer(0, buffer, 0, sizeof(glm::vec4));
    vertex_array.bindAttribute(0 /*attrib index*/, 0 /*buffer index*/);
    vertex_array.setAttribFormat(0 /*attrib index*/, VertexArray::AttribSize::three, VertexArray::AttribType::float_, false, 0);
    vertex_array.enableAttribute(0);

    // rendering
    simple::Renderer renderer;

    simple::Camera camera;

    simple::ShaderProgram program ("void main() {gl_Position = vec4(vertex_position.xzy * vec3(2.f, 2.f, 1.f) - vec3(1.f, 1.f, 0.f), 1.0f);}",
                                   "void main() {frag_color = vec4(1.0f);}");

    simple::Mesh point_mesh (Guard<Buffer>(buffer), Guard<VertexArray>(vertex_array), simple::DrawMode::points,
                             pipeline.getResultsSize(), 0);

    while (!glfwWindowShouldClose(window.get()))
    {
        renderer.draw(point_mesh, program, glm::mat4(1.0f));

        renderer.finishFrame(camera);

        glfwSwapBuffers(window.get());
        glfwPollEvents();
    }
}
