/// This example shows how the placement library can be integrated into an existing rendering pipeline.

#include "example-common.hpp"

#include "placement/placement.hpp"

#include "simple-renderer/renderer.hpp"
#include "simple-renderer/glsl_definitions.hpp"

#include "glutils/guard.hpp"
#include "glutils/buffer.hpp"
#include "glutils/vertex_array.hpp"
#include "glutils/debug.hpp"

#include "imgui.h"

#include <vector>
#include <iostream>
#include <chrono>

class SimpleInstancedMesh : public simple::Drawable
{
public:
    SimpleInstancedMesh(const std::vector<glm::vec3>& vertices, const std::vector<unsigned int>& indices = {});

    ///
    void updateInstanceData(GL::BufferHandle buffer, GLintptr offset, std::uint32_t instance_count);

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
    std::uint32_t m_vertex_count;
    std::uint32_t m_index_count;
    std::uint32_t m_instance_count;
    simple::DrawMode m_draw_mode = simple::DrawMode::triangles;
};

int main()
{
    GLFW::InitGuard glfw;

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    GLFW::Window window {"03 - Interactive placement"};

    GL::enableDebugCallback();

    // compute positions
    placement::PlacementPipeline pipeline;
    placement::WorldData world_data{/* scale= */ {100.0f, 10.0f, 100.0f},
                                    /* heightmap= */ loadTexture("assets/heightmap.png")};
    placement::LayerData layer_data{/* footprint= */ std::sqrt(2.0f) * 0.1f,
                                    /* densitymaps= */{{loadTexture("assets/densitymaps/square_gradient.png")}}};

    glm::vec2 lower_bound {0.0f, 0.0f};
    glm::vec2 upper_bound {100.0f, 100.0f};

    auto future_result = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound);

    // draw
    simple::Renderer renderer;

    simple::Camera camera;
    const float camera_fov = glm::pi<float>() / 2.f;
    const float camera_near = 0.01f;
    const float camera_far = 1000.f;

    camera.setProjectionMatrix(glm::perspectiveFov(camera_fov, 600.0f, 600.0f, camera_near, camera_far));
    window.setFramebufferSizeCallback(
            [&camera, camera_far, camera_near, camera_fov](GLFW::Window&, int x_size, int y_size)
            {
                camera.setProjectionMatrix(glm::perspectiveFov(camera_fov,
                                                               float(x_size), float(y_size),
                                                               camera_near, camera_far));
            });

    CameraController camera_controller {camera, window};
    camera_controller.setMaxRadius(100.0f);
    camera_controller.setRadius(25.0f);
    camera_controller.setAngle({glm::pi<float>() * 5. / 4., glm::pi<float>() / 3.0f});
    camera_controller.setMaxPosition({100.f, 100.f});

    simple::ShaderProgram program
    {
R"gl(
layout(location = 3) in vec3 instance_offset;

void main()
{
    const vec4 local_position = model_matrix * vec4(vertex_position, 1.f);
    gl_Position = proj_matrix * view_matrix * (local_position + vec4(instance_offset, 0.0f));
}
)gl",
R"gl(
void main() {frag_color = vec4(1.0f);}
)gl"};

    // meshes
    const auto [axes_mesh, axes_program] = makeAxes();

    simple::Mesh square_mesh {{{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}},
                              {{1, .5, 0}, {1, .5, 0}, {1, .5, 0}, {1, .5, 0}}};
    square_mesh.setDrawMode(simple::DrawMode::line_loop);

    SimpleInstancedMesh instanced_mesh {generateCirclePositions(12)};
    instanced_mesh.setDrawMode(simple::DrawMode::line_loop);
    instanced_mesh.updateInstanceData(pipeline); // generated positions are copied to the instance position buffer

    simple::Mesh cube_lines {
        {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0},
         {0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
        {},
        {},
        {0, 1,  0, 3,  0, 4,
         2, 1,  2, 3,  2, 6,
         5, 1,  5, 4,  5, 6,
         7, 3,  7, 4,  7, 6}};
    cube_lines.setDrawMode(simple::DrawMode::lines);

    // transformation matrices
    const glm::mat4 identity_matrix {1.0f};
    glm::mat4 world_scale_transform = glm::scale(identity_matrix, pipeline.getWorldScale());
    glm::mat4 position_marker_transform = glm::scale(identity_matrix, glm::vec3(footprint));
    glm::vec2 placement_bounds {upper_bound - lower_bound};
    glm::mat4 placement_bounds_transform = glm::translate(glm::scale(identity_matrix,
                                                                     glm::vec3(placement_bounds.x, 1, placement_bounds.y)),
                                                          glm::vec3(lower_bound.x, 0, lower_bound.y));

    ImGuiContextWrapper imgui_context;
    ImGuiImplWrapper imgui_impl {window.get(), true};

    auto prev_time = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window.get()))
    {
        glfwPollEvents();

        imgui_impl.newFrame();
        ImGui::NewFrame();

        auto curr_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> delta_time = curr_time - prev_time;
        prev_time = curr_time;

        ImGui::Begin("Settings");
        ImGui::Text("Frame time: %fs", delta_time.count());

        ImGui::Separator();

        glm::vec3 world_scale = pipeline.getWorldScale();
        if (ImGui::DragFloat3("World scale", glm::value_ptr(world_scale), 1.0f, 0.001, 1000))
        {
            pipeline.setWorldScale(world_scale);
            world_scale_transform = glm::scale(identity_matrix, world_scale);
            camera_controller.setMaxPosition({world_scale.x, world_scale.z});
        }

        ImGui::DragFloat("Footprint", &footprint, .01f, 0.0001, glm::max(world_scale.x, world_scale.y));

        if (ImGui::DragFloat2("Lower bound", glm::value_ptr(lower_bound))
            || ImGui::DragFloat2("Upper bound", glm::value_ptr(upper_bound)))
        {
            upper_bound = glm::clamp(upper_bound, lower_bound, {world_scale.x, world_scale.y});
            lower_bound = glm::clamp(lower_bound, {0, 0}, upper_bound);
            placement_bounds = upper_bound - lower_bound;
            placement_bounds_transform = glm::scale(glm::translate(identity_matrix, {lower_bound.x, 0, lower_bound.y}),
                                                    {placement_bounds.x, 1, placement_bounds.y});
        }
        renderer.draw(square_mesh, axes_program, placement_bounds_transform);

        // positions can be re-calculated at runtime
        if (ImGui::Button("Compute placement"))
        {
            pipeline.computePlacement(footprint, lower_bound, upper_bound);
            instanced_mesh.updateInstanceData(pipeline);
            position_marker_transform = glm::scale(identity_matrix, glm::vec3(footprint));
        }

        ImGui::End();

        camera_controller.update(delta_time.count());

        renderer.draw(instanced_mesh, program, position_marker_transform);
        renderer.draw(axes_mesh, axes_program, world_scale_transform);
        renderer.draw(cube_lines, program, world_scale_transform);

        renderer.finishFrame(camera);

        ImGui::Render();
        imgui_impl.renderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window.get());
    }
}

// SimpleInstancedMesh

SimpleInstancedMesh::SimpleInstancedMesh(const std::vector<glm::vec3> &vertices,
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
        std::memcpy(init_data, vertices.data(), vertex_data_size);
        std::memcpy(init_data + vertex_data_size, indices.data(), index_data_size);

        m_main_buffer.allocateImmutable(vertex_data_size + index_data_size, GL::BufferHandle::StorageFlags::none,
                                         init_data);
    }

    m_vertex_array.bindVertexBuffer(s_main_buffer_binding, m_main_buffer, 0, sizeof(glm::vec3));

    const auto position_location = simple::vertex_position_def.layout.location;
    m_vertex_array.bindAttribute(position_location, s_main_buffer_binding);
    m_vertex_array.setAttribFormat(position_location,
                                    GL::VertexArrayHandle::AttribSize::_3,
                                    GL::VertexArrayHandle::AttribType::_float,
                                    false, 0);
    m_vertex_array.enableAttribute(position_location);

    if (!indices.empty())
        m_vertex_array.bindElementBuffer(m_main_buffer);

    m_vertex_array.bindVertexBuffer(s_instance_buffer_binding, m_instance_buffer, 0, sizeof(glm::vec4));
    m_vertex_array.setBindingDivisor(s_instance_buffer_binding, 1);

    m_vertex_array.bindAttribute(instance_attr_location, s_instance_buffer_binding);
    m_vertex_array.setAttribFormat(instance_attr_location,
                                    GL::VertexArrayHandle::AttribSize::_3,
                                    GL::VertexArrayHandle::AttribType::_float,
                                    false, 0);
    m_vertex_array.enableAttribute(instance_attr_location);
}

void SimpleInstancedMesh::updateInstanceData(const placement::Result &result)
{
    constexpr GLsizeiptr instance_alignment = sizeof(glm::vec4);

    m_instance_count = result.getElementArrayLength();
    m_instance_buffer.allocate(m_instance_count * instance_alignment, GL::Buffer::Usage::static_draw);

    result.copyAll(m_instance_buffer);
}

void SimpleInstancedMesh::collectDrawCommands(const simple::Drawable::CommandCollector &collector) const
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
