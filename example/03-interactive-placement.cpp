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
#include <filesystem>

void drawDensityMapUI(placement::DensityMap &density_map,
                      const std::vector<std::pair<std::string, GLuint>> &available_textures)
{
    auto current_texture = std::find_if(available_textures.begin(), available_textures.end(),
                                        [&](const std::pair<std::string, GLuint> &pair)
                                        { return pair.second == density_map.texture; });

    ImGui::Text("Density map texture:");
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * .7f);
    if (ImGui::BeginCombo("", current_texture->first.c_str()))
    {
        for (const auto &[filename, gl_object]: available_textures)
        {
            ImGui::PushID(filename.c_str());
            if (ImGui::Selectable(filename.c_str()))
                density_map.texture = gl_object;
            ImGui::PopID();
        }

        ImGui::EndCombo();
    }
    ImGui::PopItemWidth();

    ImGui::PushItemWidth(45);
    ImGui::InputFloat("Scale", &density_map.scale);
    ImGui::InputFloat("Offset", &density_map.offset);
    ImGui::InputFloat("Min. value", &density_map.min_value);
    ImGui::InputFloat("Max. value", &density_map.max_value);
    ImGui::PopItemWidth();
}

std::vector<std::pair<std::string, GLuint>> loadTexturesFromDirectory(const std::string &directory)
{
    std::vector<std::pair<std::string, GLuint>> textures;

    for (const auto &entry: std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file()) continue;

        try
        { textures.emplace_back(entry.path().filename(), loadTexture(entry.path().c_str())); }

        catch (std::exception &e)
        { std::cout << "couldn't load " << entry.path() << ": " << e.what() << "\n"; }
    }

    return textures;
}

class PlacementBounds
{
public:
    PlacementBounds(glm::vec2 lower, glm::vec2 upper) : m_lower(lower), m_upper(upper), m_transform(m_makeTransform())
    {}

    [[nodiscard]] glm::vec2 getLower() const
    { return m_lower; }

    [[nodiscard]] glm::vec2 getUpper() const
    { return m_upper; }

    [[nodiscard]] glm::vec2 getMaxUpper() const
    { return m_max_upper; }

    void setLower(glm::vec2 lower)
    {
        m_lower = glm::clamp(lower, {0, 0}, m_upper);
        m_updateTransform();
    }

    void setUpper(glm::vec2 upper)
    {
        m_upper = glm::min(upper, m_max_upper);
        m_lower = glm::min(m_lower, m_upper);
        m_updateTransform();
    }

    void setMaxUpper(glm::vec2 max_upper)
    {
        m_max_upper = max_upper;
        m_upper = glm::min(m_upper, m_max_upper);
        m_lower = glm::min(m_lower, m_upper);
        m_updateTransform();
    }

    void setPosition(glm::vec3 new_position)
    {
        m_position = new_position;
        m_updateTransform();
    }

    [[nodiscard]] const glm::mat4 &getTransform() const
    { return m_transform; }

private:
    glm::mat4 m_makeTransform()
    {
        return glm::scale(glm::translate(glm::mat4(1), glm::vec3(m_lower, 0) + m_position),
                          glm::vec3(m_upper - m_lower, 1));
    }

    void m_updateTransform()
    {
        m_transform = m_makeTransform();
    }

    glm::vec2 m_lower;
    glm::vec2 m_upper;
    glm::vec2 m_max_upper{m_upper};
    glm::vec3 m_position{};
    glm::mat4 m_transform;
};

constexpr glm::vec2 initial_window_size{1024, 768};

int main()
{
    GLFW::InitGuard glfw;

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    GLFW::Window window{"03 - Interactive placement", initial_window_size};

    GL::enableDebugCallback();

    const auto density_textures = loadTexturesFromDirectory("assets/densitymaps");
    const auto heightmap_texture_filename = "assets/heightmap.png";

    // compute positions
    placement::PlacementPipeline pipeline;
    placement::WorldData world_data{/* scale= */ {100.0f, 100.0f, 10.0f},
            /* heightmap= */ loadTexture(heightmap_texture_filename)};
    placement::LayerData layer_data{/* footprint= */ 0.1f,
            /* densitymaps= */{{density_textures[0].second, 0.33f},
                               {density_textures[1].second, .33f},
                               {density_textures[2].second, .33f}}};

    PlacementBounds placement_bounds{{0, 0}, world_data.scale};
    placement_bounds.setPosition({0, 0, world_data.scale.z / 2.f});

    auto future_result = pipeline.computePlacement(world_data, layer_data, placement_bounds.getLower(),
                                                   placement_bounds.getUpper());
    float future_result_footprint = layer_data.footprint;
    bool waiting_for_results = false;

    // draw
    simple::Renderer renderer;

    simple::Camera camera;

    struct
    {
        float fov_y = glm::pi<float>() / 2.f;
        float near_plane = 0.01f;
        float far_plane = 1000.f;
        float aspect_ratio = initial_window_size.x / initial_window_size.y;
    } camera_projection;

    const auto update_camera_projection = [&]()
    {
        camera.setProjectionMatrix(glm::perspective(camera_projection.fov_y, camera_projection.aspect_ratio,
                                                    camera_projection.near_plane, camera_projection.far_plane));
    };

    update_camera_projection();
    window.setFramebufferSizeCallback([&](GLFW::Window &, int x_size, int y_size)
                                      {
                                          camera_projection.aspect_ratio = float(x_size) / float(y_size);
                                          update_camera_projection();
                                      });

    CameraController camera_controller{camera, window};
    camera_controller.setMaxRadius(100.0f);
    camera_controller.setRadius(25.0f);
    camera_controller.setAngle({glm::pi<float>() * 5. / 4., glm::pi<float>() / 3.0f});
    camera_controller.setMaxPosition({100.f, 100.f});

    simple::ShaderProgram simple_program
            {
                    "void main() { gl_Position = proj_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0f); }",
                    "void main() { frag_color = vec4(1.0f); }"
            };

    simple::ShaderProgram program
            {
                    R"gl(
layout(location = 3) in vec3 instance_offset;
layout(location = 4) in uint layer_index;

out vec3 layer_color;

const vec3 layer_colors[3] = {vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1)};

void main()
{
    const vec4 local_position = model_matrix * vec4(vertex_position, 1.f);
    gl_Position = proj_matrix * view_matrix * (local_position + vec4(instance_offset, 0.0f));
    layer_color = layer_colors[layer_index];
}
)gl",
                    R"gl(
in vec3 layer_color;
void main() {frag_color = vec4(layer_color, 1.0f);}
)gl"};

    // meshes
    const auto [axes_mesh, axes_program] = makeAxes();

    simple::Mesh square_mesh(std::array<glm::vec3, 4>{glm::vec3(0), {0, 1, 0}, {1, 1, 0}, {1, 0, 0}},
                             std::array<glm::vec3, 4>{glm::vec3{1, .5, 0}, {1, .5, 0}, {1, .5, 0}, {1, .5, 0}},
                             {});
    square_mesh.setDrawMode(simple::DrawMode::line_loop);

    SimpleInstancedMesh instanced_mesh{generateCirclePositions(12)};
    instanced_mesh.setDrawMode(simple::DrawMode::line_loop);
    instanced_mesh.updateInstanceData(
            future_result.readResult()); // generated positions are copied to the instance position buffer

    simple::Mesh cube_lines{
            std::array<glm::vec3, 8>
                    {glm::vec3{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}, {0, 1, 0}, {0, 1, 1}, {1, 1, 1}, {1, 1, 0}},
            {},
            {},
            std::array<uint, 24>
                    {0, 1, 0, 3, 0, 4,
                     2, 1, 2, 3, 2, 6,
                     5, 1, 5, 4, 5, 6,
                     7, 3, 7, 4, 7, 6}
    };
    cube_lines.setDrawMode(simple::DrawMode::lines);

    // transformation matrices
    const glm::mat4 identity_matrix{1.0f};
    glm::mat4 world_scale_transform = glm::scale(identity_matrix, world_data.scale);
    glm::mat4 position_marker_transform = glm::scale(identity_matrix, glm::vec3(layer_data.footprint / 2.0f));

    ImGuiContextWrapper imgui_context;
    ImGuiImplWrapper imgui_impl{window.get(), true};

    auto prev_time = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window.get()))
    {
        glfwPollEvents();

        imgui_impl.newFrame();
        ImGui::NewFrame();

        auto curr_time = std::chrono::steady_clock::now();
        std::chrono::duration<float> delta_time = curr_time - prev_time;
        prev_time = curr_time;

        // check for pending results
        if (waiting_for_results && future_result.isReady())
        {
            auto result = future_result.readResult();
            instanced_mesh.updateInstanceData(result);

            for (int i = 0; i < result.getNumClasses(); i++)
                std::cout << "Element count for layer " << i << ": " << result.getClassElementCount(i) << "\n";
            std::cout << "\n";

            position_marker_transform = glm::scale(identity_matrix, glm::vec3(future_result_footprint / 2.0f));
            waiting_for_results = false;
        }

        // draw UI

        if (ImGui::Begin("Settings"))
        {
            ImGui::Text("Frame time: %f s.\nFrame rate: %f FPS", delta_time.count(), 1.f / delta_time.count());

            ImGui::Separator();

            // World Data
            ImGui::Text("World Data");
            if (ImGui::DragFloat3("World scale", glm::value_ptr(world_data.scale), 1.0f, 0.001, 1000))
            {
                world_scale_transform = glm::scale(identity_matrix, world_data.scale);
                camera_controller.setMaxPosition({world_data.scale.x, world_data.scale.y});
                layer_data.footprint = glm::min(layer_data.footprint, glm::min(world_data.scale.x, world_data.scale.y));
                placement_bounds.setMaxUpper(world_data.scale);
                placement_bounds.setPosition({0, 0, world_data.scale.z});
            }
            ImGui::Text("Heightmap: %s", heightmap_texture_filename);

            ImGui::Separator();

            // Layer Data
            ImGui::Text("Layer Data");
            ImGui::DragFloat("Footprint", &layer_data.footprint);

            if (ImGui::BeginListBox("Layers", {0.f, ImGui::GetContentRegionAvail().y -
                                                    ImGui::GetTextLineHeightWithSpacing() * 1.5f}))
            {
                for (std::size_t i = 0; i < layer_data.densitymaps.size(); i++)
                {
                    ImGui::PushID(i);
                    ImGui::Text("[%ld]:", i);
                    ImGui::Indent();
                    drawDensityMapUI(layer_data.densitymaps[i], density_textures);
                    ImGui::Unindent();
                    ImGui::PopID();
                }
                ImGui::EndListBox();
            }

            ImGui::Separator();

            // Placement

            {
                glm::vec2 lower_bound = placement_bounds.getLower();
                if (ImGui::DragFloat2("Lower bound", glm::value_ptr(lower_bound)))
                    placement_bounds.setLower(lower_bound);
            }

            {
                glm::vec2 upper_bound = placement_bounds.getUpper();
                if (ImGui::DragFloat2("Upper bound", glm::value_ptr(upper_bound)))
                    placement_bounds.setUpper(upper_bound);
            }

            if (ImGui::Button("Compute placement"))
            {
                future_result = pipeline.computePlacement(world_data, layer_data, placement_bounds.getLower(),
                                                          placement_bounds.getUpper());
                future_result_footprint = layer_data.footprint;
                waiting_for_results = true;
            }
        }
        ImGui::End();

        camera_controller.update(delta_time.count());

        renderer.draw(square_mesh, axes_program, placement_bounds.getTransform());
        renderer.draw(instanced_mesh, program, position_marker_transform);
        renderer.draw(axes_mesh, axes_program, world_scale_transform);
        renderer.draw(cube_lines, simple_program, world_scale_transform);

        renderer.finishFrame(camera);

        ImGui::Render();
        imgui_impl.renderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window.get());
    }
}

// SimpleInstancedMesh

