#include "placement/placement.hpp"

#include "example-common.hpp"
#include "simple-renderer/renderer.hpp"
#include "simple-renderer/instanced_mesh.hpp"
#include "simple-renderer/texture_2d.hpp"
#include "simple-renderer/image_data.hpp"
#include "glutils/debug.hpp"
#include "imgui.h"

#include <chrono>
#include <iostream>
#include <filesystem>

constexpr glm::vec2 initial_window_size{1024, 768};

struct CameraParams
{
    float fov_y = glm::pi<float>() / 2.f;
    float near_plane = 0.01f;
    float far_plane = 10000.f;
    float aspect_ratio = 1.f;
};

class Camera
{
public:
    explicit Camera(GLFW::Window &window) : m_controller(m_renderer_camera, window)
    {
        glm::ivec2 framebuffer_size;
        glfwGetFramebufferSize(window.get(), &framebuffer_size.x, &framebuffer_size.y);
        m_setAspectRatio(framebuffer_size);

        window.setFramebufferSizeCallback([this](GLFW::Window &, int x_size, int y_size)
                                          { m_setAspectRatio({x_size, y_size}); });
    }

    [[nodiscard]] const simple::Camera &getRendererCamera() const
    { return m_renderer_camera; }

    [[nodiscard]] const CameraController &getController() const
    { return m_controller; }

    [[nodiscard]] CameraController &getController()
    { return m_controller; }

    [[nodiscard]] CameraParams getParams() const
    { return m_params; }

    void setParams(float fov_y, float near_plane, float far_plane)
    {
        m_params.fov_y = fov_y;
        m_params.near_plane = near_plane;
        m_params.far_plane = far_plane;
        m_updateProjectionMatrix();
    }

    [[nodiscard]] float getFovY() const
    { return m_params.fov_y; }

    void setFovY(float angle)
    {
        m_params.fov_y = angle;
        m_updateProjectionMatrix();
    }

    [[nodiscard]] float getNearPlane() const
    { return m_params.near_plane; }

    [[nodiscard]] float getFarPlane() const
    { return m_params.far_plane; }

    void setNearPlane(float z_value)
    {
        m_params.near_plane = z_value;
        m_updateProjectionMatrix();
    }

    void setFarPlane(float z_value)
    {
        m_params.far_plane = z_value;
        m_updateProjectionMatrix();
    }

    void setClipPlanes(float near, float far)
    {
        m_params.near_plane = near;
        m_params.far_plane = far;
        m_updateProjectionMatrix();
    }

    [[nodiscard]] float getAspectRatio() const
    { return m_params.aspect_ratio; }

private:
    void m_setAspectRatio(glm::ivec2 framebuffer_size)
    {
        m_params.aspect_ratio = static_cast<float>(framebuffer_size.x) / static_cast<float>(framebuffer_size.y);
        m_updateProjectionMatrix();
    }

    void m_updateProjectionMatrix()
    {
        m_renderer_camera.setProjectionMatrix(glm::perspective(m_params.fov_y, m_params.aspect_ratio,
                                                               m_params.near_plane, m_params.far_plane));
    }

    simple::Camera m_renderer_camera;
    CameraController m_controller;
    CameraParams m_params;
};

class PhongShader
{
public:
    [[nodiscard]] const simple::ShaderProgram &getRendererProgram() const
    { return m_program; }

    [[nodiscard]] auto lightPosition()
    { return m_program.makeAccessor(m_light_position); }

    [[nodiscard]] auto lightColor()
    { return m_program.makeAccessor(m_light_color); }

    [[nodiscard]] auto viewPosition()
    { return m_program.makeAccessor(m_view_position); }

    [[nodiscard]] auto ambientLightIntensity()
    { return m_program.makeAccessor(m_ambient_light_intensity); }

    [[nodiscard]] auto specularLightIntensity()
    { return m_program.makeAccessor(m_specular_light_intensity); }

    [[nodiscard]] auto colorTextureUnit()
    { return m_program.makeAccessor(m_color_texture); }

    [[nodiscard]] auto specularHighlightFactor()
    { return m_program.makeAccessor(m_specular_highlight_factor); }

private:
    using SP = simple::ShaderProgram;

    simple::ShaderProgram m_program = loadShaderProgram("assets/shaders/phong.vert", "assets/shaders/phong.frag");

    SP::CachedUniform <glm::vec3> m_light_position{m_program.getUniformLocation("u_light_position")};
    SP::CachedUniform <glm::vec3> m_light_color{m_program.getUniformLocation("u_light_color")};
    SP::CachedUniform <glm::vec3> m_view_position{m_program.getUniformLocation("u_view_position")};
    SP::CachedUniform<float> m_ambient_light_intensity{m_program.getUniformLocation("u_ambient_light_intensity")};
    SP::CachedUniform<float> m_specular_light_intensity{m_program.getUniformLocation("u_specular_light_intensity")};
    SP::CachedUniform<float> m_specular_highlight_factor{m_program.getUniformLocation("u_specular_highlight_factor")};
    SP::CachedUniform <uint> m_color_texture{m_program.getUniformLocation("u_color_texture")};
};

[[nodiscard]]
simple::Mesh loadSimpleMesh(const std::string &filename)
{
    const MeshData mesh_data = loadOBJ(filename);
    return {mesh_data.positions, mesh_data.normals, mesh_data.tex_coords, mesh_data.indices};
}

[[nodiscard]]
simple::InstancedMesh loadInstancedMesh(const std::string &filename)
{
    const MeshData mesh_data = loadOBJ(filename);
    return {mesh_data.positions, mesh_data.normals, mesh_data.tex_coords, mesh_data.indices};
}

template<typename Loader>
[[nodiscard]]
auto loadFromFolder(const std::string &folder_path, Loader loader)
{
    std::map<std::string, decltype(loader(std::string()))> loaded;

    for (auto &entry: std::filesystem::directory_iterator(folder_path))
    {
        if (!entry.is_regular_file())
            continue;

        try
        {
            loaded.emplace(entry.path().stem(), loader(entry.path().native()));
        }
        catch (std::exception &e)
        {
            std::cerr << "Error when loading " << entry.path().native() << ": " << e.what() << "\n";
        }
    }

    return loaded;
}

template<typename Iter>
[[nodiscard]]
auto selectionGUI(const char *label, Iter current, Iter begin, Iter end)
{
    if (!ImGui::BeginCombo(label, current->first.c_str()))
        return current;

    auto selection = current;

    for (auto option = begin; selection == current && option != end; option++)
    {
        auto &[filename, mesh] = *option;

        ImGui::PushID(filename.c_str());
        if (ImGui::Selectable(filename.c_str()))
            selection = option;
        ImGui::PopID();
    }

    ImGui::EndCombo();

    return selection;
}

class ResultMesh
{
public:
    static const simple::VertexAttributeSequence attribute_sequence;
    static constexpr std::array attribute_locations{4, 5};

    ResultMesh(const MeshData &mesh_data, const placement::Result &result, uint layer)
            : m_mesh(mesh_data.positions, mesh_data.normals, mesh_data.tex_coords, mesh_data.indices),
              m_handle(m_mesh.addInstanceData(attribute_locations, attribute_sequence, 1,
                                              result.getClassElementCount(layer), result.getBuffer().gl_object,
                                              result.getClassBufferOffset(layer)))
    {
        m_mesh.setInstanceCount(result.getClassElementCount(layer));
    }

    void updateResult(const placement::Result &result, uint layer)
    {
        m_mesh.updateInstanceData(m_handle, result.getClassElementCount(layer), result.getBuffer().gl_object,
                                  result.getClassBufferOffset(layer));
        m_mesh.setInstanceCount(result.getClassElementCount(layer));
    }

    [[nodiscard]]
    const simple::InstancedMesh &getMesh() const
    { return m_mesh; }

private:
    simple::InstancedMesh m_mesh;
    simple::InstancedMesh::InstanceDataHandle m_handle;
};

const simple::VertexAttributeSequence ResultMesh::attribute_sequence = simple::VertexAttributeSequence()
        .addAttribute<glm::vec3>()
        .addAttribute<float>();

int main()
{
    GLFW::InitGuard glfw_init;

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    GLFW::Window window{"04 - Scene", initial_window_size};

    GL::enableDebugCallback();

    ImGuiContextWrapper imgui_context;
    ImGuiImplWrapper imgui_imp{window.get(), true};

    simple::Renderer renderer;

    constexpr glm::vec3 world_size = {1000, 1000, 100};

    Camera camera{window};
    {
        auto &camera_controller = camera.getController();
        camera_controller.setMaxPosition(world_size * glm::vec3(1., 1., .25));
        camera_controller.setMaxRadius(350.f);
        camera_controller.setRadius(250.f);
        camera_controller.setPosition(world_size / 2.f);
        camera_controller.setRadialSpeed(10.f);
        camera_controller.setAngle({glm::pi<float>() / 4., glm::pi<float>() / 4.});
    }

    simple::ShaderProgram simple_program{
            "void main() { gl_Position = proj_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0f); }",
            "void main() { frag_color = vec4(1.0f); }"
    };

    constexpr uint invalid_index = 0xffFFffFF;
    GL::gl.VertexAttribI1ui(5, invalid_index);

    PhongShader phong_shader;
    phong_shader.lightPosition() = {0, 0, 100};
    phong_shader.lightColor() = {1, 1, 1};
    phong_shader.ambientLightIntensity() = .1f;
    phong_shader.specularLightIntensity() = .5f;
    phong_shader.colorTextureUnit() = 0;
    phong_shader.specularHighlightFactor() = 0.1f;

    const auto [axes_mesh, axes_shader] = makeAxes();

    const glm::mat4 base_tree_transform{1};
    const glm::mat4 base_teapot_transform = glm::rotate(glm::scale(glm::mat4(1), glm::vec3(0.1)),
                                                        glm::pi<float>() / 2.f, {1, 0, 0});

    const auto grayscale_textures = loadFromFolder("assets/textures/grayscale", [](const std::string &path)
    { return simple::Texture2D(simple::ImageData::fromFile(path)); });

    if (grayscale_textures.empty())
    {
        std::cerr << "Found no textures in assets/textures/grayscale\n";
        return 1;
    }

    const simple::Texture2D color_texture{simple::ImageData::fromFile("assets/textures/color_palette.png"), false};

    GL::Texture::bindTextureUnit(0, color_texture.getGLObject());

    placement::PlacementPipeline pipeline;
    pipeline.setBaseTextureUnit(1);

    auto current_heightmap_iter = grayscale_textures.find("heightmap");
    if (current_heightmap_iter == grayscale_textures.end())
        current_heightmap_iter = grayscale_textures.begin();

    placement::WorldData world_data{{1000, 1000, 100}, current_heightmap_iter->second.getGLObject().getName()};

    // trees
    const auto tree_mesh_data = loadFromFolder("assets/meshes/trees", loadOBJ);

    placement::LayerData tree_layer_data{2.5f, {{grayscale_textures.at("heightmap").getGLObject().getName(), -.5},
                                                {grayscale_textures.at("white").getGLObject().getName(), 1.}}};

    std::optional<placement::FutureResult> tree_future_result = pipeline.computePlacement(world_data, tree_layer_data,
                                                                                          {0, 0}, {1000, 1000});

    std::vector<std::tuple<decltype(grayscale_textures)::const_iterator, decltype(tree_mesh_data)::const_iterator,
            std::optional<ResultMesh>>> tree_result_meshes;

    tree_result_meshes.emplace_back(grayscale_textures.find("heightmap"), tree_mesh_data.begin(),
                                    std::optional<ResultMesh>());
    tree_result_meshes.emplace_back(grayscale_textures.find("white"), tree_mesh_data.begin(),
                                    std::optional<ResultMesh>());

    placement::Result tree_result{{0, 0, GL::Buffer()}};

    auto prev_frame_start_time = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window.get()))
    {
        // check for pending results
        if (tree_future_result && tree_future_result->isReady())
        {
            tree_result = tree_future_result->readResult();

            for (uint layer_index = 0; layer_index < tree_result.getNumClasses(); layer_index++)
            {
                auto &[tex_iter, mesh_iter, result_mesh] = tree_result_meshes[layer_index];
                if (result_mesh)
                    result_mesh->updateResult(tree_result, layer_index);
                else
                    result_mesh = ResultMesh(mesh_iter->second, tree_result, layer_index);
            }

            tree_future_result.reset();
        }

        glfwPollEvents();

        const auto current_frame_start_time = std::chrono::steady_clock::now();
        const std::chrono::duration<float> frame_delta = current_frame_start_time - prev_frame_start_time;
        prev_frame_start_time = current_frame_start_time;

        imgui_imp.newFrame();
        ImGui::NewFrame();

        camera.getController().update(frame_delta.count());

        phong_shader.viewPosition() = camera.getController().getCameraPosition();

        if (ImGui::Begin("Settings"))
        {
            ImGui::Text("Frame time: %fs.\nFPS: %f", frame_delta.count(), 1. / frame_delta.count());

            ImGui::PushItemWidth(ImGui::GetWindowWidth() * .5f);
            if (ImGui::CollapsingHeader("Lighting"))
            {
                // light position
                {
                    glm::vec3 current_light_position = phong_shader.lightPosition();
                    if (ImGui::DragFloat3("Light position", glm::value_ptr(current_light_position)))
                        phong_shader.lightPosition() = current_light_position;
                }

                // light color
                {
                    glm::vec3 current_light_color = phong_shader.lightColor();
                    if (ImGui::ColorEdit3("Light color", glm::value_ptr(current_light_color)))
                        phong_shader.lightColor() = current_light_color;
                }

                // ambient light
                {
                    float ambient_light = phong_shader.ambientLightIntensity();
                    if (ImGui::DragFloat("Ambient light intensity", &ambient_light, 0.05, 0., 1.))
                        phong_shader.ambientLightIntensity() = ambient_light;
                }

                // specular light
                {
                    float specular_light = phong_shader.specularLightIntensity();
                    if (ImGui::DragFloat("Specular light intensity", &specular_light, 0.05, 0., 1.))
                        phong_shader.specularLightIntensity() = specular_light;
                }
            }

            if (ImGui::CollapsingHeader("Camera"))
            {
                {
                    float fov_y = camera.getFovY();
                    if (ImGui::DragFloat("FOV", &fov_y, .001, glm::pi<float>() / 8., glm::pi<float>()))
                        camera.setFovY(fov_y);
                }
                {
                    float near = camera.getNearPlane();
                    if (ImGui::DragFloat("Near plane", &near, .001))
                        camera.setNearPlane(near);
                }
                {
                    float far = camera.getFarPlane();
                    if (ImGui::DragFloat("Far plane", &far))
                        camera.setFarPlane(far);
                }

                ImGui::Spacing();

                auto &controller = camera.getController();
                ImGui::Text("Current distance: %f", controller.getRadius());
                {
                    float max_radius = controller.getMaxRadius();
                    if (ImGui::DragFloat("Max. distance", &max_radius))
                        controller.setMaxRadius(max_radius);
                }
                {
                    float radial_speed = controller.getRadialSpeed();
                    if (ImGui::DragFloat("Scroll speed", &radial_speed, 0.01))
                        controller.setRadialSpeed(radial_speed);
                }
                {
                    float speed = controller.getSpeed();
                    if (ImGui::DragFloat("Speed", &speed, 0.1))
                        controller.setSpeed(speed);
                }
            }

            if (ImGui::CollapsingHeader("Placement"))
            {
                ImGui::BeginChild("WorldData");

                ImGui::Text("World Data");
                const auto selected_heightmap_iter = selectionGUI("Heightmap", current_heightmap_iter,
                                                                  grayscale_textures.begin(), grayscale_textures.end());

                if (selected_heightmap_iter != current_heightmap_iter)
                {
                    current_heightmap_iter = selected_heightmap_iter;
                    world_data.heightmap = current_heightmap_iter->second.getGLObject().getName();
                }

                ImGui::Separator();
                ImGui::Text("Layer Data");

                ImGui::DragFloat("Footprint", &tree_layer_data.footprint, 0.01f, 0.01f, FLT_MAX);

                if (ImGui::BeginListBox("Tree Layers"))
                {
                    for (int i = 0; i < tree_layer_data.densitymaps.size(); i++)
                    {
                        ImGui::PushID(i);
                        ImGui::Text("[%d]", i);
                        ImGui::SameLine();
                        if (ImGui::CollapsingHeader("DensityMap"))
                        {
                            auto &[tex_iter, mesh_iter, result_mesh] = tree_result_meshes[i];
                            const auto selected_mesh = selectionGUI("Mesh", mesh_iter, tree_mesh_data.begin(),
                                                                    tree_mesh_data.end());
                            if (selected_mesh != mesh_iter)
                            {
                                mesh_iter = selected_mesh;
                                result_mesh = ResultMesh(mesh_iter->second, tree_result, i);
                            }

                            const auto selected_density_map_iter = selectionGUI("Texture", tex_iter,
                                                                                grayscale_textures.begin(),
                                                                                grayscale_textures.end());
                            if (selected_density_map_iter != tex_iter)
                            {
                                tex_iter = selected_heightmap_iter;
                                tree_layer_data.densitymaps[i].texture = tex_iter->second.getGLObject().getName();
                            }
                            auto& density_map = tree_layer_data.densitymaps[i];
                            ImGui::DragFloat("Scale", &density_map.scale, 0.001);
                            ImGui::DragFloat("Offset", &density_map.offset, 0.001);
                            ImGui::DragFloat2("Min./Max. value", &density_map.min_value, 0.001);
                        }
                        ImGui::PopID();
                    }
                }
                ImGui::EndListBox();

                if (ImGui::Button("Compute Placement"))
                    tree_future_result = pipeline.computePlacement(world_data, tree_layer_data, {0, 0}, world_size);

                ImGui::EndChild();
            }

            ImGui::PopItemWidth();
        }
        ImGui::End();

        // Render

        renderer.draw(axes_mesh, axes_shader, glm::scale(glm::mat4(1),
                                                         glm::vec3(glm::max(1.f, camera.getController().getRadius() /
                                                                                 2.f))));

        for (auto &[_0, _1, mesh]: tree_result_meshes)
            if (mesh)
                renderer.draw(mesh->getMesh(), phong_shader.getRendererProgram(), base_tree_transform);

        renderer.finishFrame(camera.getRendererCamera());

        ImGui::Render();
        imgui_imp.renderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window.get());
    }
}