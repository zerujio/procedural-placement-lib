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

template<typename Iter, typename SelectionCallback>
auto selectionGUI(const char *label, Iter current, Iter begin, Iter end, SelectionCallback selection_callback)
{
    if (!ImGui::BeginCombo(label, current->first.c_str()))
        return current;

    auto selection = current;

    for (auto option = begin; selection == current && option != end; option++)
    {
        auto &[filename, mesh] = *option;

        ImGui::PushID(filename.c_str());
        if (ImGui::Selectable(filename.c_str()))
        {
            selection = option;
            selection_callback(selection);
        }
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

class PlacementGroup
{
public:
    using TextureIter = std::map<std::string, simple::Texture2D>::const_iterator;
    using MeshDataIter = std::map<std::string, MeshData>::const_iterator;

    [[nodiscard]]
    float getFootprint() const { return m_layer_data.footprint; }
    void setFootprint(float diameter) { m_layer_data.footprint = diameter; }

    void computePlacement(placement::PlacementPipeline& pipeline, placement::WorldData& world_data,
                          glm::vec2 lower_bound, glm::vec2 upper_bound)
    {
        m_future_result = pipeline.computePlacement(world_data, m_layer_data, lower_bound, upper_bound);
    }

    void checkResult()
    {
        if (!m_future_result || !m_future_result->isReady())
            return;

        m_result = m_future_result->readResult();
        m_future_result.reset();

        for (uint i = 0; i < m_result->getNumClasses(); i++)
        {
            auto& mesh_opt = m_meshes[i];
            if (mesh_opt)
                mesh_opt->updateResult(m_result.value(), i);
            else
                mesh_opt = ResultMesh(m_iters[i].second->second, m_result.value(), i);
        }
    }

    [[nodiscard]]
    uint getNumLayers() const { return m_layer_data.densitymaps.size(); }

    void addLayer(TextureIter texture, MeshDataIter mesh)
    {
        m_layer_data.densitymaps.emplace_back(placement::DensityMap{texture->second.getGLObject().getName()});
        m_meshes.emplace_back();
        m_iters.emplace_back(texture, mesh);
    }

    void removeLayer()
    {
        m_layer_data.densitymaps.pop_back();
        m_meshes.pop_back();
        m_iters.pop_back();
    }

    [[nodiscard]] TextureIter getLayerTexture(uint layer_index) const { return m_iters.at(layer_index).first; }
    void setLayerTexture(uint layer_index, TextureIter texture_iter)
    {
        m_layer_data.densitymaps.at(layer_index).texture = texture_iter->second.getGLObject().getName();
        m_iters[layer_index].first = texture_iter;
    }

    [[nodiscard]] MeshDataIter getLayerMesh(uint layer_index) const { return m_iters.at(layer_index).second; }
    void setLayerMesh(uint layer_index, MeshDataIter mesh_data_iter)
    {
        auto& mesh = m_meshes.at(layer_index);
        if (mesh && m_result)
            mesh = ResultMesh(mesh_data_iter->second, m_result.value(), layer_index);

        m_iters[layer_index].second = mesh_data_iter;
    }
    [[nodiscard]] const auto& getMeshes() const { return m_meshes; }

    struct LayerParams
    {
        float scale;
        float offset;
        float min_value;
        float max_value;
    };

    [[nodiscard]] LayerParams getLayerParams(uint layer_index)
    {
        const auto &dm = m_layer_data.densitymaps.at(layer_index);
        return {dm.scale, dm.offset, dm.min_value, dm.max_value};
    }

    void setLayerParams(uint layer_index, LayerParams layer_params)
    {
        auto &dm = m_layer_data.densitymaps.at(layer_index);
        dm.scale = layer_params.scale;
        dm.offset = layer_params.offset;
        dm.min_value = layer_params.min_value;
        dm.max_value = layer_params.max_value;
    }

private:
    placement::LayerData m_layer_data;
    std::optional<placement::Result> m_result;
    std::optional<placement::FutureResult> m_future_result;
    std::vector<std::optional<ResultMesh>> m_meshes;
    std::vector<std::pair<TextureIter, MeshDataIter>> m_iters;
};

void placementGroupGUI(PlacementGroup &placement_group, const std::map<std::string, simple::Texture2D> &textures,
                       const std::map<std::string, MeshData> &meshes)
{
    float footprint = placement_group.getFootprint();
    if (ImGui::DragFloat("Footprint", &footprint, 0.01f, 0.01f, FLT_MAX))
        placement_group.setFootprint(footprint);

    if (ImGui::BeginListBox("Layers", {0, ImGui::GetTextLineHeightWithSpacing() * 10.f}))
    {
        for (int i = 0; i < placement_group.getNumLayers(); i++)
        {
            if (ImGui::GetContentRegionAvail().y == 0)
                break;

            ImGui::PushID(i);
            ImGui::Text("[%d]", i);
            ImGui::SameLine();
            if (ImGui::CollapsingHeader("DensityMap"))
            {
                selectionGUI("Mesh", placement_group.getLayerMesh(i), meshes.begin(), meshes.end(),
                            [i, &placement_group](decltype(meshes.begin()) iter)
                            { placement_group.setLayerMesh(i, iter); });
                selectionGUI("Texture", placement_group.getLayerTexture(i), textures.begin(), textures.end(),
                             [i, &placement_group](decltype(textures.begin()) iter)
                             { placement_group.setLayerTexture(i, iter); });

                auto params = placement_group.getLayerParams(i);
                bool params_changed = false;
                params_changed = ImGui::DragFloat("Scale", &params.scale, 0.001) || params_changed;
                params_changed = ImGui::DragFloat("Offset", &params.offset, 0.001) || params_changed;
                params_changed = ImGui::DragFloat2("Min./Max. value", &params.min_value, 0.001) || params_changed;
                if (params_changed)
                    placement_group.setLayerParams(i, params);
            }
            ImGui::PopID();
        }
        ImGui::EndListBox();
    }
}

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
    phong_shader.ambientLightIntensity() = .5f;
    phong_shader.specularLightIntensity() = .2f;
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

    PlacementGroup tree_placement_group;
    tree_placement_group.setFootprint(2.5);

    tree_placement_group.addLayer(grayscale_textures.find("heightmap"), tree_mesh_data.begin());
    tree_placement_group.setLayerParams(0, {-1., 0., -1., 1.});

    {
        const float num_layers = tree_mesh_data.size();
        uint current_layer = 1;
        const auto linear_gradient_iter = grayscale_textures.find("linear_gradient");

        {
            PlacementGroup::LayerParams blue_pine_params{1 / num_layers, 0, 0, 1};
            for (int i = 1; i <= 5; i++)
            {
                tree_placement_group.addLayer(linear_gradient_iter,
                                              tree_mesh_data.find("BluePineTree" + std::to_string(i)));
                tree_placement_group.setLayerParams(current_layer++, blue_pine_params);
            }
        }

        {
            PlacementGroup::LayerParams params{- 1/num_layers, 1/num_layers, 0, 1};
            for (int i = 1; i <= 3; i++)
            {
                tree_placement_group.addLayer(linear_gradient_iter,
                                              tree_mesh_data.find("PineTree" + std::to_string(i)));
                tree_placement_group.setLayerParams(current_layer++, params);
            }
        }
    }

    tree_placement_group.computePlacement(pipeline, world_data, {0, 0}, world_size);

    // stones
    const auto stone_mesh_data = loadFromFolder("assets/meshes/stones", loadOBJ);

    PlacementGroup stone_placement_group;
    stone_placement_group.setFootprint(1.f);

    {
        const auto heightmap_iter = grayscale_textures.find("heightmap");
        const PlacementGroup::LayerParams layer_params {1.f / stone_mesh_data.size(), 0, 0, 1};

        uint current_layer = 0;
        for (auto mesh_it = stone_mesh_data.begin(); mesh_it != stone_mesh_data.end(); mesh_it++)
        {
            stone_placement_group.addLayer(heightmap_iter, mesh_it);
            stone_placement_group.setLayerParams(current_layer++, layer_params);
        }
    }

    stone_placement_group.computePlacement(pipeline, world_data, {0, 0}, world_size);

    auto prev_frame_start_time = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window.get()))
    {
        // check for pending results
        tree_placement_group.checkResult();
        stone_placement_group.checkResult();

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
                ImGui::BeginChild("Placement");

                ImGui::Text("World Data\nScale: %fx x %fy x %fz",
                            world_data.scale.x, world_data.scale.y, world_data.scale.z);

                selectionGUI("Heightmap", current_heightmap_iter, grayscale_textures.begin(), grayscale_textures.end(),
                             [&](decltype(current_heightmap_iter) new_iter)
                             {
                                world_data.heightmap = new_iter->second.getGLObject().getName();
                                current_heightmap_iter = new_iter;
                             });

                ImGui::Spacing();
                ImGui::Separator();

                ImGui::Text("Trees");
                ImGui::PushID("Trees");
                placementGroupGUI(tree_placement_group, grayscale_textures, tree_mesh_data);
                ImGui::PopID();

                ImGui::Spacing();
                ImGui::Separator();

                ImGui::Text("Rocks");
                ImGui::PushID("Rocks");
                placementGroupGUI(stone_placement_group, grayscale_textures, stone_mesh_data);
                ImGui::PopID();

                ImGui::Separator();

                if (ImGui::Button("Compute Placement"))
                {
                    tree_placement_group.computePlacement(pipeline, world_data, {0, 0}, world_size);
                    stone_placement_group.computePlacement(pipeline, world_data, {0, 0}, world_size);
                }

                ImGui::EndChild();
            }

            ImGui::PopItemWidth();
        }
        ImGui::End();

        // Render

        renderer.draw(axes_mesh, axes_shader,
                      glm::scale(glm::mat4(1), glm::vec3(glm::max(1.f, camera.getController().getRadius() / 2.f))));

        for (auto &mesh_opt : tree_placement_group.getMeshes())
            if (mesh_opt)
                renderer.draw(mesh_opt->getMesh(), phong_shader.getRendererProgram(), base_tree_transform);

        for (auto &mesh_opt : stone_placement_group.getMeshes())
            if (mesh_opt)
                renderer.draw(mesh_opt->getMesh(), phong_shader.getRendererProgram(), glm::mat4(1.f));

        renderer.finishFrame(camera.getRendererCamera());

        ImGui::Render();
        imgui_imp.renderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window.get());
    }
}