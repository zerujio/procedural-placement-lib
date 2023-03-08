#include "placement/placement.hpp"

#include "example-common.hpp"
#include "simple-renderer/renderer.hpp"
#include "simple-renderer/instanced_mesh.hpp"
#include "simple-renderer/texture_2d.hpp"
#include "simple-renderer/image_data.hpp"
#include "glutils/debug.hpp"
#include "imgui.h"
#include "external/json.hpp"

#include <chrono>
#include <iostream>
#include <filesystem>
#include <fstream>

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
    float getFootprint() const
    { return m_layer_data.footprint; }

    void setFootprint(float diameter)
    { m_layer_data.footprint = diameter; }

    void computePlacement(placement::PlacementPipeline &pipeline, placement::WorldData &world_data,
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
            auto &mesh_opt = m_meshes[i];
            if (mesh_opt)
                mesh_opt->updateResult(m_result.value(), i);
            else
                mesh_opt = ResultMesh(m_iters[i].second->second, m_result.value(), i);
        }
    }

    [[nodiscard]]
    uint getNumLayers() const
    { return m_layer_data.densitymaps.size(); }

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

    [[nodiscard]] TextureIter getLayerTexture(uint layer_index) const
    { return m_iters.at(layer_index).first; }

    void setLayerTexture(uint layer_index, TextureIter texture_iter)
    {
        m_layer_data.densitymaps.at(layer_index).texture = texture_iter->second.getGLObject().getName();
        m_iters[layer_index].first = texture_iter;
    }

    [[nodiscard]] MeshDataIter getLayerMesh(uint layer_index) const
    { return m_iters.at(layer_index).second; }

    void setLayerMesh(uint layer_index, MeshDataIter mesh_data_iter)
    {
        auto &mesh = m_meshes.at(layer_index);
        if (mesh && m_result)
            mesh = ResultMesh(mesh_data_iter->second, m_result.value(), layer_index);

        m_iters[layer_index].second = mesh_data_iter;
    }

    [[nodiscard]] const auto &getMeshes() const
    { return m_meshes; }

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

    if (ImGui::Button("Add layer"))
    {
        const auto n_layers = placement_group.getNumLayers();
        if (n_layers > 0)
        {
            placement_group.addLayer(placement_group.getLayerTexture(n_layers - 1),
                                     placement_group.getLayerMesh(n_layers - 1));
            placement_group.setLayerParams(n_layers, placement_group.getLayerParams(n_layers - 1));
        }
        else
            placement_group.addLayer(textures.begin(), meshes.begin());
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove Layer"))
        placement_group.removeLayer();
}

class HeightmapComputeShader
{
public:
    static constexpr glm::uvec2 work_group_size{8, 8};

    void operator()(glm::uvec2 num_work_groups, GLint heightmap_tex_unit, GLuint position_binding,
                    GLuint normals_binding, GLuint tex_coord_binding, GLuint indices_binding)
    {
        m_program.setUniform(m_heightmap, heightmap_tex_unit);
        m_program.setShaderStorageBlockBindingIndex(m_positions, position_binding);
        m_program.setShaderStorageBlockBindingIndex(m_normals, normals_binding);
        m_program.setShaderStorageBlockBindingIndex(m_tex_coords, tex_coord_binding);
        m_program.setShaderStorageBlockBindingIndex(m_indices, indices_binding);
        m_program.dispatch({num_work_groups, 1});
    }

private:
    placement::ComputeShaderProgram m_program{loadComputeShaderProgram("assets/shaders/heightmap.comp")};
    using CS = placement::ComputeShaderProgram;
    CS::ShaderStorageBlock m_positions{m_program.getShaderStorageBlockIndex("Positions")};
    CS::ShaderStorageBlock m_normals{m_program.getShaderStorageBlockIndex("Normals")};
    CS::ShaderStorageBlock m_tex_coords{m_program.getShaderStorageBlockIndex("TexCoords")};
    CS::ShaderStorageBlock m_indices{m_program.getShaderStorageBlockIndex("Indices")};
    CS::TypedUniform<int> m_heightmap{m_program.getUniformLocation("u_heightmap")};
};

class TerrainMesh : public simple::Drawable
{
public:
    TerrainMesh()
    {
        m_vertex_attributes.bindIndexBuffer(m_vertex_buffer);
    }

    void generate(glm::uvec2 num_work_groups, GLuint heightmap_tex_unit)
    {
        const glm::uvec2 grid_size = num_work_groups * HeightmapComputeShader::work_group_size;
        const GLuint num_vertices = grid_size.x * grid_size.y;

        m_num_indices = (grid_size.x - 1) * (grid_size.y - 1) * 6;

        const std::size_t required_size = sizeof(float) * (4 + 4 + 2) * num_vertices + sizeof(uint) * m_num_indices;

        if (required_size > m_vertex_buffer.getBufferSize())
        {
            m_vertex_buffer = simple::VertexBuffer(required_size);
            m_vertex_attributes.bindIndexBuffer(m_vertex_buffer);
        }

        const auto buffer_handle = m_vertex_buffer.getBufferHandle();
        const auto makeInitializer = [=](GLuint binding_index)
        {
            return [=](simple::WBufferRef ref)
            {
                buffer_handle.bindRange(GL::Buffer::IndexedTarget::shader_storage, binding_index,
                                        ref.getOffset(), ref.getSize());
            };
        };

        const auto bindProperty = [this, &makeInitializer](int location,
                                                           const simple::VertexAttributeSequence &attribs,
                                                           uint num_vertices)
        {
            if (location < m_vertex_buffer.getSectionCount())
                m_vertex_buffer.updateAttributeData(location, makeInitializer(location));
            else
            {
                const auto &desc = m_vertex_buffer.addAttributeData(makeInitializer(location), num_vertices, attribs);
                m_vertex_attributes.bindAttributes(m_vertex_buffer, desc, std::array{location});
            }

        };

        {
            const auto vec3_attrib = simple::VertexAttributeSequence()
                    .addAttribute<glm::vec3>()
                    .addPadding(sizeof(float));
            // positions
            bindProperty(0, vec3_attrib, num_vertices);

            //normals
            bindProperty(1, vec3_attrib, num_vertices);
        }

        // tex coords
        bindProperty(2, simple::VertexAttributeSequence().addAttribute<glm::vec2>(), num_vertices);

        // indices
        bindProperty(3, simple::VertexAttributeSequence().addAttribute<uint>(), m_num_indices);
        m_index_buffer_offset = m_vertex_buffer.getSectionDescriptor(3).buffer_offset;

        m_compute_shader(num_work_groups, heightmap_tex_unit, 0, 1, 2, 3);
        GL::gl.MemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT);
    }

    void collectDrawCommands(const CommandCollector &collector) const override
    {
        m_vertex_attributes.emplaceDrawCommand<simple::DrawElementsCommand>(collector,
                                                                            {draw_mode,
                                                                             m_num_indices,
                                                                             simple::IndexType::unsigned_int,
                                                                             m_index_buffer_offset});
    }

    simple::DrawMode draw_mode = simple::DrawMode::triangles;

private:
    HeightmapComputeShader m_compute_shader;
    simple::VertexAttributeSpecification m_vertex_attributes;
    simple::VertexBuffer m_vertex_buffer{4096};
    std::uintptr_t m_index_buffer_offset{0};
    std::uint32_t m_num_indices{0};
};

class TerrainPhongShader
{
    using SP = simple::ShaderProgram;

public:
    [[nodiscard]]
    const simple::ShaderProgram &getRendererProgram() const
    { return m_program; };

    void setViewPosition(glm::vec3 position) const
    { setUniform(m_view_position, position); }

    void setLightColor(glm::vec3 color) const
    { setUniform(m_light_color, color); }

    void setLightPosition(glm::vec3 position) const
    { setUniform(m_light_position, position); }

    void setAmbientLightIntensity(float value) const
    { setUniform(m_ambient_light_intensity, value); }

    void setSpecularLightIntensity(float value) const
    { setUniform(m_specular_light_intensity, value); }

    void setSpecularHighlightFactor(float value) const
    { setUniform(m_specular_highlight_factor, value); }

    [[nodiscard]] auto heightmap()
    { return m_program.makeAccessor(m_heightmap); }

    [[nodiscard]] auto colorTexUnit()
    { return m_program.makeAccessor(m_color_palette); }

    [[nodiscard]] auto lowColorIndex()
    { return m_program.makeAccessor(m_low_color); }

    [[nodiscard]] auto highColorIndex()
    { return m_program.makeAccessor(m_high_color); }

private:
    template<typename U, typename T>
    void setUniform(const U &uniform, const T &value) const
    { m_program.setUniform(uniform, value); }

    simple::ShaderProgram m_program{
            loadShaderProgram("assets/shaders/phong.vert", "assets/shaders/phong_terrain.frag")};
    SP::TypedUniform <glm::vec3> m_view_position{m_program.getUniformLocation("u_view_position")};
    SP::TypedUniform <glm::vec3> m_light_color{m_program.getUniformLocation("u_light_color")};
    SP::TypedUniform <glm::vec3> m_light_position{m_program.getUniformLocation("u_light_position")};
    SP::TypedUniform<float> m_ambient_light_intensity{m_program.getUniformLocation("u_ambient_light_intensity")};
    SP::TypedUniform<float> m_specular_light_intensity{m_program.getUniformLocation("u_specular_light_intensity")};
    SP::TypedUniform<float> m_specular_highlight_factor{m_program.getUniformLocation("u_specular_highlight_factor")};
    SP::CachedUniform<int> m_heightmap{m_program.getUniformLocation("u_heightmap")};
    SP::CachedUniform<int> m_color_palette{m_program.getUniformLocation("u_color_palette")};
    SP::CachedUniform <uint> m_high_color{m_program.getUniformLocation("u_color_palette_high")};
    SP::CachedUniform <uint> m_low_color{m_program.getUniformLocation("u_color_palette_low")};
};

nlohmann::json loadHeightmapConfig(const std::filesystem::path &path)
{
    std::ifstream ifstream{path};

    if (!ifstream.is_open())
        throw std::runtime_error("couldn't open heightmap config file: " + path.native());

    return nlohmann::json::parse(ifstream);
}

int main()
{
    GLFW::InitGuard glfw_init;

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    GLFW::Window window{"04 - Scene", initial_window_size};

    GL::enableDebugCallback();

    const std::filesystem::path assets_folder{"assets/"};

    const auto heightmap_config = loadHeightmapConfig(assets_folder / "heightmap.json");

    ImGuiContextWrapper imgui_context;
    ImGuiImplWrapper imgui_imp{window.get(), true};

    simple::Renderer renderer;

    Camera camera{window};
    {
        auto &camera_controller = camera.getController();
        camera_controller.setMaxRadius(350.f);
        camera_controller.setRadius(250.f);
        camera_controller.setRadialSpeed(10.f);
        camera_controller.setAngle({glm::pi<float>() / 4., glm::pi<float>() / 4.});
    }

    simple::ShaderProgram simple_program{
            "void main() { gl_Position = proj_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0f); }",
            "void main() { frag_color = vec4(1.0f); }"
    };

    constexpr uint invalid_index = 0xffFFffFF;
    GL::gl.VertexAttribI1ui(5, invalid_index);

    constexpr GLuint color_texture_unit = 0;
    constexpr GLuint heightmap_texture_unit = 1;

    PhongShader phong_shader;
    phong_shader.lightPosition() = {0, 0, 10000};
    phong_shader.lightColor() = {1., 1., 1.};
    phong_shader.ambientLightIntensity() = .4f;
    phong_shader.specularLightIntensity() = .05f;
    phong_shader.specularHighlightFactor() = 0.1f;
    phong_shader.colorTextureUnit() = color_texture_unit;

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

    GL::Texture::bindTextureUnit(color_texture_unit, color_texture.getGLObject());

    placement::PlacementPipeline pipeline;
    pipeline.setBaseTextureUnit(glm::max(heightmap_texture_unit, color_texture_unit) + 1);

    auto current_heightmap_iter = grayscale_textures.find(heightmap_config["file"].get<std::filesystem::path>().stem());

    if (current_heightmap_iter == grayscale_textures.end())
    {
        std::cerr << "heightmap file (assets/" << heightmap_config["file"].get<std::string>() << ") is missing\n";
        return 1;
    }

    placement::WorldData world_data{/*scale=*/{1, 1, 1},
            /*heightmap=*/current_heightmap_iter->second.getGLObject().getName()};

    world_data.scale.z = heightmap_config["max elevation"].get<float>() - heightmap_config["min elevation"].get<float>();
    world_data.scale.x = world_data.scale.z / heightmap_config["z/x scale"].get<float>();
    world_data.scale.y = world_data.scale.x *
            float(current_heightmap_iter->second.getSize().y) / float(current_heightmap_iter->second.getSize().x);

    camera.getController().setMaxPosition(world_data.scale);
    camera.getController().setMaxRadius(world_data.scale.z * 0.5f);
    camera.getController().setPosition(world_data.scale * glm::vec3 {.5, .5, .125});

    // trees
    const auto tree_mesh_data = loadFromFolder("assets/meshes/trees", loadOBJ);

    PlacementGroup tree_placement_group;
    tree_placement_group.setFootprint(1.75);

    tree_placement_group.addLayer(current_heightmap_iter, tree_mesh_data.begin());
    tree_placement_group.setLayerParams(0, {-.1, 0., -1., 1.});

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
            PlacementGroup::LayerParams params{-1 / num_layers, 1 / num_layers, 0, 1};
            for (int i = 1; i <= 3; i++)
            {
                tree_placement_group.addLayer(linear_gradient_iter,
                                              tree_mesh_data.find("PineTree" + std::to_string(i)));
                tree_placement_group.setLayerParams(current_layer++, params);
            }
        }
    }

    // stones
    const auto stone_mesh_data = loadFromFolder("assets/meshes/stones", loadOBJ);

    PlacementGroup stone_placement_group;
    stone_placement_group.setFootprint(2.25f);

    {
        const PlacementGroup::LayerParams layer_params{1.f / stone_mesh_data.size(), 0, 0, 1};

        uint current_layer = 0;
        for (auto mesh_it = stone_mesh_data.begin(); mesh_it != stone_mesh_data.end(); mesh_it++)
        {
            stone_placement_group.addLayer(current_heightmap_iter, mesh_it);
            stone_placement_group.setLayerParams(current_layer++, layer_params);
        }
    }

    struct
    {
        [[nodiscard]] constexpr std::pair<glm::vec2, glm::vec2> getPlacementBounds() const
        {
            const glm::vec2 center {current_cell};
            const glm::vec2 delta {cell_vicinity};
            return {(center - delta) * cell_size, (center + delta) * cell_size};
        }

        bool updatePosition(glm::vec2 current_position)
        {
            const glm::uvec2 cell {current_position / cell_size};
            if (cell == current_cell)
                return false;
            current_cell = cell;
            return true;
        }

        float cell_size{100.f};
        glm::uvec2 current_cell{0, 0};
        glm::uvec2 cell_vicinity{3, 3};
    } placement_grid;

    glm::mat4 placement_transform {1.f};

    const auto dispatchPlacementCompute = [&]()
    {
        const auto [lower_bound, upper_bound] = placement_grid.getPlacementBounds();

        const glm::vec2 xy_scale = world_data.scale;
        tree_placement_group.computePlacement(pipeline, world_data, glm::max(lower_bound, {0, 0}),
                                              glm::min(upper_bound, xy_scale));
        stone_placement_group.computePlacement(pipeline, world_data, glm::max(lower_bound, {0, 0}),
                                               glm::min(upper_bound, xy_scale));

        placement_transform = glm::translate(glm::mat4(1), {lower_bound, 0});
    };

    dispatchPlacementCompute();

    // terrain
    TerrainMesh terrain_mesh;
    GL::Texture::bindTextureUnit(heightmap_texture_unit, current_heightmap_iter->second.getGLObject());
    glm::uvec2 terrain_resolution = current_heightmap_iter->second.getSize() / 8u;
    const auto dispatchTerrainCompute = [&]()
    { terrain_mesh.generate(terrain_resolution, heightmap_texture_unit); };

    dispatchTerrainCompute();

    const auto terrain_transform{glm::scale(glm::mat4(1), world_data.scale)};

    TerrainPhongShader terrain_shader;
    terrain_shader.colorTexUnit() = color_texture_unit;
    terrain_shader.heightmap() = heightmap_texture_unit;
    terrain_shader.setLightPosition(phong_shader.lightPosition());
    terrain_shader.setLightColor(phong_shader.lightColor());
    terrain_shader.setAmbientLightIntensity(phong_shader.ambientLightIntensity());
    terrain_shader.setSpecularLightIntensity(phong_shader.specularLightIntensity());
    terrain_shader.setSpecularHighlightFactor(phong_shader.specularHighlightFactor());
    terrain_shader.lowColorIndex() = 12;
    terrain_shader.highColorIndex() = 16;

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

        auto camera_position = camera.getController().getPosition();

        if (placement_grid.updatePosition(camera_position))
            dispatchPlacementCompute();

        terrain_shader.setViewPosition(phong_shader.viewPosition() = camera.getController().getCameraPosition());

        if (ImGui::Begin("Settings"))
        {
            ImGui::Text("Frame time: %fs.\nFPS: %f", frame_delta.count(), 1. / frame_delta.count());

            ImGui::Separator();

            ImGui::PushItemWidth(ImGui::GetWindowWidth() * .5f);

            if (ImGui::DragFloat3("Position", glm::value_ptr(camera_position)))
                camera.getController().setPosition(camera_position);

            if (ImGui::CollapsingHeader("Lighting"))
            {
                // light position
                {
                    glm::vec3 current_light_position = phong_shader.lightPosition();
                    if (ImGui::DragFloat3("Light position", glm::value_ptr(current_light_position)))
                        terrain_shader.setLightPosition(phong_shader.lightPosition() = current_light_position);
                }

                // light color
                {
                    glm::vec3 current_light_color = phong_shader.lightColor();
                    if (ImGui::ColorEdit3("Light color", glm::value_ptr(current_light_color)))
                        terrain_shader.setLightColor(phong_shader.lightColor() = current_light_color);
                }

                // ambient light
                {
                    float ambient_light = phong_shader.ambientLightIntensity();
                    if (ImGui::DragFloat("Ambient light intensity", &ambient_light, 0.05, 0., 1.))
                        terrain_shader.setAmbientLightIntensity(phong_shader.ambientLightIntensity() = ambient_light);
                }

                // specular light
                {
                    float specular_light = phong_shader.specularLightIntensity();
                    if (ImGui::DragFloat("Specular light intensity", &specular_light, 0.05, 0., 1.))
                        terrain_shader.setSpecularLightIntensity(
                                phong_shader.specularLightIntensity() = specular_light);
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

                ImGui::Text("Current grid cell: x=%d, y=%d",
                            placement_grid.current_cell.x,
                            placement_grid.current_cell.y);

                ImGui::DragFloat("Grid cell size", &placement_grid.cell_size, 10, 1, FLT_MAX, "%.1f");

                ImGui::DragInt2("Placement area", reinterpret_cast<int*>(glm::value_ptr(placement_grid.cell_vicinity)),
                                1, 0, INT_MAX);

                ImGui::Separator();

                ImGui::Text("World Data\nScale: %fx x %fy x %fz",
                            world_data.scale.x, world_data.scale.y, world_data.scale.z);

                selectionGUI("Heightmap", current_heightmap_iter, grayscale_textures.begin(), grayscale_textures.end(),
                             [&](decltype(current_heightmap_iter) new_iter)
                             { current_heightmap_iter = new_iter; });

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

                    const GL::TextureHandle heightmap_texture = current_heightmap_iter->second.getGLObject();

                    if (world_data.heightmap != heightmap_texture.getName())
                    {
                        GL::Texture::bindTextureUnit(heightmap_texture_unit, heightmap_texture);
                        dispatchTerrainCompute();
                    }

                    world_data.heightmap = heightmap_texture.getName();
                    dispatchPlacementCompute();
                }

                ImGui::EndChild();
            }

            if (ImGui::CollapsingHeader("Terrain"))
            {
                if (ImGui::DragInt2("Terrain mesh resolution (x8)", reinterpret_cast<int*>(glm::value_ptr(terrain_resolution)),
                                    1, 1, INT_MAX))
                    dispatchTerrainCompute();

                // terrain color
                {
                    ImGui::Text("Terrain color palette indices");
                    glm::ivec2 terrain_color_indices{terrain_shader.lowColorIndex(), terrain_shader.highColorIndex()};
                    if (ImGui::DragInt2("Low/High", glm::value_ptr(terrain_color_indices), .5, 0, 48))
                    {
                        terrain_shader.lowColorIndex() = terrain_color_indices.x;
                        terrain_shader.highColorIndex() = terrain_color_indices.y;
                    }
                }
            }

            ImGui::PopItemWidth();
        }
        ImGui::End();

        // Render

        renderer.draw(axes_mesh, axes_shader,
                      glm::scale(glm::mat4(1), glm::vec3(glm::max(1.f, camera.getController().getRadius() / 2.f))));

        for (auto &mesh_opt: tree_placement_group.getMeshes())
            if (mesh_opt)
                renderer.draw(mesh_opt->getMesh(), phong_shader.getRendererProgram(), base_tree_transform);

        for (auto &mesh_opt: stone_placement_group.getMeshes())
            if (mesh_opt)
                renderer.draw(mesh_opt->getMesh(), phong_shader.getRendererProgram(), glm::mat4(1.f));

        terrain_mesh.draw_mode = glfwGetKey(window.get(), GLFW_KEY_SPACE) == GLFW_PRESS ? simple::DrawMode::lines : simple::DrawMode::triangles;
        renderer.draw(terrain_mesh, terrain_shader.getRendererProgram(), terrain_transform);

        renderer.finishFrame(camera.getRendererCamera());

        ImGui::Render();
        imgui_imp.renderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window.get());
    }
}