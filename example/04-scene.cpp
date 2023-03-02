#include "placement/placement.hpp"

#include "example-common.hpp"
#include "simple-renderer/renderer.hpp"
#include "glutils/debug.hpp"
#include "imgui.h"

#include <chrono>
#include <iostream>

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
    [[nodiscard]] const simple::ShaderProgram& getRendererProgram() const { return m_program; }

    [[nodiscard]] auto lightPosition() { return m_program.makeAccessor(m_light_position); }
    [[nodiscard]] auto lightColor() { return m_program.makeAccessor(m_light_color); }
    [[nodiscard]] auto viewPosition() { return m_program.makeAccessor(m_view_position); }
    [[nodiscard]] auto ambientLightIntensity() {return m_program.makeAccessor(m_ambient_light_intensity); }
    [[nodiscard]] auto specularLightIntensity() {return m_program.makeAccessor(m_specular_light_intensity); }

private:
    using SP = simple::ShaderProgram;

    simple::ShaderProgram m_program = loadShaderProgram("assets/shaders/phong.vert", "assets/shaders/phong.frag");

    SP::CachedUniform<glm::vec3> m_light_position {m_program.getUniformLocation("u_light_position")};
    SP::CachedUniform<glm::vec3> m_light_color {m_program.getUniformLocation("u_light_color")};
    SP::CachedUniform<glm::vec3> m_view_position {m_program.getUniformLocation("u_view_position")};
    SP::CachedUniform<float> m_ambient_light_intensity {m_program.getUniformLocation("u_ambient_light_intensity")};
    SP::CachedUniform<float> m_specular_light_intensity {m_program.getUniformLocation("u_specular_light_intensity")};
};

int main()
{
    GLFW::InitGuard glfw_init;

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
    GLFW::Window window{"04 - Scene", initial_window_size};

    GL::enableDebugCallback();

    ImGuiContextWrapper imgui_context;
    ImGuiImplWrapper imgui_imp{window.get(), true};

    simple::Renderer renderer;

    Camera camera{window};
    camera.getController().setMaxRadius(100);
    camera.getController().setRadius(50);


    simple::ShaderProgram simple_program{
            "void main() { gl_Position = proj_matrix * view_matrix * model_matrix * vec4(vertex_position, 1.0f); }",
            "void main() { frag_color = vec4(1.0f); }"
    };

    MeshData mesh_data = loadOBJ("assets/meshes/Low_Poly_Forest_tree01.obj");
    MeshData teapot_mesh_data = loadOBJ("assets/meshes/teapot.obj");

    simple::Mesh mesh{mesh_data.positions, mesh_data.normals, mesh_data.tex_coords, mesh_data.indices};
    simple::Mesh teapot_mesh {teapot_mesh_data.positions, teapot_mesh_data.normals, teapot_mesh_data.tex_coords,
                              teapot_mesh_data.indices};

    PhongShader phong_shader;
    phong_shader.lightPosition() = {0, 0, 100};
    phong_shader.lightColor() = {1, 1, 1};
    phong_shader.ambientLightIntensity() = .1f;
    phong_shader.specularLightIntensity() = .5f;

    GL::gl.VertexAttribI1ui(5, 0xFFffFFff);

    const auto [axes_mesh, axes_shader] = makeAxes();

    const glm::mat4 base_tree_transform =
            glm::rotate(glm::scale(glm::identity<glm::mat4>(), glm::vec3(0.001f)),
                        glm::pi<float>() / 2.f, {1, 0, 0});

    const glm::mat4 base_teapot_transform =
            glm::rotate(glm::identity<glm::mat4>(), glm::pi<float>() / 2.f, {1, 0, 0});

    auto prev_frame_start_time = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window.get()))
    {
        glfwPollEvents();

        const auto current_frame_start_time = std::chrono::steady_clock::now();
        const std::chrono::duration<float> frame_delta = current_frame_start_time - prev_frame_start_time;
        prev_frame_start_time = current_frame_start_time;

        imgui_imp.newFrame();
        ImGui::NewFrame();

        camera.getController().update(frame_delta.count());

        phong_shader.viewPosition() = camera.getController().getCameraPosition();

        ImGui::Begin("Settings");

        if (ImGui::CollapsingHeader("Illumination"))
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

        ImGui::End();

        // Render

        renderer.draw(axes_mesh, axes_shader, glm::scale(glm::mat4(1), {100, 100, 100}));
        renderer.draw(mesh, phong_shader.getRendererProgram(), glm::rotate(base_tree_transform, float(glfwGetTime()), {0, 1, 0}));
        renderer.draw(teapot_mesh, phong_shader.getRendererProgram(), glm::rotate(base_teapot_transform, float(glfwGetTime()), {0, 1, 0}));
        renderer.finishFrame(camera.getRendererCamera());

        ImGui::Render();
        imgui_imp.renderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window.get());
    }
}