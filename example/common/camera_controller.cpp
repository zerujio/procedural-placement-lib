#include "camera_controller.hpp"
#include "glfw_wrapper.hpp"

#include "simple-renderer/camera.hpp"

#include "glm/vec3.hpp"
#include "glm/trigonometric.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "imgui.h"

CameraController::CameraController(simple::Camera &camera, GLFW::Window& window) : m_camera(camera), m_window(window)
{
    window.setMouseButtonCallback([this](GLFW::Window& window, int button, int action, int)
    {
        if (ImGui::GetIO().WantCaptureMouse || button != GLFW_MOUSE_BUTTON_1)
            return;

        if (action == GLFW_PRESS)
        {
            m_dragging = true;

            double x, y;
            glfwGetCursorPos(window.get(), &x, &y);
            m_cursor_prev = glm::vec2(x, y);
        }
        else if (action == GLFW_RELEASE)
            m_dragging = false;
    });

    window.setScrollCallback([this](GLFW::Window&, double, double yoffset)
    {
        if (ImGui::GetIO().WantCaptureMouse)
            return;

        m_scroll_input += yoffset * m_radius;
    });
}

void CameraController::update(float delta)
{
    using namespace glm;

    auto get_key = [this](int key)
    {
        return glfwGetKey(m_window.get(), key) == GLFW_PRESS;
    };

    const int forward_input = get_key(GLFW_KEY_W) - get_key(GLFW_KEY_S);
    const int lateral_input = get_key(GLFW_KEY_D) - get_key(GLFW_KEY_A);

    if (forward_input || lateral_input)
    {
        const float distance = m_speed * delta * m_radius;

        const vec3 forward {-sin(m_angle.x), -cos(m_angle.x), 0.f};
        m_position += float(forward_input) * forward * distance;

        const vec3 side = cross(forward, vec3(0, 0, 1));
        m_position += float(lateral_input) * side * distance;

        setPosition(m_position); // clamp and mark state dirty
    }

    if (m_dragging)
    {
        dvec2 cursor;
        glfwGetCursorPos(m_window.get(), &cursor.x, &cursor.y);
        const vec2 cursor_delta = glm::vec2(cursor) - m_cursor_prev;
        m_cursor_prev = cursor;

        setAngle(m_angle + cursor_delta * delta * m_angular_speed * glm::vec2(1, -1));
    }

    if (m_scroll_input != 0)
    {
        setRadius(m_radius - m_scroll_input * delta * m_radial_speed);
        m_scroll_input = 0.0f;
    }

    if (m_dirty)
    {
        m_updateViewMatrix();
        m_dirty = false;
    }
}

glm::vec3 CameraController::getCameraPosition() const
{
    return  m_position + glm::vec3{
            m_radius * glm::sin(m_angle.y) * glm::sin(m_angle.x),
            m_radius * glm::sin(m_angle.y) * glm::cos(m_angle.x),
            m_radius * glm::cos(m_angle.y)
    };
}

void CameraController::m_updateViewMatrix() const
{
    m_camera.setViewMatrix(glm::lookAt(getCameraPosition(), m_position, {0.0f, 0.0f, 1.0f}));
}
