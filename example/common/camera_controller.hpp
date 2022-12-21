#ifndef PROCEDURALPLACEMENTLIB_CAMERA_CONTROLLER_HPP
#define PROCEDURALPLACEMENTLIB_CAMERA_CONTROLLER_HPP

#include "glm/vec2.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/common.hpp"

namespace simple { class Camera; }

namespace GLFW { class Window; }

class CameraController
{
public:
    CameraController(simple::Camera& camera, GLFW::Window& window);

    void update(float delta);

    [[nodiscard]] glm::vec2 getPosition() const { return m_position; }
    void setPosition(glm::vec2 position) { m_setClamped(m_position, position, m_min_position, m_max_position); }

    [[nodiscard]] glm::vec2 getMaxPosition() const { return m_max_position; }
    void setMaxPosition(glm::vec2 max_position)
    {
        m_max_position = max_position;
        setPosition(getPosition());
    }

    [[nodiscard]] glm::vec2 getMinPosition() const { return m_min_position; }
    void setMinPosition(glm::vec2 min_position)
    {
        m_min_position = min_position;
        setPosition(getPosition());
    }

    [[nodiscard]] float getSpeed() const { return m_speed; }
    void setSpeed(float speed) { m_speed = speed; }

    [[nodiscard]] glm::vec2 getAngle() const { return m_angle; }
    void setAngle(glm::vec2 angle) { m_setClamped(m_angle, angle, m_min_angle, m_max_angle); }

    [[nodiscard]] glm::vec2 getMaxAngle() const { return m_max_angle; }
    void setMaxAngle(glm::vec2 max_angle)
    {
        m_max_angle = max_angle;
        setAngle(getAngle());
    }

    [[nodiscard]] glm::vec2 getMinAngle() const { return m_min_angle; }
    void setMinAngle(glm::vec2 min_angle)
    {
        m_min_angle = min_angle;
        setAngle(getAngle());
    }

    [[nodiscard]] float getAngularSpeed() const { return m_angular_speed; }
    void setAngularSpeed(float angular_speed) { m_angular_speed = angular_speed; }

    [[nodiscard]] float getRadius() const { return m_radius; }
    void setRadius(float radius) { m_set(m_radius, glm::clamp(radius, m_min_radius, m_max_radius)); }

    [[nodiscard]] float getMaxRadius() const { return m_max_radius; }
    void setMaxRadius(float max_radius)
    {
        m_max_radius = max_radius;
        setRadius(getRadius());
    }

    [[nodiscard]] float getMinRadius() const { return m_min_radius; }
    void setMinRadius(float min_radius)
    {
        m_min_radius = min_radius;
        setRadius(getRadius());
    }

    [[nodiscard]] float getRadialSpeed() const { return m_radial_speed; }
    void setRadialSpeed(float radial_speed) { m_radial_speed = radial_speed; }

private:
    void m_updateViewMatrix() const;

    template<typename T>
    void m_set(T& variable, T value)
    {
        variable = value;
        m_dirty = true;
    }

    template<typename T>
    void m_setClamped(T&variable, T value, T min, T max)
    {
        m_set(variable, glm::clamp(value, min, max));
    }

    simple::Camera& m_camera;
    GLFW::Window& m_window;

    glm::vec2 m_position {0.0f};
    glm::vec2 m_max_position {1.0f};
    glm::vec2 m_min_position {0.0f};
    float m_speed {1.0f};

    glm::vec2 m_angle {0.0f, glm::pi<float>() / 2.0f};
    glm::vec2 m_max_angle {std::numeric_limits<float>::max(), glm::pi<float>() / 2.0f};
    glm::vec2 m_min_angle {std::numeric_limits<float>::min(), 0.01f};
    float m_angular_speed {glm::pi<float>() * 0.1f};

    float m_radius {1.0f};
    float m_max_radius {1.0f};
    float m_min_radius {0.1f};
    float m_radial_speed {1.0f};

    float m_scroll_input {0.0f};
    glm::vec2 m_cursor_prev {0.0f};

    bool m_dirty {true};
    bool m_dragging {false};
};


#endif //PROCEDURALPLACEMENTLIB_CAMERA_CONTROLLER_HPP
