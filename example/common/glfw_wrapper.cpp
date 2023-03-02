#include "glfw_wrapper.hpp"

namespace GLFW
{

template<auto CallbackPtr, typename ... Args>
void Window::s_invoke(GLFWwindow *glfw_window, Args... args)
{
    auto window = static_cast<Window *>(glfwGetWindowUserPointer(glfw_window));
    const auto &[current_callback, prev_callback] = window->*CallbackPtr;

    if (prev_callback)
        prev_callback(glfw_window, args...);

    if (current_callback)
        current_callback(*window, args...);
}

Window::Window(const char *title, glm::uvec2 initial_size) :
        m_window(glfwCreateWindow(initial_size.x, initial_size.y, title, nullptr, nullptr))
{
    if (!m_window)
        throw Error();

    glfwMakeContextCurrent(m_window.get());

    if (!GL::loadGLContext(glfwGetProcAddress))
        throw std::runtime_error("Failed to load OpenGL context");

    const auto on_frame_buffer_size_changed = [](GLFWwindow* window, int width, int height)
    {
        GL::gl.Viewport(0, 0, width, height);
        s_invoke<&Window::m_framebuffer_size_callback>(window, width, height);
    };

    m_framebuffer_size_callback.second = glfwSetFramebufferSizeCallback(m_window.get(), on_frame_buffer_size_changed);
    m_scroll_callback.second = glfwSetScrollCallback(m_window.get(), s_invoke<&Window::m_scroll_callback>);
    m_mouse_button_callback.second = glfwSetMouseButtonCallback(m_window.get(), s_invoke<&Window::m_mouse_button_callback>);

    glfwSetWindowUserPointer(m_window.get(), this);
}

Window::~Window()
{
    glfwSetFramebufferSizeCallback(m_window.get(), m_framebuffer_size_callback.second);
    glfwSetScrollCallback(m_window.get(), m_scroll_callback.second);
    glfwSetMouseButtonCallback(m_window.get(), m_mouse_button_callback.second);

    glfwSetWindowUserPointer(m_window.get(), nullptr);
}

}