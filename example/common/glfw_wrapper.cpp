#include "glfw_wrapper.hpp"

namespace GLFW
{

static void onWindowResize(GLFWwindow*, int width, int height)
{
    glutils::gl.Viewport(0, 0, width, height);
}

Window::Window(const char *title, glm::uvec2 initial_size) :
        m_window(glfwCreateWindow(initial_size.x, initial_size.y, title, nullptr, nullptr))
{
    if (!m_window)
        throw Error();

    glfwMakeContextCurrent(m_window.get());

    if (!glutils::loadGLContext(glfwGetProcAddress))
        throw std::runtime_error("Failed to load OpenGL context");

    glfwSetWindowSizeCallback(m_window.get(), onWindowResize);

    glfwSetWindowUserPointer(m_window.get(), this);
}

}