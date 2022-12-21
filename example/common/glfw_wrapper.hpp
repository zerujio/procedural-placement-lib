#ifndef PROCEDURALPLACEMENTLIB_GLFW_WRAPPER_HPP
#define PROCEDURALPLACEMENTLIB_GLFW_WRAPPER_HPP

#include "glutils/gl.hpp" // this header includes glad/gl.h
#include "GLFW/glfw3.h"

#include "glm/vec2.hpp"

#include <stdexcept>
#include <memory>
#include <functional>

namespace GLFW
{

/// Wraps a GLFW error string in an exception.
class Error final : public std::runtime_error
{
public:
    Error() : std::runtime_error(s_getGlfwErrorString()) {}
private:
    static const char* s_getGlfwErrorString()
    {
        const char* str;
        glfwGetError(&str);
        return str;
    }
};

/// Initializes GLFW on construction, terminates it on destruction.
struct InitGuard final
{
    InitGuard()
    {
        if (glfwInit() != GLFW_TRUE)
            throw Error();
    }

    virtual ~InitGuard()
    {
        glfwTerminate();
    }
};


class Window final
{
public:
    explicit Window(const char* title, glm::uvec2 initial_size = {600, 600});

    [[nodiscard]] GLFWwindow* get() const {return m_window.get();}

    // mouse scroll input
    using ScrollCallback = std::function<void(Window&, double, double)>;

    [[nodiscard]] const ScrollCallback &getScrollCallback() const { return m_scroll_callback; }

    template<typename Function>
    void setScrollCallback(Function&& function)
    {
        m_setCallback<&Window::m_scroll_callback>(glfwSetScrollCallback, std::forward<Function>(function));
    }

    // mouse button input
    using MouseButtonCallback = std::function<void(Window&, int, int, int)>;

    [[nodiscard]] const MouseButtonCallback& getMouseButtonCallback() const { return m_mouse_button_callback; }

    template<typename Function>
    void setMouseButtonCallback(Function&& function)
    {
        m_setCallback<&Window::m_mouse_button_callback>(glfwSetMouseButtonCallback, std::forward<Function>(function));
    }

    // framebuffer resize
    using FramebufferSizeCallback = std::function<void(Window&, int, int)>;

    [[nodiscard]] const FramebufferSizeCallback& getFramebufferSizeCallback() const {return m_framebuffer_size_callback;}

    template<typename Function>
    void setFramebufferSizeCallback(Function&& function)
    {
        m_setCallback<&Window::m_framebuffer_size_callback>(glfwSetFramebufferSizeCallback,
                                                            std::forward<Function>(function));
    }

private:
    struct DestroyWindow { void operator()(GLFWwindow* window) { glfwDestroyWindow(window);} };

    template<typename ... Args>
    using glfwCallbackType = void (*)(GLFWwindow*, Args...);

    template<typename ... Args>
    using glfwSetCallbackType = glfwCallbackType<Args...> (*) (GLFWwindow*, glfwCallbackType<Args...>);

    template<auto CallbackPtr, typename ... Args>
    static void s_invoke(GLFWwindow* glfw_window, Args... args)
    {
        auto window = static_cast<Window*>(glfwGetWindowUserPointer(glfw_window));
        (window->*CallbackPtr)(*window, args...);
    }

    template<auto Callback, typename Function, typename ... Args>
    void m_setCallback(glfwSetCallbackType<Args...> glfw_set_callback, Function&& function)
    {
        auto& callback = this->*Callback;
        callback = function;
        glfw_set_callback(get(), callback ? s_invoke<Callback, Args...> : nullptr);
    }

    std::unique_ptr<GLFWwindow, DestroyWindow> m_window;
    ScrollCallback m_scroll_callback;
    MouseButtonCallback m_mouse_button_callback;
    FramebufferSizeCallback m_framebuffer_size_callback;
};

} // GLFW

#endif //PROCEDURALPLACEMENTLIB_GLFW_WRAPPER_HPP
