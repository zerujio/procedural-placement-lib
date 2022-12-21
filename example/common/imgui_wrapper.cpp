#include "imgui_wrapper.hpp"

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <stdexcept>

ImGuiContextWrapper::ImGuiContextWrapper()
{
    IMGUI_CHECKVERSION();
    m_context = ImGui::CreateContext();
}

ImGuiContextWrapper::~ImGuiContextWrapper()
{
    ImGui::DestroyContext(m_context);
}

ImGuiImplWrapper::ImGuiImplWrapper(GLFWwindow *window, bool install_callbacks)
{
    if (!ImGui_ImplGlfw_InitForOpenGL(window, install_callbacks))
        throw std::runtime_error("ImGui_ImplGlfw_InitForOpenGL failed");
    if (!ImGui_ImplOpenGL3_Init())
        throw std::runtime_error("ImGui_ImplOpenGL3_Init failed");
}

ImGuiImplWrapper::~ImGuiImplWrapper()
{
    ImGui_ImplGlfw_Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
}

void ImGuiImplWrapper::newFrame() const
{
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplOpenGL3_NewFrame();
}

void ImGuiImplWrapper::renderDrawData(ImDrawData *data) const
{
    ImGui_ImplOpenGL3_RenderDrawData(data);
}
