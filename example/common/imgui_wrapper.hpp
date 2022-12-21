#ifndef PROCEDURALPLACEMENTLIB_IMGUI_WRAPPER_HPP
#define PROCEDURALPLACEMENTLIB_IMGUI_WRAPPER_HPP

struct GLFWwindow;
struct ImGuiContext;
struct ImDrawData;

class ImGuiContextWrapper
{
public:
    ImGuiContextWrapper();
    ~ImGuiContextWrapper();

    [[nodiscard]] ImGuiContext* get() const {return m_context;}

private:
    ImGuiContext* m_context;
};

struct ImGuiImplWrapper
{
    ImGuiImplWrapper(GLFWwindow* window, bool install_callbacks);
    ~ImGuiImplWrapper();

    void newFrame() const;
    void renderDrawData(ImDrawData* data) const;
};

#endif //PROCEDURALPLACEMENTLIB_IMGUI_WRAPPER_HPP
