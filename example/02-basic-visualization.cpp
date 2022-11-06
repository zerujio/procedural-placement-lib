#include "placement/placement.hpp"
#include "glutils/debug.hpp"
#include "simple-renderer/renderer.hpp"
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "stb_image.h"

#include <stdexcept>
#include <memory>


struct GLFWError : public std::runtime_error
{
    // construct an exception with the message returned by glfwGetError()
    GLFWError() : std::runtime_error(getGLFWErrorString()) {}

private:
    static auto getGLFWErrorString() -> const char*
    {
        const char* str;
        glfwGetError(&str);
        return str;
    }
};

struct GLFWInitGuard
{
    GLFWInitGuard()
    {
        if (glfwInit() != GLFW_TRUE)
            throw GLFWError();
    }

    virtual ~GLFWInitGuard()
    {
        glfwTerminate();
    }
};

unsigned int loadTexture(const char* filename)
{
    glm::ivec2 texture_size;
    int channels;

    auto texture_data = stbi_load(filename,&texture_size.x, &texture_size.y, &channels, 0);

    if (!texture_data)
        throw std::runtime_error(stbi_failure_reason());

    using glutils::gl;

    unsigned int texture;
    gl.GenTextures(1, &texture);

    gl.BindTexture(GL_TEXTURE_2D, texture);

    {
        const GLenum formats[]{GL_RED, GL_RG, GL_RGB, GL_RGBA};
        const GLenum format = formats[channels - 1];
        gl.TexImage2D(GL_TEXTURE_2D, 0, format, texture_size.x, texture_size.y, 0, format, GL_UNSIGNED_BYTE, texture_data);
    }

    gl.GenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(texture_data);

    return texture;
}

int main()
{
    GLFWInitGuard glfw_init_guard;

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    std::unique_ptr<GLFWwindow, void(*)(GLFWwindow*)> glfw_window (
            glfwCreateWindow(600, 600, "02-basic-visualization", nullptr, nullptr),
            glfwDestroyWindow);

    if (!glfw_window)
        throw GLFWError();

    glfwMakeContextCurrent(glfw_window.get());

    if (!glutils::loadGLContext(glfwGetProcAddress))
        throw std::runtime_error("Failed to load OpenGL context");

    glutils::enableDebugCallback();
    glutils::gl.Enable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    //placement
    GLuint white_texture = loadTexture("assets/white.png");
    GLuint black_texture = loadTexture("assets/black.png");

    placement::PlacementPipeline pipeline;
    pipeline.setDensityTexture(white_texture);
    pipeline.setHeightTexture(black_texture);
    pipeline.setWorldScale({1.f, 1.f, 1.f});

    const std::vector<glm::vec3> positions = pipeline.computePlacement(0.01f, glm::vec2(0.f), glm::vec2(1.f));

    // rendering
    simple::Renderer renderer;

    simple::Camera camera;

    simple::ShaderProgram program ("void main() {gl_Position = vec4((vertex_position.xzy - vec3()), 1.0f);}",
                                   "void main() {frag_color = vec4(1.0f);}");

    simple::Mesh point_mesh (positions);
    point_mesh.setDrawMode(simple::Mesh::DrawMode::points);

    while (!glfwWindowShouldClose(glfw_window.get()))
    {
        renderer.draw(program, point_mesh, glm::mat4(1.0f));

        renderer.finishFrame(camera);

        glfwSwapBuffers(glfw_window.get());
        glfwPollEvents();
    }
}
