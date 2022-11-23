#include "placement/placement.hpp"
#include "glutils/debug.hpp"
#include "glutils/guard.hpp"
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

void onWindowResize(GLFWwindow* _window, int width, int height)
{
    glutils::gl.Viewport(0, 0, width, height);
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

    glfwSetWindowSizeCallback(glfw_window.get(), onWindowResize);

    //placement
    GLuint densitymap = loadTexture("assets/heightmap.png"); // deliberately using heightmap as densitymap
    GLuint heightmap = loadTexture("assets/black.png");

    placement::PlacementPipeline pipeline;
    pipeline.setDensityTexture(densitymap);
    pipeline.setHeightTexture(heightmap);
    pipeline.setWorldScale({1.f, 1.f, 1.f});
    pipeline.setRandomSeed(89581751);

    pipeline.computePlacement(0.001f, glm::vec2(0.f), glm::vec2(1.f));

    // copy results between gpu buffers
    using namespace glutils;
    Buffer buffer = Buffer::create();
    buffer.allocate(pipeline.getResultsSize() * sizeof(glm::vec4), Buffer::Usage::dynamic_draw);

    pipeline.copyResultsToGPUBuffer(buffer.getName());

    VertexArray vertex_array = VertexArray::create();
    vertex_array.bindVertexBuffer(0, buffer, 0, sizeof(glm::vec4));
    vertex_array.bindAttribute(0 /*attrib index*/, 0 /*buffer index*/);
    vertex_array.setAttribFormat(0 /*attrib index*/, VertexArray::AttribSize::three, VertexArray::AttribType::float_, false, 0);
    vertex_array.enableAttribute(0);

    // rendering
    simple::Renderer renderer;

    simple::Camera camera;

    simple::ShaderProgram program ("void main() {gl_Position = vec4(vertex_position.xzy * vec3(2.f, 2.f, 1.f) - vec3(1.f, 1.f, 0.f), 1.0f);}",
                                   "void main() {frag_color = vec4(1.0f);}");

    simple::Mesh point_mesh (Guard<Buffer>(buffer), Guard<VertexArray>(vertex_array), GL_POINTS, pipeline.getResultsSize(), 0);

    while (!glfwWindowShouldClose(glfw_window.get()))
    {
        renderer.draw(program, point_mesh, glm::mat4(1.0f));

        renderer.finishFrame(camera);

        glfwSwapBuffers(glfw_window.get());
        glfwPollEvents();
    }
}
