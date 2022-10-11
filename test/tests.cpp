#include "catch.hpp"

#include "placement/placement.hpp"
#include "placement/placement_pipeline.hpp"

#include "glutils/debug.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <memory>

template <auto DeleteFunction>
struct Deleter
{
    template <class T>
    void operator() (T* ptr) { DeleteFunction(ptr); }
};

class TestContext
{
public:
    TestContext()
    {
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        window = decltype(window)(glfwCreateWindow(1, 1, "TEST", nullptr, nullptr));
        if (!window)
            throw std::runtime_error("window creation failed");
        glfwMakeContextCurrent(window.get());

        if (!gladLoadGLContext(&gl, glfwGetProcAddress) or !placement::loadGLContext(glfwGetProcAddress))
            throw std::runtime_error("OpenGL context loading failed");

        gl.DebugMessageCallback(TestContext::GLDebugMessage, nullptr);
    }

    auto loadTexture(const char* path) const -> GLuint
    {
        GLuint texture;
        glm::ivec2 texture_size;
        int channels;
        std::unique_ptr<stbi_uc[], Deleter<stbi_image_free>> texture_data
                {stbi_load(path, &texture_size.x, &texture_size.y, &channels, 0)};

        REQUIRE(texture_data);

        gl.GenTextures(1, &texture);
        gl.BindTexture(GL_TEXTURE_2D, texture);
        const GLenum formats[]{GL_RED, GL_RG, GL_RGB, GL_RGBA};
        const GLenum format = formats[channels - 1];
        gl.TexImage2D(GL_TEXTURE_2D, 0, format, texture_size.x, texture_size.y, 0, format, GL_UNSIGNED_BYTE,
                      texture_data.get());
        gl.GenerateMipmap(GL_TEXTURE_2D);

        return texture;
    }

    static void GLDebugMessage(GLenum source, GLenum type, unsigned int id, GLenum severity, GLsizei length,
                               const char *message, const void *user_ptr)
    {
#define GL_DEBUG_MESSAGE "[OpenGL debug message " << id << "]\n"                                        \
                            << "Source  : " << glutils::getDebugMessageSourceString(source) << "\n"     \
                            << "Type    : " << glutils::getDebugMessageTypeString(type)     << "\n"     \
                            << "Severity: " << glutils::getDebugMessageSeverityString(severity) << "\n" \
                            << "Message : " << message << "\n"

        switch (severity)
        {
            case GL_DEBUG_SEVERITY_HIGH:
                FAIL(GL_DEBUG_MESSAGE);
                break;

            case GL_DEBUG_SEVERITY_MEDIUM:
                FAIL_CHECK(GL_DEBUG_MESSAGE);
                break;

            case GL_DEBUG_SEVERITY_LOW:
                WARN(GL_DEBUG_MESSAGE);
                break;

            case GL_DEBUG_SEVERITY_NOTIFICATION:
            default:
                INFO(GL_DEBUG_MESSAGE)
                break;
        }
    }

private:
    struct GLFWInitGuard
    {
        GLFWInitGuard() { glfwInit(); }
        ~GLFWInitGuard() { glfwTerminate(); }
    } glfw_init_guard;

    std::unique_ptr<GLFWwindow, Deleter<glfwDestroyWindow>> window;

    GladGLContext gl {};
};

TEST_CASE("PlacementPipeline", "[placement][pipeline]")
{
    TestContext context;

    const GLuint white_texture = context.loadTexture("assets/white.png");
    const GLuint black_texture = context.loadTexture("assets/black.png");

    placement::PlacementPipeline pipeline;
    const glm::vec3 world_scale = {10.0f, 1.0f, 10.0f};
    pipeline.setWorldScale(world_scale);
    pipeline.setHeightTexture(black_texture);
    pipeline.setDensityTexture(white_texture);

    SECTION("Placement with < 0 area should return an empty vector")
    {
        auto points = pipeline.computePlacement(1.0f, {0.0f, 0.0f}, {-1.0f, -1.0f});
        CHECK(points.empty());

        points = pipeline.computePlacement(1.0f, {0.0f, 0.0f}, {10.0f, -1.0f});
        CHECK(points.empty());

        points = pipeline.computePlacement(1.0f, {0.0f, 0.0f}, {-1.0f, 10.0f});
        CHECK(points.empty());
    }

    SECTION("Placement with space for a single object")
    {
        auto points = pipeline.computePlacement(0.5f, {0.0f, 0.0f}, {1.0f, 1.0f});
        CHECK(points.size() == 1);

        points = pipeline.computePlacement(0.5f, {1.5f, 1.5f}, {2.5f, 2.5f});
        CHECK(points.size() == 1);
    }

    SECTION("Full area placement")
    {
        constexpr float footprint = 0.5f;
        auto points = pipeline.computePlacement(footprint, {0.0f, 0.0f}, {world_scale.x + footprint, world_scale.z + footprint});
        CHECK(points.size() == 100);
    }
}

TEST_CASE("GenerationKernel", "[generation][kernel][pipeline]")
{
    TestContext context;

    placement::GenerationKernel kernel;

    CHECK(kernel.getHeightmapTexUnit() == placement::GenerationKernel::s_default_heightmap_tex_unit);
    CHECK(kernel.getDensitymapTexUnit() == placement::GenerationKernel::s_default_densitymap_tex_unit);
    CHECK(kernel.getPositionBufferBinding() == placement::GenerationKernel::s_default_position_buffer_binding);
    CHECK(kernel.getIndexBufferBinding() == placement::GenerationKernel::s_default_index_buffer_binding);
}

TEST_CASE("ReductionKernel", "[reduction][kernel][pipeline]")
{
    TestContext context;

    placement::ReductionKernel kernel;

    CHECK(kernel.getPositionBufferBinding() == placement::ReductionKernel::s_default_position_buffer_binding);
    CHECK(kernel.getIndexBufferBinding() == placement::ReductionKernel::s_default_index_buffer_binding);
}