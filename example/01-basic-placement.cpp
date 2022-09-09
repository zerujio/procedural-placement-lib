#include <placement.hpp>

#include <glutils/gl.h>
#include <glutils/debug.hpp>
#include <glutils/guard.hpp>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <iostream>

using glutils::gl;

// helper window struct
struct Window
{
    Window(int width, int height, const char* title, GLFWmonitor* monitor = nullptr, GLFWwindow* share = nullptr)
    : ptr(glfwCreateWindow(width, height, title, monitor, share))
    {}

    ~Window()
    {
        glfwDestroyWindow(ptr);
    }

    GLFWwindow* ptr;
};

int main()
{
    glfwInit();
    {
        // create window
        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5); // compute shaders require at least version 4.3
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        Window window{1024, 768, "01-basic-placement", nullptr, nullptr};
        if (!window.ptr)
        {
            std::cerr << "Window creation failed" << std::endl;
            return -1;
        }

        // load opengl context for rendering
        glfwMakeContextCurrent(window.ptr);
        if (!glutils::loadGLContext(glfwGetProcAddress))
        {
            std::cerr << "OpenGL context loading failed" << std::endl;
            return -1;
        }
        glutils::enableDebugCallback();

        // load texture
        GLuint texture;
        glm::ivec2 texture_size;
        {
            int channels;
            std::uint8_t *texture_data = stbi_load("heightmap.png", &texture_size.x, &texture_size.y, &channels, 0);
            if (!texture_data)
            {
                std::cerr << "texture load failed: " << stbi_failure_reason() << std::endl;
                return -2;
            }
            gl.GenTextures(1, &texture);
            gl.BindTexture(GL_TEXTURE_2D, texture);
            const GLenum formats[]{GL_RED, GL_RG, GL_RGB, GL_RGBA};
            const GLenum format = formats[channels - 1];
            gl.TexImage2D(GL_TEXTURE_2D, 0, format, texture_size.x, texture_size.y, 0, format, GL_UNSIGNED_BYTE,
                          texture_data);
            gl.GenerateMipmap(GL_TEXTURE_2D);

            stbi_image_free(texture_data);
        }

        // load context for placement
        placement::loadGL(glfwGetProcAddress);

        // set world data (texture is used both as height and density map).
        placement::WorldData world_data{texture, texture};
        world_data.scale.x = texture_size.x;
        world_data.scale.z = texture_size.y;

        auto points = placement::computePlacement(world_data, 100.0f, {0.0f, 0.0f},
                                                  {world_data.scale.x, world_data.scale.z});

        std::cout << std::endl << "placement results:\n";
        for (const auto &p: points)
            std::cout << p.x << ", " << p.y << ", " << p.z << '\n';
    }
    glfwTerminate();
    return 0;
}