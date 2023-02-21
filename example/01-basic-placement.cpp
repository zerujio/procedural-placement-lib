#include "placement/placement.hpp"

#include "glad/gl.h"
#include "GLFW/glfw3.h"
#include "stb_image.h"
#include "glm/glm.hpp"
#include "glutils/debug.hpp"

#include <iostream>
#include <vector>
#include <chrono>

static GladGLContext gl;

int main()
{
    glfwInit();

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4); // compute shaders require at least version 4.3,
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5); // and DSA requires at least 4.5
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

    GLFWwindow* window = glfwCreateWindow(1024, 768, "01-basic-placement", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Window creation failed" << std::endl;
        return -1;
    }

    glfwMakeContextCurrent(window);
    if (!gladLoadGLContext(&gl, glfwGetProcAddress))
    {
        std::cerr << "OpenGL context loading failed" << std::endl;
        return -1;
    }

    gl.Enable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

    // load texture
    GLuint texture;
    glm::ivec2 texture_size;
    {
        int channels;
        std::uint8_t *texture_data = stbi_load("assets/heightmap.png", &texture_size.x, &texture_size.y, &channels, 0);
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

    // placement-lib
    {
        // load the GL context in the placement lib
        if (!placement::loadGLContext(glfwGetProcAddress))
        {
            std::cerr << "placement::loadGLContext failed" << std::endl;
            return -1;
        }
        GL::enableDebugCallback();

        // instantiate the placement pipeline. This will load and compile the required compute shaders.
        placement::PlacementPipeline pipeline;

        // use same texture for height and density
        const placement::WorldData world_data {/*scale=*/{texture_size.x, 1.0f, texture_size.y}, /*heightmap=*/texture};
        const placement::LayerData layer_data {/*footprint=*/5.0f, /*densitymaps=*/{{texture}}};

        const glm::vec2 lower_bound{glm::vec2(texture_size.x, texture_size.y) / 2.0f};
        const glm::vec2 upper_bound{lower_bound + 100.0f};

        placement::FutureResult future_result = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound);

        const auto start_time = std::chrono::steady_clock::now();

        const placement::Result results = future_result.readResult();

        const auto wait_time = std::chrono::steady_clock::now() - start_time;
        std::cout << "waited for " << wait_time.count() << "ns\n";

        std::vector<placement::Result::Element> result_vector;
        result_vector.resize(results.getElementArrayLength());
        results.copyAllToHost(result_vector.begin());

        std::cout << std::endl << "placement results:\n";
        for (const auto &element: result_vector)
            std::cout << "position={" << element.position.x << ", " << element.position.y << ", " << element.position.z
                    << "}, class_index=" << element.class_index << '\n';
    }

    glfwTerminate();
    return 0;
}