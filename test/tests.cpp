#include "placement/placement.hpp"
#include "placement/placement_pipeline.hpp"

#include "glutils/debug.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <memory>
#include <random>
#include <ostream>

// defined to here to make it available to catch.hpp
template<auto L, typename T, auto Q>
auto operator<< (std::ostream& out, glm::vec<L, T, Q> v) -> std::ostream&
{
    constexpr auto sep = ", ";
    out << "{" << v.x << sep << v.y;
    if constexpr (L > 2)
    {
        out << sep << v.z;
        if constexpr (L > 3)
            out << sep << v.w;
    }
    return out << "}";
}

template<class T>
auto operator<< (std::ostream& out, std::vector<T> vector) ->std::ostream&
{
    if (vector.empty())
        return out << "[]";

    out << "[";
    for (const T& x : vector)
        out << x << ", ";
    return out << "]";
}

template<class T>
auto operator<< (std::ostream& out, placement::PlacementPipelineKernel::Candidate candidate) -> std::ostream &
{
    return out << "{" << candidate.position << ", " << candidate.index << "}";
}

#include "catch.hpp"

using namespace placement;

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

    [[nodiscard]]
    auto getGL() const -> const GladGLContext& {return gl;}

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
        const auto bounds = GENERATE(std::make_pair(glm::vec2{0.0f, 0.0f}, glm::vec2{1.0f, 1.0f}),
                                     std::make_pair(glm::vec2(1.5f, 1.5f), glm::vec2{2.5f, 2.5f}));
        const auto& lower_bound = bounds.first;
        const auto& upper_bound = bounds.second;

        const auto points = pipeline.computePlacement(0.5f, lower_bound, upper_bound);

        REQUIRE(points.size() == 1);

        const auto& point = points.front();
        CHECK(point.x >= lower_bound.x);
        CHECK(point.z >= lower_bound.y);
        CHECK(point.x < upper_bound.x);
        CHECK(point.z < upper_bound.y);
    }

    SECTION("Full area placement")
    {
        constexpr float footprint = 0.5f;
        const glm::vec2 lower_bound {0.0f, 0.0f};
        const glm::vec2 upper_bound {world_scale.x + footprint, world_scale.z + footprint};

        auto points = pipeline.computePlacement(footprint, lower_bound, upper_bound);
        CHECK(points.size() == 100);

        for (int i = 0; i < points.size(); i++)
        {
            INFO("i = " << i);
            INFO("points[i] = {" << points[i].x << ", " << points[i].y << ", " << points[i].z << "}");
            const glm::vec2 point2d {points[i].x, points[i].z};
            CHECK(glm::all(glm::greaterThanEqual(point2d, lower_bound) && glm::lessThan(point2d, upper_bound)));
            for (int j = 0; j < i; j++)
            {
                INFO("j = " << j);
                INFO("points[j] = {" << points[j].x << ", " << points[j].y << ", " << points[j].z << "}");
                CHECK(glm::length(point2d - glm::vec2(points[j].x, points[j].z)) >= 2 * footprint);
            }
        }
    }
}

TEMPLATE_TEST_CASE("Common PlacementPipelineKernel operations", "[kernel][pipeline]", GenerationKernel, ReductionKernel)
{
    TestContext context;

    TestType kernel;

    SECTION("Default SSB binding points")
    {
        CHECK(kernel.getShaderStorageBlock().getBindingIndex() == TestType::default_ssb_binding);;
    }

    SECTION("Binding point operations")
    {
        glutils::Guard<glutils::Buffer> buffer;
        constexpr std::size_t element_count = 8;
        constexpr GLsizeiptr buffer_size = sizeof(typename TestType::Candidate) * element_count;
        buffer->allocateImmutable(buffer_size, glutils::Buffer::StorageFlags(), nullptr);

        const GLuint binding_index = GENERATE(1, 4, 7, 3, 0, 2);

        const auto ssb = kernel.getShaderStorageBlock();
        ssb.setBindingIndex(binding_index);
        CHECK(ssb.getBindingIndex() == binding_index);
    }
}

TEST_CASE("GenerationKernel", "[generation][kernel][pipeline]")
{
    TestContext context;

    GenerationKernel kernel;

    SECTION("default value")
    {
        CHECK(kernel.getHeightmapSampler().getTextureUnit() == GenerationKernel::s_default_heightmap_tex_unit);
        CHECK(kernel.getDensitymapSampler().getTextureUnit() == GenerationKernel::s_default_densitymap_tex_unit);
    }

    SECTION("set texture unit")
    {
        auto getSampler = GENERATE(&GenerationKernel::getHeightmapSampler, &GenerationKernel::getDensitymapSampler);
        const auto sampler = (kernel.*getSampler)();

        const auto texture_unit = GENERATE(1, 3, 0, 6, 4);
        sampler.setTextureUnit(texture_unit);
        CHECK(sampler.getTextureUnit() == texture_unit);
    }
}

TEST_CASE("ReductionKernel", "[reduction][kernel][pipeline]")
{
    TestContext context;
    ReductionKernel kernel;

    using Candidate = ReductionKernel::Candidate;

    auto sequential_reduction = [](std::vector<Candidate> candidate_buffer)
    {
        const auto buffer_copy = candidate_buffer;

        unsigned int last = 0;
        for (auto& x : candidate_buffer)
        {
            x.index += last;
            last = x.index;
        }

        for (int i = 0; i < candidate_buffer.size(); i++)
        {
            const bool valid = buffer_copy[i].index;
            if (valid)
                candidate_buffer[candidate_buffer[i].index - 1].position = buffer_copy[i].position;
        }

        return candidate_buffer;
    };

    std::default_random_engine generator {0};
    std::uniform_real_distribution distribution;
    auto random_vec3 = [&]()
    {
        return glm::vec3(distribution(generator), distribution(generator), distribution(generator));
    };

    using Indices = std::vector<unsigned int>;
    const std::vector<unsigned int> indices = GENERATE(Indices{0}, Indices{1},
                                                    Indices{0, 1}, Indices{1, 0}, Indices{0, 0}, Indices{1, 1},
                                                    Indices{0, 1, 0, 1}, Indices{1, 1, 0, 1}, Indices{1, 1, 1, 1},
                                                    Indices{1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1},
                                                    Indices{0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1,
                                                            1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1});

    std::vector<Candidate> candidates;
    candidates.reserve(indices.size());
    for (auto index : indices)
        candidates.emplace_back(Candidate{random_vec3(), index});

    const auto buffer_size = candidates.size() * sizeof(Candidate);

    glutils::Guard<glutils::Buffer> buffer;
    buffer->allocateImmutable(buffer_size,
                              glutils::Buffer::StorageFlags::map_read | glutils::Buffer::StorageFlags::dynamic_storage,
                              nullptr);
    buffer->write(0, buffer_size, candidates.data());
    buffer->bindRange(glutils::Buffer::IndexedTarget::shader_storage, ReductionKernel::default_ssb_binding, 0, buffer_size);

    const auto num_work_groups = indices.size() / ReductionKernel::work_group_size
            + indices.size() % ReductionKernel::work_group_size != 0;

    kernel(num_work_groups); // <<< dispatch compute
    context.getGL().MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    auto results = static_cast<const Candidate*>(buffer->map(glutils::Buffer::AccessMode::read_only));
    const auto expected = sequential_reduction(candidates);

    for (std::size_t i = 0; i < candidates.size(); i++)
    {
        INFO("i=" << i);
        CHECK(results[i].index == expected[i].index);
        CHECK(results[i].position == expected[i].position);
    }
    buffer->unmap();
}