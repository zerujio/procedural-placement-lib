#include "placement/placement.hpp"
#include "placement/placement_pipeline.hpp"

#include "glutils/debug.hpp"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>

#include <memory>
#include <ostream>
#include <algorithm>

// included here to make it available to catch.hpp
#include "ostream_operators.hpp"
#include "catch.hpp"

using namespace placement;

GladGLContext gl;

auto loadTexture(const char* path) -> GLuint
{
    GLuint texture;
    glm::ivec2 texture_size;
    int channels;
    std::unique_ptr<stbi_uc[]> texture_data {stbi_load(path, &texture_size.x, &texture_size.y, &channels, 0)};

    if (!texture_data)
        throw std::runtime_error(stbi_failure_reason());

    gl.GenTextures(1, &texture);
    gl.BindTexture(GL_TEXTURE_2D, texture);
    const GLenum formats[]{GL_RED, GL_RG, GL_RGB, GL_RGBA};
    const GLenum format = formats[channels - 1];
    gl.TexImage2D(GL_TEXTURE_2D, 0, format, texture_size.x, texture_size.y, 0, format, GL_UNSIGNED_BYTE,
                  texture_data.get());
    gl.GenerateMipmap(GL_TEXTURE_2D);

    return texture;
}

class ContextInitializer : public Catch::TestEventListenerBase
{
public:
    using Catch::TestEventListenerBase::TestEventListenerBase;

    void testRunStarting(const Catch::TestRunInfo&) override
    {
        if (!glfwInit())
        {
            const char* msg = nullptr;
            glfwGetError(&msg);
            throw std::runtime_error(msg);
        }

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

        m_window = glfwCreateWindow(1, 1, "TEST", nullptr, nullptr);
        if (!m_window)
        {
            const char* msg = nullptr;
            glfwGetError(&msg);
            throw std::runtime_error(msg);
        }
        glfwMakeContextCurrent(m_window);

        if (!gladLoadGLContext(&gl, glfwGetProcAddress) or !placement::loadGLContext(glfwGetProcAddress))
            throw std::runtime_error("OpenGL context loading failed");

        gl.DebugMessageCallback(glDebugCallback, nullptr);
        gl.Enable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }

    void testRunEnded(const Catch::TestRunStats&) override
    {
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    void sectionEnded(const Catch::SectionStats&) override
    {
        gl.Finish();
    }

private:

    GLFWwindow* m_window {nullptr};

    static void glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                                const GLchar* message, const void* user_ptr)
    {
        if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
            return;

        UNSCOPED_INFO("[GL DEBUG MESSAGE " << id << "] " << message);
    }
};

CATCH_REGISTER_LISTENER(ContextInitializer)


TEST_CASE("PlacementPipeline", "[placement][pipeline]")
{
    placement::PlacementPipeline pipeline;

    const glm::vec3 world_scale = {10.0f, 1.0f, 10.0f};
    pipeline.setWorldScale(world_scale);

    const auto black_texture = loadTexture("assets/black.png");
    pipeline.setHeightTexture(black_texture);

    const auto white_texture = loadTexture("assets/white.png");
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

    SECTION("determinism (trivial)")
    {
        pipeline.setWorldScale({1.0f, 1.0f, 1.0f});

        const auto positions0 = pipeline.computePlacement(0.01f, glm::vec2(0.f), glm::vec2(1.f));
        const auto positions1 = pipeline.computePlacement(0.01f, glm::vec2(0.f), glm::vec2(1.f));

        CHECK(!positions0.empty());
        CHECK(!positions1.empty());
        CHECK(positions0.size() == positions1.size());
        CHECK(positions0 == positions1);
    }

    const float footprint = GENERATE(take(3, random(0.01f, 0.1f)));
    INFO("footprint = " << footprint);

    const float boundary_offset_x = GENERATE(take(3, random(0.f, 0.8f)));
    const float boundary_offset_y = GENERATE(take(3, random(0.f, 0.8f)));
    const glm::vec2 lower_bound(boundary_offset_x, boundary_offset_y);

    INFO("lower_bound = " << lower_bound);

    const float boundary_size_x = GENERATE(take(3, random(0.2f, 1.0f)));
    const float boundary_size_y = GENERATE(take(3, random(0.2f, 1.0f)));
    const glm::vec2 upper_bound = lower_bound + glm::vec2(boundary_size_x, boundary_size_y);

    INFO("upper_bound = " << upper_bound);

    SECTION("Boundary and separation")
    {
        auto points = pipeline.computePlacement(footprint, lower_bound, upper_bound);

        REQUIRE(!points.empty());

        for (int i = 0; i < points.size(); i++)
        {
            INFO("i = " << i);
            INFO("points[i] = {" << points[i].x << ", " << points[i].y << ", " << points[i].z << "}");
            const glm::vec2 point2d{points[i].x, points[i].z};
            CHECK(glm::all(glm::greaterThanEqual(point2d, lower_bound) && glm::lessThan(point2d, upper_bound)));

            for (int j = 0; j < i; j++)
            {
                INFO("j = " << j);
                INFO("points[j] = {" << points[j].x << ", " << points[j].y << ", " << points[j].z << "}");
                CHECK(glm::length(point2d - glm::vec2(points[j].x, points[j].z)) >= Approx(2.0f * footprint));
            }
        }
    }

    SECTION("Determinism")
    {
        const auto result0 = pipeline.computePlacement(footprint, lower_bound, upper_bound);
        REQUIRE(!result0.empty());

        const auto result1 = pipeline.computePlacement(footprint, lower_bound, upper_bound);
        REQUIRE(!result1.empty());

        REQUIRE(result0 == result1);

        const auto result2 = pipeline.computePlacement(footprint * 1.3f , lower_bound, upper_bound);

        CHECK(result0 != result2);
    }

    const GLuint textures[] = {white_texture, black_texture};
    gl.DeleteTextures(2, &textures[0]);
}

TEMPLATE_TEST_CASE("Common PlacementPipelineKernel operations", "[kernel][pipeline]", GenerationKernel, ReductionKernel)
{
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

        const GLuint binding_index = GENERATE(range(1, 8));

        const auto ssb = kernel.getShaderStorageBlock();
        ssb.setBindingIndex(binding_index);
        CHECK(ssb.getBindingIndex() == binding_index);
    }
}

TEST_CASE("GenerationKernel", "[generation][kernel][pipeline]")
{
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

    SECTION("correctness")
    {
        const glm::vec3 world_scale {1.0f};

        const auto footprint = GENERATE(take(3, random(0.0f, 0.5f)));
        INFO("footprint=" << footprint);

        const auto offset_x = GENERATE(take(3, random(0.0f, 1.0f)));
        const auto offset_y = GENERATE(take(3, random(0.0f, 1.0f)));
        const glm::vec2 lower_bound {offset_x, offset_y};
        INFO("lower_bound=" << lower_bound);

        const auto length_x = GENERATE(take(3, random(0.0f, 1.0f)));
        const auto length_y = GENERATE(take(3, random(0.0f, 1.0f)));
        const glm::vec2 upper_bound = lower_bound + glm::vec2(length_x, length_y);
        INFO("upper_bound=" << upper_bound);

        const auto white_texture = loadTexture("assets/white.png");
        const auto black_texture = loadTexture("assets/black.png");
        gl.BindTextureUnit(GenerationKernel::s_default_heightmap_tex_unit, black_texture);
        gl.BindTextureUnit(GenerationKernel::s_default_densitymap_tex_unit, white_texture);

        const auto num_work_groups = kernel.setArgs(world_scale, footprint, lower_bound, upper_bound);
        const auto num_invocations = num_work_groups * GenerationKernel::work_group_size;

        REQUIRE(num_work_groups != glm::uvec2(0, 0));

        glutils::Guard<glutils::Buffer> buffer;
        buffer->allocateImmutable(sizeof(GenerationKernel::Candidate) * num_invocations.x * num_invocations.y,
                                  glutils::Buffer::StorageFlags::map_read,
                                  nullptr);
        buffer->bindBase(glutils::Buffer::IndexedTarget::shader_storage, GenerationKernel::default_ssb_binding);

        kernel.dispatchCompute(num_work_groups); // << execute compute kernel

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        const auto candidate_count = num_invocations.x * num_invocations.y;

        using Candidate = GenerationKernel::Candidate;
        std::vector<Candidate> candidates;
        candidates.resize(candidate_count);
        buffer->read(0, sizeof(Candidate) * candidate_count, candidates.data());

        SECTION("boundaries")
        {
            for (int i = 0; i < candidate_count; i++)
            {
                const Candidate candidate = candidates[i];
                INFO("candidate = " << candidate);

                const glm::vec2 position2d {candidate.position.x, candidate.position.z};

                if (candidate.index)
                {
                    CHECK(candidate.index == 1);
                    CHECK(glm::all(glm::greaterThanEqual(position2d, lower_bound)));
                    CHECK(glm::all(glm::lessThan(position2d, upper_bound)));
                }
                else
                {
                    CHECK(glm::any(glm::bvec4(
                            glm::lessThan(position2d, lower_bound),
                            glm::greaterThanEqual(position2d, upper_bound))));
                }
            }
        }

        SECTION("separation")
        {
            for (int i = 0; i < candidate_count; i++)
            {
                INFO("i = " << i);
                for (int j = 0; j < i; j++)
                {
                    INFO("j = " << j);
                    CHECK(glm::length(candidates[i].position - candidates[j].position) >= Approx(2 * footprint));
                }
            }
        }

        SECTION("determinism")
        {
            std::vector<Candidate> candidates_duplicate;
            candidates_duplicate.resize(candidate_count);

            kernel.dispatchCompute(num_work_groups);

            buffer->read(0, sizeof(Candidate) * candidate_count, candidates_duplicate.data());

            std::equal(candidates.cbegin(), candidates.cend(),
                       candidates_duplicate.cbegin(), candidates_duplicate.cend(),
                       [](const Candidate& l, const Candidate& r){return l.position == r.position && l.index == r.index;});
        }

        const GLuint textures[] = {white_texture, black_texture};
        gl.DeleteTextures(2, &textures[0]);
    }
}


TEST_CASE("ReductionKernel", "[reduction][kernel][pipeline]")
{
    using Candidate = PlacementPipelineKernel::Candidate;

    ReductionKernel kernel;

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

    using Indices = std::vector<unsigned int>;
    const auto indices = GENERATE(Indices{0}, Indices{1},
                                  Indices{0, 0}, Indices{0, 1}, Indices{1, 0}, Indices{1, 1},
                                  take(6, chunk(10, random(0u, 1u))),
                                  take(5, chunk(20, random(0u, 1u))),
                                  take(3, chunk(64, random(0u, 1u))),
                                  take(3, chunk(333, random(0u, 1u))),
                                  take(1, chunk(1024, random(0u, 1u))));

    std::vector<Candidate> candidates;
    candidates.reserve(indices.size());
    for (auto index : indices)
        candidates.emplace_back(Candidate{glm::vec3(static_cast<float>(candidates.size())), index});

    const auto num_work_groups = candidates.size() / ReductionKernel::work_group_size
            + candidates.size() % ReductionKernel::work_group_size != 0;
    candidates.resize(num_work_groups * ReductionKernel::work_group_size); // insert zero padding
    const auto buffer_size = candidates.size() * sizeof(Candidate);

    glutils::Guard<glutils::Buffer> buffer;
    buffer->allocateImmutable(buffer_size, glutils::Buffer::StorageFlags::none, candidates.data());
    buffer->bindRange(glutils::Buffer::IndexedTarget::shader_storage, ReductionKernel::default_ssb_binding, 0,
                      buffer_size);

    kernel(num_work_groups); // <<< dispatch compute

    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    auto actual_values = candidates;
    buffer->read(0, buffer_size, actual_values.data());

    const auto expected_values = sequential_reduction(candidates);

    INFO("Initial values:  " << candidates      << "\ncount: " << candidates.size());
    INFO("Expected values: " << expected_values << "\ncount: " << expected_values.size());
    INFO("Actual values:   " << actual_values   << "\ncount: " << actual_values.size());
    CHECK(std::equal(expected_values.cbegin(), expected_values.cend(),
                     actual_values.cbegin(), actual_values.cend(),
                     [](const Candidate& l, const Candidate& r){return l.position == r.position && l.index == r.index;}));
}