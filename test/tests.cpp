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

GladGLContext gl;

auto loadTexture(const char* path) -> GLuint
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
    GLFWwindow* m_window;
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

    SECTION("Placement area == world size")
    {
        const float footprint = GENERATE(take(3, random(0.1f, 1.0f)));
        INFO("footprint = " << footprint);

        const glm::vec2 lower_bound{0.0f};
        INFO("lower_bound = " << lower_bound);

        const glm::vec2 upper_bound{world_scale.x, world_scale.z};
        INFO("upper_bound = " << upper_bound);

        SECTION("Boundary and separation")
        {
            auto points = pipeline.computePlacement(footprint, lower_bound, upper_bound);

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
            const auto reference_points = pipeline.computePlacement(footprint, lower_bound, upper_bound);

            for (int i = 0; i < 3; i++)
                CHECK(pipeline.computePlacement(footprint, lower_bound, upper_bound) == reference_points);
        }
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

        const GLuint textures[] = {white_texture, black_texture};
        gl.DeleteTextures(2, &textures[0]);
    }
}

TEST_CASE("ReductionKernel", "[reduction][kernel][pipeline]")
{
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

    using Indices = std::vector<unsigned int>;
    const auto indices = GENERATE(Indices{0}, Indices{1},
                                  Indices{0, 0}, Indices{0, 1}, Indices{1, 0}, Indices{1, 1},
                                  take(6, chunk(10, random(0u, 1u))),
                                  take(5, chunk(20, random(0u, 1u))),
                                  take(3, chunk(64, random(0u, 1u))),
                                  take(3, chunk(333, random(0u, 1u))),
                                  take(1, chunk(1024, random(0u, 1u))));

    std::default_random_engine generator {0};
    std::uniform_real_distribution distribution;
    auto random_vec3 = [&]()
    {
        return glm::vec3(distribution(generator), distribution(generator), distribution(generator));
    };

    std::vector<Candidate> candidates;
    candidates.reserve(indices.size());
    for (auto index : indices)
        candidates.emplace_back(Candidate{random_vec3(), index});

    const auto num_work_groups = candidates.size() / ReductionKernel::work_group_size
            + candidates.size() % ReductionKernel::work_group_size != 0;
    candidates.resize(num_work_groups * ReductionKernel::work_group_size); // insert zero padding
    const auto buffer_size = candidates.size() * sizeof(Candidate);

    glutils::Guard<glutils::Buffer> buffer;
    buffer->allocateImmutable(buffer_size,
                              glutils::Buffer::StorageFlags::map_read | glutils::Buffer::StorageFlags::dynamic_storage,
                              nullptr);
    buffer->write(0, buffer_size, candidates.data());
    buffer->bindRange(glutils::Buffer::IndexedTarget::shader_storage, ReductionKernel::default_ssb_binding, 0, buffer_size);

    kernel(num_work_groups); // <<< dispatch compute
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

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