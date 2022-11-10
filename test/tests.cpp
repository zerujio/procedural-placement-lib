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

    SECTION("Determinism (trivial)")
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
        CHECK(kernel.getPositionShaderStorageBlock().getBindingIndex() == PlacementPipelineKernel::default_position_ssb_binding);
        CHECK(kernel.getIndexShaderStorageBlock().getBindingIndex() == PlacementPipelineKernel::default_index_ssb_binding);
    }

    SECTION("Binding point operations")
    {
        const GLuint binding_index = GENERATE(range(1, 8));

        auto ssb = kernel.getPositionShaderStorageBlock();
        ssb.setBindingIndex(binding_index);
        CHECK(ssb.getBindingIndex() == binding_index);

        ssb = kernel.getIndexShaderStorageBlock();
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

        const auto candidate_count = num_invocations.x * num_invocations.y;

        const auto position_buffer_size = GenerationKernel::getPositionBufferRequiredSize(candidate_count);
        const auto index_buffer_size = GenerationKernel::getIndexBufferRequiredSize(candidate_count);

        glutils::Guard<glutils::Buffer> buffer;
        buffer->allocateImmutable(position_buffer_size + index_buffer_size,
                                  glutils::Buffer::StorageFlags::map_read,
                                  nullptr);

        const glutils::BufferRange position_buffer_range {*buffer, 0, position_buffer_size};
        const glutils::BufferRange index_buffer_range {*buffer, position_buffer_size, index_buffer_size};


        position_buffer_range.bindRange(glutils::Buffer::IndexedTarget::shader_storage,
                                        GenerationKernel::default_position_ssb_binding);
        index_buffer_range.bindRange(glutils::Buffer::IndexedTarget::shader_storage,
                                     GenerationKernel::default_index_ssb_binding);

        kernel.dispatchCompute(num_work_groups); // << execute compute kernel

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // read results
        std::vector<glm::vec4> positions;
        positions.resize(candidate_count);
        position_buffer_range.read(positions.data());

        std::vector<GLuint> indices;
        indices.resize(candidate_count);
        index_buffer_range.read(indices.data());

        SECTION("boundaries")
        {
            for (int i = 0; i < candidate_count; i++)
            {
                const glm::vec3 candidate_position = positions[i];
                const GLuint candidate_index = indices[i];
                INFO("Candidate: position = " << candidate_position << ", index = " << candidate_index);

                const glm::vec2 position2d {candidate_position.x, candidate_position.z};

                if (candidate_index)
                {
                    CHECK(candidate_index == 1);
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
                    CHECK(glm::length(positions[i] - positions[j]) >= Approx(2 * footprint));
                }
            }
        }

        SECTION("determinism")
        {
            std::vector<glm::vec4> positions_duplicate = positions;
            positions_duplicate.resize(candidate_count);

            std::vector<GLuint> indices_duplicate;
            indices_duplicate.resize(candidate_count);

            kernel.dispatchCompute(num_work_groups);

            position_buffer_range.read(positions_duplicate.data());
            index_buffer_range.read(indices_duplicate.data());

            CHECK(positions == positions_duplicate);
            CHECK(indices == indices_duplicate);
        }

        const GLuint textures[] = {white_texture, black_texture};
        gl.DeleteTextures(2, &textures[0]);
    }
}


TEST_CASE("ReductionKernel", "[reduction][kernel][pipeline]")
{
    ReductionKernel kernel;

    auto sequential_reduction = [](std::vector<glm::vec3> position_buffer, std::vector<GLuint> index_buffer)
    {
        const auto index_buffer_copy = index_buffer;

        unsigned int index_acc = 0;
        for (auto& index : index_buffer)
        {
            index_acc += index;
            index = index_acc;
        }

        for (int i = 0; i < position_buffer.size(); i++)
        {
            const bool valid = index_buffer_copy[i];
            if (valid)
                position_buffer[index_buffer[i] - 1] = position_buffer[i];
        }

        return std::make_pair(position_buffer, index_buffer);
    };

    using Indices = std::vector<unsigned int>;
    auto indices = GENERATE(Indices{0}, Indices{1},
                                  Indices{0, 0}, Indices{0, 1}, Indices{1, 0}, Indices{1, 1},
                                  take(6, chunk(10, random(0u, 1u))),
                                  take(5, chunk(20, random(0u, 1u))),
                                  take(3, chunk(64, random(0u, 1u))),
                                  take(3, chunk(333, random(0u, 1u))),
                                  take(1, chunk(1024, random(0u, 1u))));

    std::vector<glm::vec4> vec4_positions;
    while (vec4_positions.size() < indices.size())
        vec4_positions.emplace_back(glm::vec4(static_cast<float>(vec4_positions.size())));

    const auto num_work_groups = ReductionKernel::computeNumWorkGroups(indices.size());

    // insert zero padding
    const auto padded_count = num_work_groups * ReductionKernel::work_group_size;
    vec4_positions.resize(padded_count);
    indices.resize(padded_count);

    glutils::Guard<glutils::Buffer> buffer;
    const glutils::BufferRange position_range {*buffer, 0, ReductionKernel::getPositionBufferRequiredSize(padded_count)};
    const glutils::BufferRange index_range {*buffer, position_range.size, ReductionKernel::getIndexBufferRequiredSize(padded_count)};

    buffer->allocateImmutable(position_range.size + index_range.size, glutils::Buffer::StorageFlags::dynamic_storage, nullptr);

    position_range.write(vec4_positions.data());
    position_range.bindRange(glutils::Buffer::IndexedTarget::shader_storage, ReductionKernel::default_position_ssb_binding);

    index_range.write(indices.data());
    index_range.bindRange(glutils::Buffer::IndexedTarget::shader_storage, ReductionKernel::default_index_ssb_binding);

    kernel(num_work_groups); // <<< dispatch compute

    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::vector<glm::vec3> positions;
    positions.reserve(vec4_positions.size());
    for (const auto& vec4_p : vec4_positions)
        positions.emplace_back(vec4_p);

    const auto expected_values = sequential_reduction(positions, indices);

    // read positions
    position_range.read(vec4_positions.data());
    REQUIRE(positions.size() == vec4_positions.size());
    for (int i = 0; i < vec4_positions.size(); i++)
        positions[i] = vec4_positions[i];

    // read indices
    index_range.read(indices.data());

    INFO("positions.size() = " << positions.size());
    const auto& expected_positions = expected_values.first;
    INFO("expected_positions.size() = " << expected_positions.size());
    CHECK(positions == expected_positions);

    INFO("indices.size() = " << indices.size());
    const auto& expected_indices = expected_values.second;
    INFO("expected_indices.size() = " << expected_indices.size());
    CHECK(indices == expected_indices);
}
