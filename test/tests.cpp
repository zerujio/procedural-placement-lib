#include "placement/placement.hpp"
#include "placement/placement_pipeline.hpp"
#include "../src/disk_distribution_generator.hpp"

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

class TextureLoader
{
public:
    ~TextureLoader()
    {
        clear();
    }

    GLuint load(const char* filename)
    {
        const GLuint new_tex = s_loadTexture(filename);
        m_loaded_textures[filename] = new_tex;
        return new_tex;
    }

    GLuint load(const std::string& filename)
    {
        return load(filename.c_str());
    }

    GLuint get(const char* filename) const
    {
        const auto it = m_loaded_textures.find(filename);
        if (it == m_loaded_textures.end())
            throw std::runtime_error("no loaded texture with given filename");
        return it->second;
    }

    GLuint get(const std::string& filename) const
    {
        return get(filename.c_str());
    }

    GLuint operator[](const std::string& filename)
    {
        return operator[](filename.c_str());
    }

    GLuint operator[] (const char* filename)
    {
        const auto it = m_loaded_textures.find(filename);
        if (it == m_loaded_textures.end())
            return load(filename);
        return it->second;
    }

    void unload(const char* filename)
    {
        const auto it = m_loaded_textures.find(filename);
        if (it != m_loaded_textures.end())
        {
            gl.DeleteTextures(1, &it->second);
            m_loaded_textures.erase(it);
        }
    }

    void unload(const std::string& filename){unload(filename.c_str());}

    void clear()
    {
        if (m_loaded_textures.empty())
            return;

        std::vector<GLuint> names;
        names.reserve(m_loaded_textures.size());
        for (const auto& pair : m_loaded_textures)
            names.emplace_back(pair.second);
        m_loaded_textures.clear();
        gl.DeleteTextures(names.size(), names.data());
    }

private:
    std::map<std::string, GLuint> m_loaded_textures;

    static GLuint s_loadTexture(const char* filename)
    {
        GLuint texture;
        glm::ivec2 texture_size;
        int channels;
        std::unique_ptr<stbi_uc[]> texture_data {stbi_load(filename, &texture_size.x, &texture_size.y, &channels, 0)};

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
};

TextureLoader s_texture_loader;

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

        gl.DebugMessageCallback(s_glDebugCallback, nullptr);
        gl.Enable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }

    void testRunEnded(const Catch::TestRunStats&) override
    {
        s_texture_loader.clear();
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    void sectionEnded(const Catch::SectionStats&) override
    {
        gl.Finish();
    }

private:

    GLFWwindow* m_window {nullptr};

    static void s_glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                                  const GLchar* message, const void* user_ptr)
    {
        if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
            return;

        UNSCOPED_INFO("[GL DEBUG MESSAGE " << id << "] " << message);
    }
};

CATCH_REGISTER_LISTENER(ContextInitializer)

template<typename Vec>
bool vecOrder(const Vec& l, const Vec& r)
{
    if (l.x < r.x)
        return true;
    else if (r.x < l.x)
        return false;

    if constexpr (Vec::length() > 1)
    {
        if (l.y < r.y)
            return true;
        else if (r.y < l.y)
            return false;
    }

    if constexpr (Vec::length() > 2)
    {
        if (l.z < r.z)
            return true;
        else if (r.z < l.z)
            return false;
    }

    if constexpr (Vec::length() > 3)
    {
        if (l.w < r.w)
            return true;
        else if (r.w < l.w)
            return false;
    }

    return false;
};

TEST_CASE("PlacementPipeline", "[pipeline]")
{
    placement::PlacementPipeline pipeline;

    const glm::vec3 world_scale = {10.0f, 1.0f, 10.0f};
    pipeline.setWorldScale(world_scale);

    const auto black_texture = s_texture_loader["assets/black.png"];
    pipeline.setHeightTexture(black_texture);

    const auto white_texture = s_texture_loader["assets/white.png"];
    pipeline.setDensityTexture(white_texture);

    SECTION("Placement with < 0 area should return an empty vector")
    {
        pipeline.computePlacement(1.0f, {0.0f, 0.0f}, {-1.0f, -1.0f});
        auto points = pipeline.copyResultsToCPU();
        CHECK(points.empty());

        pipeline.computePlacement(1.0f, {0.0f, 0.0f}, {10.0f, -1.0f});
        points = pipeline.copyResultsToCPU();
        CHECK(points.empty());

        pipeline.computePlacement(1.0f, {0.0f, 0.0f}, {-1.0f, 10.0f});
        points = pipeline.copyResultsToCPU();
        CHECK(points.empty());
    }

    SECTION("Placement with one work group")
    {
        const glm::vec2 scale = GenerationKernel::s_work_group_scale * glm::vec2(GenerationKernel::work_group_size);
        pipeline.setWorldScale({scale.x, 1.0f, scale.y});
        pipeline.computePlacement(1.0f, {0.0f, 0.0f}, scale);
        const auto points = pipeline.copyResultsToCPU();

        CAPTURE(scale);
        CHECK(points.size() == GenerationKernel::work_group_size.x * GenerationKernel::work_group_size.y);
    }

    SECTION("Determinism (simple)")
    {
        pipeline.setWorldScale({1.0f, 1.0f, 1.0f});

        pipeline.computePlacement(0.01f, glm::vec2(0.f), glm::vec2(1.f));
        auto positions0 = pipeline.copyResultsToCPU();

        pipeline.computePlacement(0.01f, glm::vec2(0.f), glm::vec2(1.f));
        auto positions1 = pipeline.copyResultsToCPU();

        CHECK(!positions0.empty());
        CHECK(!positions1.empty());

        {
            CAPTURE(positions0, positions1);
            REQUIRE(positions0.size() == positions1.size());
        }

        std::sort(positions0.begin(), positions0.end(), vecOrder<glm::vec3>);
        std::sort(positions1.begin(), positions1.end(), vecOrder<glm::vec3>);

        std::vector<glm::vec3> diff;
        diff.resize(positions0.size());

        const auto diff_end = std::set_symmetric_difference(positions0.begin(), positions0.end(),
                                                            positions1.begin(), positions1.end(),
                                                            diff.begin(), vecOrder<glm::vec3>);
        diff.erase(diff_end, diff.end());
        CAPTURE(diff);
        CHECK(diff.empty());
    }

    const float footprint = GENERATE(take(3, random(0.01f, 0.1f)));
    INFO("footprint = " << footprint);

    const float boundary_offset_x = GENERATE(take(3, random(0.f, 0.4f)));
    const float boundary_offset_y = GENERATE(take(3, random(0.f, 0.4f)));
    const glm::vec2 lower_bound(boundary_offset_x, boundary_offset_y);

    INFO("lower_bound = " << lower_bound);

    const float boundary_size_x = GENERATE(take(3, random(0.6f, 1.0f)));
    const float boundary_size_y = GENERATE(take(3, random(0.6f, 1.0f)));
    const glm::vec2 upper_bound = lower_bound + glm::vec2(boundary_size_x, boundary_size_y);

    INFO("upper_bound = " << upper_bound);

    SECTION("Determinism")
    {
        auto computePlacement = [&]()
        {
            pipeline.computePlacement(footprint, lower_bound, upper_bound);
            auto results = pipeline.copyResultsToCPU();
            std::sort(results.begin(), results.end(), vecOrder<glm::vec3>);
            return results;
        };

        const auto result0 = computePlacement();
        CAPTURE(result0);
        CHECK(!result0.empty());

        const auto result1 = computePlacement();
        CAPTURE(result1);
        CHECK(!result1.empty());

        auto computeDiff = [](const std::vector<glm::vec3>& l, const std::vector<glm::vec3>& r)
        {
            std::vector<glm::vec3> diff;
            diff.resize(std::max(l.size(), r.size()));
            const auto diff_end = std::set_symmetric_difference(l.begin(), l.end(), r.begin(), r.end(),
                                                                diff.begin(), vecOrder<glm::vec3>);
            diff.erase(diff_end, diff.end());
            return diff;
        };

        const auto diff01 = computeDiff(result0, result1);
        CAPTURE(diff01);
        CHECK(diff01.empty());

        const auto result2 = computePlacement();
        CAPTURE(result2);
        CHECK(!result2.empty());

        const auto diff02 = computeDiff(result0, result2);
        CAPTURE(diff02);
        CHECK(diff02.empty());
    }

    SECTION("Boundary and separation")
    {
        pipeline.computePlacement(footprint, lower_bound, upper_bound);
        const auto points = pipeline.copyResultsToCPU();

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

    SECTION("CPU/GPU read")
    {
        pipeline.computePlacement(footprint, lower_bound, upper_bound);

        REQUIRE(pipeline.getResultsSize());

        std::vector<glm::vec3> gpu_results;
        gpu_results.reserve(pipeline.getResultsSize());

        using namespace GL;
        Buffer buffer;
        const auto buffer_size = static_cast<GLsizeiptr>(pipeline.getResultsSize() * sizeof(glm::vec4));
        buffer.allocateImmutable(buffer_size, BufferHandle::StorageFlags::map_read);

        pipeline.copyResultsToGPUBuffer(buffer.getName(), 0);

        {
            auto mapped_ptr = static_cast<const glm::vec4*>(buffer.map(BufferHandle::AccessMode::read_only));
            REQUIRE(mapped_ptr);
            for (auto ptr = mapped_ptr; ptr - mapped_ptr < pipeline.getResultsSize(); ptr++)
                gpu_results.emplace_back(*ptr);
            buffer.unmap();
        }

        const auto cpu_results = pipeline.copyResultsToCPU();
        REQUIRE(cpu_results.size() == pipeline.getResultsSize());

        CHECK(gpu_results == cpu_results);
    }
}

TEMPLATE_TEST_CASE("Common PlacementPipelineKernel operations", "[kernel][pipeline]", GenerationKernel, IndexAssignmentKernel, IndexedCopyKernel)
{
    TestType kernel;

    SECTION("Binding point operations")
    {
        const GLuint binding_index = GENERATE(range(1, 8));

        kernel.setCandidateBufferBindingIndex(binding_index);
        CHECK(kernel.getCandidateBufferBindingIndex() == binding_index);
    }
}

TEST_CASE("GenerationKernel", "[generation][kernel]")
{
    GenerationKernel kernel;

    SECTION("set texture unit")
    {
        for (unsigned int i = 0; i < 8; i++)
        {
            kernel.setDensityTextureUnit(i);
            CHECK(kernel.getDensitytextureUnit() == i);

            kernel.setHeightTextureUnit(i);
            CHECK(kernel.getHeightTextureUnit() == i);
        }
    }

    SECTION("correctness")
    {
        GenerationKernel::PositionStencilMatrix position_stencil;
        for (auto i = 0u; i < position_stencil.size(); i++)
            for (auto j = 0u; j < position_stencil.front().size(); j++)
                position_stencil[i][j] = glm::vec2(i, j) * GenerationKernel::s_work_group_scale;

        kernel.setPositionStencil(position_stencil);

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

        const auto white_texture = s_texture_loader["assets/white.png"];
        const auto black_texture = s_texture_loader["assets/black.png"];

        kernel.setHeightTextureUnit(0);
        gl.BindTextureUnit(kernel.getHeightTextureUnit(), black_texture);

        kernel.setDensityTextureUnit(1);
        gl.BindTextureUnit(kernel.getDensitytextureUnit(), white_texture);

        const auto candidate_count = kernel.setArgs(world_scale, footprint, lower_bound, upper_bound);

        REQUIRE(candidate_count > 0);

        GL::Buffer buffer;
        const auto buffer_size = GenerationKernel::calculateCandidateBufferSize(candidate_count);
        buffer.allocateImmutable(buffer_size, GL::BufferHandle::StorageFlags::map_read, nullptr);

        buffer.bindBase(GL::BufferHandle::IndexedTarget::shader_storage, kernel.getCandidateBufferBindingIndex());

        kernel.dispatchCompute(); // << execute compute kernel

        // read results
        struct Candidate
        {
            glm::vec3 position;
            unsigned int valid;
        };
        std::vector<Candidate> candidates;
        candidates.resize(candidate_count);
        buffer.read(0, buffer_size, candidates.data());

        SECTION("boundaries")
        {
            for (int i = 0; i < candidate_count; i++)
            {
                const glm::vec3 candidate_position = candidates[i].position;
                const GLuint candidate_index = candidates[i].valid;
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
                CAPTURE(candidates[i].position);
                for (int j = 0; j < i; j++)
                {
                    CAPTURE(candidates[j].position);
                    INFO("j = " << j);
                    CHECK(glm::length(candidates[i].position - candidates[j].position) >= Approx(2 * footprint));
                }
            }
        }

        SECTION("determinism")
        {
            auto candidates_duplicate = candidates;

            kernel.dispatchCompute();

            buffer.read(0, buffer_size, candidates_duplicate.data());

            std::vector<glm::vec3> positions;
            std::vector<unsigned int> validity;

            positions.reserve(candidate_count);
            validity.reserve(candidate_count);

            for (auto& c : candidates)
            {
                positions.emplace_back(c.position);
                validity.emplace_back(c.valid);
            }

            std::vector<glm::vec3> positions_dup;
            std::vector<unsigned int> validity_dup;

            positions_dup.reserve(candidate_count);
            validity_dup.reserve(candidate_count);

            for (auto& c : candidates_duplicate)
            {
                positions_dup.emplace_back(c.position);
                validity_dup.emplace_back(c.valid);
            }

            CHECK(positions == positions_dup);
            CHECK(validity == validity_dup);
        }
    }
}

TEST_CASE("IndexAssignmentKernel", "[reduction][kernel]")
{
    using Indices = std::vector<unsigned int>;
    auto indices = GENERATE(Indices{0}, Indices{1},
                            Indices{0, 0}, Indices{0, 1}, Indices{1, 0}, Indices{1, 1},
                            take(6, chunk(10, random(0u, 1u))),
                            take(5, chunk(20, random(0u, 1u))),
                            take(3, chunk(64, random(0u, 1u))),
                            take(3, chunk(333, random(0u, 1u))),
                            take(3, chunk(1024, random(0u, 1u))),
                            take(3, chunk(15000, random(0u, 1u))));

    struct Candidate
    {
        glm::vec3 position;
        unsigned int valid;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(indices.size());
    for (auto i : indices)
        candidates.emplace_back(Candidate{glm::vec3(0.0f), i});

    const unsigned int expected_sum = std::count(indices.cbegin(), indices.cend(), 1u);

    using namespace GL;

    Buffer buffer;
    const BufferHandle::Range candidate_range {0, IndexAssignmentKernel::calculateCandidateBufferSize(candidates.size())};
    const BufferHandle::Range index_range {candidate_range.size,
                                           IndexAssignmentKernel::calculateIndexBufferSize(candidates.size())};
    const BufferHandle::Range index_sum_range {index_range.offset, sizeof(GLuint)};
    const BufferHandle::Range index_array_range {index_range.offset + index_sum_range.size,
                                           index_range.size - index_sum_range.size};
    const GLsizeiptr buffer_size = candidate_range.size + index_sum_range.size + index_array_range.size;

    buffer.allocateImmutable(buffer_size, BufferHandle::StorageFlags::dynamic_storage);

    GLuint computed_sum = 0;
    buffer.write(index_sum_range, &computed_sum);      // initialize sum
    buffer.write(candidate_range, candidates.data());  // initialize candidates

    IndexAssignmentKernel kernel;
    kernel.setCandidateBufferBindingIndex(0);
    kernel.setIndexBufferBindingIndex(1);

    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, kernel.getIndexBufferBindingIndex(), index_range);
    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, kernel.getCandidateBufferBindingIndex(), candidate_range);

    kernel.dispatchCompute(candidates.size());

    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    buffer.read(index_sum_range, &computed_sum);

    std::vector<int> computed_indices;
    computed_indices.resize(indices.size());
    buffer.read(index_array_range, computed_indices.data());

    SECTION("correctness")
    {
        CAPTURE(indices);
        CAPTURE(computed_indices);

        CHECK(computed_sum == expected_sum);

        const auto expected_invalid = indices.size() - expected_sum;

        std::map<int, std::size_t> count;

        for (auto i : computed_indices)
            count[i]++;

        CHECK(count[-1u] == expected_invalid);

        std::vector<std::pair<int, std::size_t>> non_unique;
        for (const auto& p : count)
            if (p.first != -1u && p.second > 1)
                non_unique.emplace_back(p);

        CAPTURE(non_unique);
        CHECK(non_unique.empty());
    }

    SECTION("determinism")
    {
        GLuint second_computed_sum = 0;
        buffer.write(index_sum_range, &second_computed_sum);
        kernel.dispatchCompute(candidates.size());

        std::set<int> first_computed_set;
        for (int i : computed_indices)
            if (i != -1)
                first_computed_set.insert(i);

        gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        buffer.read(index_sum_range, &second_computed_sum);

        CAPTURE(indices);
        CHECK(computed_sum == second_computed_sum);

        std::vector<int> second_computed_indices;
        second_computed_indices.resize(indices.size());
        buffer.read(index_array_range, second_computed_indices.data());

        std::set<int> second_computed_set;
        for (int i : second_computed_indices)
            if (i != -1)
                second_computed_set.insert(i);

        CHECK(first_computed_set == second_computed_set);
    }
}

TEST_CASE("IndexedCopyKernel", "[reduction][kernel]")
{
    using Indices = std::vector<unsigned int>;
    auto validity_indices = GENERATE(take(6, chunk(10, random(0u, 1u))),
                                     take(5, chunk(20, random(0u, 1u))),
                                     take(3, chunk(64, random(0u, 1u))),
                                     take(3, chunk(333, random(0u, 1u))),
                                     take(3, chunk(1024, random(0u, 1u))),
                                     take(3, chunk(15000, random(0u, 1u))));

    std::vector<glm::vec4> candidates;
    candidates.reserve(validity_indices.size());

    std::vector<glm::vec3> valid_positions;
    valid_positions.reserve(validity_indices.size());

    std::vector<unsigned int> copy_indices;
    copy_indices.reserve(validity_indices.size());

    unsigned int index_sum = 0;
    for (auto valid : validity_indices)
    {
        candidates.emplace_back(glm::vec4(candidates.size()));
        if (valid)
        {
            valid_positions.emplace_back(candidates.back());
            copy_indices.emplace_back(index_sum);
        }
        else
            copy_indices.emplace_back(-1u);
        index_sum += valid;
    }

    CAPTURE(validity_indices);

    using namespace GL;

    Buffer buffer;
    const BufferHandle::Range candidate_range {0, static_cast<GLsizeiptr>(candidates.size() * sizeof(glm::vec4))};
    const BufferHandle::Range position_range {candidate_range.size,
                                              static_cast<GLsizeiptr>(valid_positions.size() * sizeof(glm::vec4))};
    const BufferHandle::Range index_sum_range {position_range.offset + position_range.size, sizeof(unsigned int)};
    const BufferHandle::Range index_array_range {index_sum_range.offset + index_sum_range.size,
                                                 static_cast<GLsizeiptr>(copy_indices.size() * sizeof(GLuint))};
    const BufferHandle::Range index_range {index_sum_range.offset, index_sum_range.size + index_array_range.size};

    buffer.allocateImmutable(candidate_range.size + position_range.size + index_range.size,
                              BufferHandle::StorageFlags::dynamic_storage | BufferHandle::StorageFlags::map_read);

    buffer.write(candidate_range, candidates.data());
    buffer.write(index_sum_range, &index_sum);
    buffer.write(index_array_range, copy_indices.data());

    IndexedCopyKernel kernel;
    kernel.setCandidateBufferBindingIndex(0);
    kernel.setPositionBufferBindingIndex(1);
    kernel.setIndexBufferBindingIndex(2);

    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, kernel.getCandidateBufferBindingIndex(), candidate_range);
    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, kernel.getPositionBufferBindingIndex(), position_range);
    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, kernel.getIndexBufferBindingIndex(), index_range);

    kernel.dispatchCompute(candidates.size());

    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<glm::vec3> copied_positions;
    copied_positions.reserve(valid_positions.size());
    auto mapped_ptr = static_cast<glm::vec4*>(buffer.mapRange(position_range, BufferHandle::AccessFlags::read));

    REQUIRE(mapped_ptr);

    for (std::size_t i = 0; i < valid_positions.size(); i++)
        copied_positions.emplace_back(mapped_ptr[i]);

    buffer.unmap();

    CHECK(valid_positions == copied_positions);
}

TEST_CASE("DiskDistributionGenerator")
{
    const uint seed = GENERATE(take(10, random(0u, -1u)));
    CAPTURE(seed);

    auto checkCollision = [](glm::vec2 p, glm::vec2 q, glm::vec2 bounds, float footprint)
    {
        for (int dx = -1; dx <= 1; dx++)
            for (int dy = -1; dy <= 1; dy++)
            {
                const glm::ivec2 tile_offset {dx, dy};
                const glm::vec2 offset = glm::vec2(tile_offset) * bounds;

                CHECK(glm::distance(p, q + offset) >= Approx(2.0f * footprint));
            }
    };

    SECTION("GenerationKernel usage")
    {
        constexpr auto wg_size = GenerationKernel::work_group_size;
        CAPTURE(wg_size);

        DiskDistributionGenerator generator {1.0f, wg_size * GenerationKernel::s_spacing_factor};
        generator.setSeed(seed);
        generator.setMaxAttempts(100);

        const auto bounds = glm::vec2(wg_size) * GenerationKernel::s_work_group_scale;
        CAPTURE(bounds);

        for (std::size_t i = 0; i < 64; i++)
        {
            CAPTURE(i);
            REQUIRE_NOTHROW(generator.generate());
        }

        const auto& positions = generator.getPositions();

        for (auto p = positions.begin(); p != positions.end(); p++)
        {
            CAPTURE(*p, p - positions.begin());
            CHECK(p->x >= 0.0f);
            CHECK(p->y >= 0.0f);
            CHECK(p->x <= bounds.x);
            CHECK(p->y <= bounds.y);

            for (auto q = positions.begin(); q != p; q++)
            {
                CAPTURE(*q, q - positions.begin());
                if (p != q)
                    checkCollision(*p, *q, bounds, 1.0f);
            }
        }
    }

    SECTION("randomized")
    {
        const unsigned int x_cell_count = GENERATE(take(3, random(10u, 100u)));
        const unsigned int y_cell_count = GENERATE(take(3, random(10u, 100u)));
        const glm::uvec2 grid_size {x_cell_count, y_cell_count};

        const float footprint = GENERATE(take(3, random(0.001f, 1.0f)));

        const glm::vec2 bounds = glm::vec2(x_cell_count, y_cell_count) * 2.0f * footprint / std::sqrt(2.0f);

        CAPTURE(grid_size, bounds);

        SECTION("DiskDistributionGrid::getBounds()")
        {
            DiskDistributionGrid grid {footprint, grid_size};
            CHECK(grid.getBounds() == bounds);
        }

        DiskDistributionGenerator generator (footprint, {x_cell_count, y_cell_count});
        generator.setMaxAttempts(100);

        SECTION("trivial case")
        {
            glm::vec2 pos;
            REQUIRE_NOTHROW(pos = generator.generate());
            CHECK(pos.x <= bounds.x);
            CHECK(pos.x >= 0.0f);
            CHECK(pos.y <= bounds.y);
            CHECK(pos.y >= 0.0f);
        }

        SECTION("minimum distance")
        {
            for (int i = 0; i < int(bounds.x); i++)
                REQUIRE_NOTHROW(generator.generate());

            for (const glm::vec2& p : generator.getPositions())
            {
                CAPTURE(p);
                for (const glm::vec2& q : generator.getPositions())
                {
                    CAPTURE(q);
                    if (p != q)
                        checkCollision(p, q, bounds, footprint);
                }
            }
        }

        SECTION("bounds")
        {
            for (int i = 0; i < int(bounds.x); i++)
            {
                glm::vec2 position;
                REQUIRE_NOTHROW(position = generator.generate());
                CHECK(position.x <= bounds.x);
                CHECK(position.x >= 0.0f);
                CHECK(position.y <= bounds.y);
                CHECK(position.y >= 0.0f);
            }
        }
    }
}
