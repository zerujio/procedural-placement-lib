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
#include <map>
#include <execution>
#include <thread>

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

    GLuint load(const char *filename)
    {
        const GLuint new_tex = s_loadTexture(filename);
        m_loaded_textures[filename] = new_tex;
        return new_tex;
    }

    GLuint load(const std::string &filename)
    {
        return load(filename.c_str());
    }

    GLuint get(const char *filename) const
    {
        const auto it = m_loaded_textures.find(filename);
        if (it == m_loaded_textures.end())
            throw std::runtime_error("no loaded texture with given filename");
        return it->second;
    }

    GLuint get(const std::string &filename) const
    {
        return get(filename.c_str());
    }

    GLuint operator[](const std::string &filename)
    {
        return operator[](filename.c_str());
    }

    GLuint operator[](const char *filename)
    {
        const auto it = m_loaded_textures.find(filename);
        if (it == m_loaded_textures.end())
            return load(filename);
        return it->second;
    }

    void unload(const char *filename)
    {
        const auto it = m_loaded_textures.find(filename);
        if (it != m_loaded_textures.end())
        {
            gl.DeleteTextures(1, &it->second);
            m_loaded_textures.erase(it);
        }
    }

    void unload(const std::string &filename)
    { unload(filename.c_str()); }

    void clear()
    {
        if (m_loaded_textures.empty())
            return;

        std::vector<GLuint> names;
        names.reserve(m_loaded_textures.size());
        for (const auto &pair: m_loaded_textures)
            names.emplace_back(pair.second);
        m_loaded_textures.clear();
        gl.DeleteTextures(names.size(), names.data());
    }

private:
    std::map<std::string, GLuint> m_loaded_textures;

    static GLuint s_loadTexture(const char *filename)
    {
        GLuint texture;
        glm::ivec2 texture_size;
        int channels;
        std::unique_ptr<stbi_uc[], void (*)(void *)> texture_data
                {stbi_load(filename, &texture_size.x, &texture_size.y, &channels, 0), stbi_image_free};

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

    void testCaseStarting(const Catch::TestCaseInfo &) override
    {
        if (!glfwInit())
        {
            const char *msg = nullptr;
            glfwGetError(&msg);
            throw std::runtime_error(msg);
        }

        glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);

        m_window = glfwCreateWindow(1, 1, "TEST", nullptr, nullptr);
        if (!m_window)
        {
            const char *msg = nullptr;
            glfwGetError(&msg);
            throw std::runtime_error(msg);
        }
        glfwMakeContextCurrent(m_window);

        if (!gladLoadGLContext(&gl, glfwGetProcAddress) or !placement::loadGLContext(glfwGetProcAddress))
            throw std::runtime_error("OpenGL context loading failed");

        gl.DebugMessageCallback(s_glDebugCallback, nullptr);
        gl.Enable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    }

    void testCaseEnded(const Catch::TestCaseStats &) override
    {
        s_texture_loader.clear();
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

private:

    GLFWwindow *m_window{nullptr};

    static void s_glDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                                  const GLchar *message, const void *user_ptr)
    {
        if (severity == GL_DEBUG_SEVERITY_NOTIFICATION)
            return;

        UNSCOPED_INFO("[GL DEBUG MESSAGE " << id << "] " << message);
    }
};

CATCH_REGISTER_LISTENER(ContextInitializer)

template<typename Vec>
bool vecOrder(const Vec &l, const Vec &r)
{
    if constexpr (Vec::length() == 1)
        return std::make_tuple(l.x) < std::make_tuple(r.x);

    if constexpr (Vec::length() == 2)
        return std::make_tuple(l.x, l.y) < std::make_tuple(r.x, r.y);

    if constexpr (Vec::length() == 3)
        return std::make_tuple(l.x, l.y, l.z) < std::make_tuple(r.x, r.y, r.z);

    if constexpr (Vec::length() == 4)
        return std::make_tuple(l.x, l.y, l.z, l.w) < std::make_tuple(r.x, r.y, r.z, r.w);
};

// begin: Utilities

template<typename T>
struct Difference
{
    std::size_t index;
    std::pair<T, T> elements;

    template<typename... Args>
    Difference(std::size_t index, Args &&... args)
            : index(index), elements(std::forward<Args>(args)...)
    {}
};

template<typename LArray, typename RArray>
[[nodiscard]] auto findDifferences(const LArray &l_array, const RArray &r_array)
{
    if (std::size(l_array) != std::size(r_array))
        throw std::logic_error("attempt to diff arrays of different size");

    std::vector<Difference<decltype(l_array[0])>> diffs;

    for (std::size_t i = 0; i < std::size(l_array); i++)
    {
        const auto &l = l_array[i];
        const auto &r = r_array[i];
        if (!(l == r))
            diffs.emplace_back(i, l, r);
    }

    return diffs;
}


template<typename T>
std::ostream &operator<<(std::ostream &out, const Difference<T> &diff)
{
    return out << '[' << diff.index << "] " << diff.elements.first << "!=" << diff.elements.second;
}

template<typename T>
class EqualToValue
{
public:
    explicit EqualToValue(T value) : m_value(value)
    {}

    [[nodiscard]] bool operator()(const T &other_value)
    { return m_value == other_value; }

private:
    T m_value;
};


bool elementCompare(const placement::Result::Element &l, const placement::Result::Element &r)
{
    return std::make_tuple(l.class_index, l.position.x, l.position.y, l.position.z) <
           std::make_tuple(r.class_index, r.position.x, r.position.y, r.position.z);
};

namespace placement {

bool operator==(const Result::Element &l, const Result::Element &r)
{ return l.position == r.position && l.class_index == r.class_index; }

} // placement

// end: Utilities

TEST_CASE("PlacementPipeline", "[pipeline]")
{
    using Element = placement::Result::Element;

    placement::PlacementPipeline pipeline;

    placement::WorldData world_data{{10.f, 10.f, 1.f}, s_texture_loader["assets/textures/black.png"]};
    placement::LayerData layer_data{1.f, {{s_texture_loader["assets/textures/white.png"]}}};

    SECTION("Placement with zero area should return an empty vector")
    {
        auto result = pipeline.computePlacement(world_data, layer_data, {0.0f, 0.0f}, {-1.0f, -1.0f}).readResult();
        CHECK(result.getNumClasses() == 1);
        CHECK(result.getElementArrayLength() == 0);

        auto points = result.copyAllToHost();
        CHECK(points.empty());

        result = pipeline.computePlacement(world_data, layer_data, {0.0f, 0.0f}, {10.0f, -1.0f}).readResult();
        CHECK(result.getNumClasses() == 1);
        CHECK(result.getElementArrayLength() == 0);

        points = result.copyAllToHost();
        CHECK(points.empty());

        result = pipeline.computePlacement(world_data, layer_data, {0.0f, 0.0f}, {-1.0f, 10.0f}).readResult();
        CHECK(result.getNumClasses() == 1);
        CHECK(result.getElementArrayLength() == 0);

        points = result.copyAllToHost();
        REQUIRE(points.empty());
    }

    SECTION("Determinism (simple)")
    {
        const auto result_0 = pipeline.computePlacement(world_data, layer_data, glm::vec2(0.f), world_data.scale)
                                      .readResult();
        REQUIRE(result_0.getElementArrayLength() > 0);

        const auto result_1 = pipeline.computePlacement(world_data, layer_data, glm::vec2(0.f), world_data.scale)
                                      .readResult();
        REQUIRE(result_1.getElementArrayLength() > 0);

        auto positions_0 = result_0.copyAllToHost();
        auto positions_1 = result_1.copyAllToHost();

        {
            CAPTURE(positions_0, positions_1);
            REQUIRE(positions_0.size() == positions_1.size());
        }

        std::sort(positions_0.begin(), positions_0.end(), elementCompare);
        std::sort(positions_1.begin(), positions_1.end(), elementCompare);

        std::vector<placement::Result::Element> diff;
        diff.resize(positions_0.size());

        const auto diff_end = std::set_symmetric_difference(positions_0.begin(), positions_0.end(),
                                                            positions_1.begin(), positions_1.end(),
                                                            diff.begin(), elementCompare);
        diff.erase(diff_end, diff.end());
        CAPTURE(diff);
        CHECK(diff.empty());
    }

    const float footprint = GENERATE(take(3, random(0.01f, 0.1f)));
    layer_data.footprint = footprint;
    CAPTURE(footprint);

    const float boundary_offset_x = GENERATE(take(3, random(0.f, 0.4f)));
    const float boundary_offset_y = GENERATE(take(3, random(0.f, 0.4f)));
    const glm::vec2 lower_bound(boundary_offset_x, boundary_offset_y);
    CAPTURE(lower_bound);

    const float boundary_size_x = GENERATE(take(3, random(0.6f, 1.0f)));
    const float boundary_size_y = GENERATE(take(3, random(0.6f, 1.0f)));
    const glm::vec2 upper_bound = lower_bound + glm::vec2(boundary_size_x, boundary_size_y);

    CAPTURE(upper_bound);

    SECTION("Determinism")
    {
        auto compute_placement = [&]()
        {
            auto positions = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound)
                                     .readResult()
                                     .copyAllToHost();
            std::sort(positions.begin(), positions.end(), elementCompare);
            return positions;
        };

        const auto result_0 = compute_placement();
        CAPTURE(result_0);
        CHECK(!result_0.empty());

        const auto result_1 = compute_placement();
        CAPTURE(result_1);
        CHECK(!result_1.empty());

        auto compute_diff = [](const std::vector<Element> &l, const std::vector<Element> &r)
        {
            std::vector<Element> diff;
            diff.resize(std::max(l.size(), r.size()));
            const auto diff_end = std::set_symmetric_difference(l.begin(), l.end(), r.begin(), r.end(),
                                                                diff.begin(), elementCompare);
            diff.erase(diff_end, diff.end());
            return diff;
        };

        const auto diff_01 = compute_diff(result_0, result_1);
        CAPTURE(diff_01);
        CHECK(diff_01.empty());

        const auto result_2 = compute_placement();
        CAPTURE(result_2);
        CHECK(!result_2.empty());

        const auto diff_02 = compute_diff(result_0, result_2);
        CAPTURE(diff_02);
        CHECK(diff_02.empty());
    }

    SECTION("Boundary and separation")
    {
        const auto elements = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound)
                                      .readResult()
                                      .copyAllToHost();

        REQUIRE(!elements.empty());

        std::vector<std::pair<glm::vec2, glm::vec2>> collisions;
        std::vector<glm::vec2> out_of_bounds;

        for (int i = 0; i < elements.size(); i++)
        {
            const glm::vec2 element_position_2d = elements[i].position;

            if (glm::any(glm::lessThan(element_position_2d, lower_bound))
                || glm::any(glm::greaterThanEqual(element_position_2d, upper_bound)))
                out_of_bounds.emplace_back(element_position_2d);

            for (int j = 0; j < i; j++)
            {
                const glm::vec2 other_position_2d = elements[j].position;

                if (glm::distance(element_position_2d, other_position_2d) < footprint)
                    collisions.emplace_back(element_position_2d, other_position_2d);
            }
        }

        {
            INFO("Out of bounds positions:");
            CAPTURE(out_of_bounds);
            CHECK(out_of_bounds.empty());
        }

        {
            INFO("Colliding positions");
            CAPTURE(collisions);
            CHECK(collisions.empty());
        }
    }

    SECTION("CPU/GPU read")
    {
        const auto results = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound).readResult();

        REQUIRE(results.getElementArrayLength() > 0);

        std::vector<Element> gpu_results;
        gpu_results.resize(results.getElementArrayLength());

        {
            GL::Buffer buffer;
            const auto buffer_size = static_cast<GLsizeiptr>(results.getElementArrayLength() * sizeof(Element));
            buffer.allocateImmutable(buffer_size, GL::Buffer::StorageFlags::none);

            results.copyAll(buffer);

            buffer.read(0, buffer_size, gpu_results.data());
        }

        const auto cpu_results = results.copyAllToHost();
        CHECK(cpu_results.size() == results.getElementArrayLength());

        CHECK(gpu_results == cpu_results);
    }
}

TEST_CASE("PlacementPipeline (multiclass)", "[pipeline][multiclass]")
{
    using namespace placement;

    constexpr float footprint = 0.01f;

    PlacementPipeline pipeline;
    WorldData world_data{{1.f, 1.f, 1.f}, s_texture_loader["assets/textures/heightmap.png"]};
    const GLuint white_texture = s_texture_loader["assets/textures/white.png"];
    LayerData layer_data{footprint,
                         {{white_texture, .4f}, {white_texture, .3f}, {white_texture, .2f}, {white_texture, .1f}}};

    const std::size_t num_classes = layer_data.densitymaps.size();

    const glm::vec2 lower_bound{0};
    const glm::vec2 upper_bound{1};

    auto results = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound).readResult();

    SECTION("Accessors")
    {
        SECTION("Host")
        {
            const auto all_results = results.copyAllToHost();
            CHECK(results.getElementArrayLength() == all_results.size());

            auto begin_iter = all_results.begin();
            std::vector<Result::Element> all_results_subsection;
            for (std::size_t i = 0; i < num_classes; i++)
            {
                const auto class_size = results.getClassElementCount(i);
                all_results_subsection.insert(all_results_subsection.cend(), begin_iter, begin_iter + class_size);
                begin_iter += class_size;

                const auto class_results = results.copyClassToHost(i);
                CHECK(results.getClassElementCount(i) == class_results.size());

                CHECK(class_results == all_results_subsection);
                all_results_subsection.clear();
            }
        }

        SECTION("Device")
        {
            GL::Buffer buffer;
            const auto buffer_size = results.getElementArrayLength() * sizeof(glm::vec4);
            buffer.allocateImmutable(buffer_size, GL::Buffer::StorageFlags::none);

            results.copyAll(buffer);

            std::vector<Result::Element> all_elements;
            all_elements.resize(results.getElementArrayLength());
            buffer.read(0, buffer_size, all_elements.data());

            const auto expected = results.copyAllToHost();

            CHECK(all_elements == expected);
        }
    }

    using Element = Result::Element;

    SECTION("Boundaries and footprint")
    {
        const auto elements = results.copyAllToHost();
        CAPTURE(elements.size());

        std::vector<Element> out_of_bounds;
        std::vector<std::pair<Element, Element>> collisions;

        for (std::size_t i = 0; i < elements.size(); i++)
        {
            const glm::vec2 position2d = elements[i].position;

            if (glm::any(glm::lessThan(position2d, lower_bound)) ||
                glm::any(glm::greaterThanEqual(position2d, upper_bound)))
                out_of_bounds.emplace_back(elements[i]);

            for (std::size_t j = 0; j < i; j++)
            {
                const glm::vec2 other_position2d = elements[j].position;

                if (glm::distance(position2d, other_position2d) < footprint)
                    collisions.emplace_back(elements[i], elements[j]);
            }
        }

        {
            INFO("Boundary check:");
            CAPTURE(out_of_bounds.size(), out_of_bounds);
            CHECK(out_of_bounds.empty());
        }

        {
            INFO("Footprint check:");
            CAPTURE(collisions.size(), collisions);
            CHECK(collisions.empty());
        }
    }

    SECTION("Determinism")
    {
        const auto sort_result = [](const Result &result)
        {
            auto elements = result.copyAllToHost();
            std::sort(elements.begin(), elements.end(), elementCompare);
            return elements;
        };

        auto results_1 = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound).readResult();
        auto results_2 = pipeline.computePlacement(world_data, layer_data, lower_bound, upper_bound).readResult();

        const auto positions_0 = sort_result(results);
        const auto positions_1 = sort_result(results_1);
        const auto positions_2 = sort_result(results_2);

        {
            const auto diffs_01 = findDifferences(positions_0, positions_1);
            CAPTURE(diffs_01);
            CHECK(diffs_01.empty());
        }

        {
            const auto diffs_02 = findDifferences(positions_0, positions_2);
            CAPTURE(diffs_02);
            CHECK(diffs_02.empty());
        }
    }
}

TEST_CASE("GenerationKernel", "[generation][kernel]")
{
    GenerationKernel kernel;

    constexpr auto wg_size = GenerationKernel::work_group_size;
    constexpr glm::vec2 wg_scale{1.0f};

    glm::vec2 position_stencil[wg_size.x][wg_size.y];
    for (auto i = 0u; i < wg_size.x; i++)
        for (auto j = 0u; j < wg_size.y; j++)
            position_stencil[i][j] = glm::vec2(i, j) * wg_scale;

    constexpr glm::vec3 world_scale{1.0f};

    const auto black_texture = s_texture_loader["assets/textures/black.png"];

    const uint height_texture_unit = 0;
    gl.BindTextureUnit(height_texture_unit, black_texture);

    const auto footprint = GENERATE(take(3, random(0.01f, 0.1f)));
    CAPTURE(footprint);

    const glm::uvec2 wg_count{world_scale / (glm::vec3(wg_scale, 1.0f) * glm::vec3(wg_size))};

    const std::size_t candidate_count = wg_count.x * wg_count.y * wg_size.x * wg_size.y;

    GL::Buffer buffer;
    const GL::Buffer::Range candidate_range{0, GenerationKernel::getCandidateBufferSizeRequirement({wg_count, 1})};
    const GL::Buffer::Range world_uv_range{candidate_range.offset + candidate_range.size,
                                           GenerationKernel::getWorldUVBufferSizeRequirement({wg_count, 1})};
    const GL::Buffer::Range density_range{world_uv_range.offset + world_uv_range.size,
                                          GenerationKernel::getDensityBufferMemoryRequirement({wg_count, 1})};

    buffer.allocateImmutable(candidate_range.size + world_uv_range.size + density_range.size,
                             GL::BufferHandle::StorageFlags::map_read);

    constexpr uint candidate_binding_index = 0;
    constexpr uint world_uv_binding_index = 1;
    constexpr uint density_binding_index = 2;

    buffer.bindRange(GL::Buffer::IndexedTarget::shader_storage, candidate_binding_index, candidate_range);
    buffer.bindRange(GL::Buffer::IndexedTarget::shader_storage, world_uv_binding_index, world_uv_range);
    buffer.bindRange(GL::Buffer::IndexedTarget::shader_storage, density_binding_index, density_range);

    kernel(wg_count, /*work group index offset*/ {0, 0}, footprint, world_scale,
           height_texture_unit, candidate_binding_index, world_uv_binding_index, density_binding_index);
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<Result::Element> candidates;
    candidates.resize(candidate_count);
    buffer.read(candidate_range, candidates.data());

    std::vector<glm::vec2> world_uvs;
    world_uvs.resize(candidate_count);
    buffer.read(world_uv_range, world_uvs.data());

    std::vector<float> densities;
    densities.resize(candidate_count);
    buffer.read(density_range, densities.data());

    SECTION("correctness")
    {
        using Candidate = Result::Element;

        {
            constexpr uint invalid_index = 0xFFffFFff;
            INFO("candidate class index must be initialized to " << invalid_index);
            CHECK(std::all_of(candidates.begin(), candidates.end(),
                              [=](const Candidate &c)
                              { return c.class_index == invalid_index; }));
        }

        {
            INFO("density must be initialized to 0.0f");
            CHECK(std::all_of(densities.begin(), densities.end(), EqualToValue(0.0f)));
        }

        {
            INFO("uv * world_scale must equal position.xy");
            CHECK(std::equal(candidates.begin(), candidates.end(), world_uvs.begin(),
                             [=](const Candidate &c, glm::vec2 uv)
                             {
                                 return c.position.x == Approx(uv.x * world_scale.x) &&
                                        c.position.y == Approx(uv.y * world_scale.y);
                             }));
        }

        {
            INFO("minimum separation equal to footprint");
            for (auto iter_i = candidates.begin(); iter_i != candidates.end(); iter_i++)
            {
                const glm::vec2 pos_i = iter_i->position;
                for (auto iter_j = candidates.begin(); iter_j != iter_i; iter_j++)
                {
                    const glm::vec2 pos_j = iter_j->position;
                    REQUIRE(glm::distance(pos_i, pos_j) > footprint);
                }
            }
        }
    }

    SECTION("determinism")
    {
        kernel(wg_count, /*work group index offest*/ {0, 0}, footprint, world_scale,
               height_texture_unit, candidate_binding_index, world_uv_binding_index, density_binding_index);
        gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        auto candidates_duplicate = candidates;
        auto world_uvs_duplicate = world_uvs;
        auto densities_duplicate = densities;

        buffer.read(candidate_range, candidates_duplicate.data());
        buffer.read(world_uv_range, world_uvs_duplicate.data());
        buffer.read(density_range, densities_duplicate.data());

        CHECK(candidates == candidates_duplicate);
        CHECK(world_uvs == world_uvs_duplicate);
        CHECK(densities == densities_duplicate);
    }
}

TEST_CASE("EvaluationKernel", "[evaluation][kernel]")
{
    const uint wg_count_x = GENERATE(take(3, random(1, 4)));
    const uint wg_count_y = GENERATE(take(3, random(1, 4)));
    const glm::uvec2 wg_count = {wg_count_x, wg_count_y};
    CAPTURE(wg_count);

    const glm::vec2 world_boundaries{10.f};

    const float lower_bound_x = GENERATE(take(1, random(0.f, 10.f)));
    const float lower_bound_y = GENERATE(take(1, random(0.f, 10.f)));
    const float placement_area_x = GENERATE(take(1, random(0.f, 10.f)));
    const float placement_area_y = GENERATE(take(1, random(0.f, 10.f)));

    const glm::vec2 lower_bound{lower_bound_x, lower_bound_y};
    const glm::vec2 upper_bound = lower_bound + glm::vec2{placement_area_x, placement_area_y};

    const GLsizeiptr candidate_count_x = wg_count_x * EvaluationKernel::work_group_size.x;
    const GLsizeiptr candidate_count_y = wg_count_y * EvaluationKernel::work_group_size.y;
    const GLsizeiptr candidate_count = candidate_count_x * candidate_count_y;

    EvaluationKernel kernel;

    std::vector<Result::Element> candidates;
    std::vector<Result::Element> expected_result;
    std::vector<glm::vec2> world_uvs;
    std::vector<float> densities;

    candidates.reserve(candidate_count);
    expected_result.reserve(candidate_count);
    world_uvs.reserve(candidate_count);
    densities.reserve(candidate_count);

    constexpr uint invalid_index = 0xFFffFFff;

    for (std::size_t i = 0; i < candidate_count_x; i++)
    {
        const float world_u = static_cast<float>(i) / static_cast<float>(candidate_count_x);
        const float position_x = world_u * world_boundaries.x;

        for (std::size_t j = 0; j < candidate_count_y; j++)
        {
            const float world_v = static_cast<float>(j) / static_cast<float>(candidate_count_y);
            const float position_y = world_v * world_boundaries.y;

            Result::Element new_candidate{glm::vec3(position_x, position_y, 0.f), invalid_index};

            candidates.push_back(new_candidate);
            world_uvs.emplace_back(world_u, world_v);
            densities.emplace_back(0.0f);

            const bool inside_bounds = glm::all(glm::greaterThanEqual(glm::vec2(new_candidate.position), lower_bound))
                                       && glm::all(glm::lessThan(glm::vec2(new_candidate.position), upper_bound));

            if (inside_bounds)
                new_candidate.class_index = 0;

            expected_result.emplace_back(new_candidate);
        }
    }

    constexpr GLsizeiptr candidate_size = sizeof(Result::Element);
    constexpr GLsizeiptr world_uv_size = sizeof(glm::vec2);
    constexpr GLsizeiptr density_size = sizeof(float);

    const GL::Buffer buffer;
    const GL::Buffer::Range candidate_range{0, candidate_size * candidate_count};
    const GL::Buffer::Range world_uv_range{candidate_range.size, world_uv_size * candidate_count};
    const GL::Buffer::Range density_range{candidate_range.size + world_uv_range.size, density_size * candidate_count};

    buffer.allocateImmutable(candidate_range.size + world_uv_range.size + density_range.size,
                             GL::Buffer::StorageFlags::dynamic_storage | GL::Buffer::StorageFlags::map_read);

    buffer.write(candidate_range, candidates.data());
    buffer.write(world_uv_range, world_uvs.data());
    buffer.write(density_range, densities.data());

    constexpr uint candidate_binding_index = 0;
    constexpr uint world_uv_binding_index = 1;
    constexpr uint density_binding_index = 2;

    buffer.bindRange(GL::Buffer::IndexedTarget::shader_storage, candidate_binding_index, candidate_range);
    buffer.bindRange(GL::Buffer::IndexedTarget::shader_storage, world_uv_binding_index, world_uv_range);
    buffer.bindRange(GL::Buffer::IndexedTarget::shader_storage, density_binding_index, density_range);

    constexpr uint density_tex_unit = 0;
    const GLuint density_texture = s_texture_loader["assets/textures/white.png"];
    gl.BindTextureUnit(density_tex_unit, density_texture);

    kernel(wg_count, {0, 0}, /*class index*/ 0, lower_bound, upper_bound, density_tex_unit, DensityMap(),
           candidate_binding_index, world_uv_binding_index, density_binding_index);
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    {
        INFO("Candidates");

        auto mapped_ptr = static_cast<const Result::Element *>(buffer.mapRange(candidate_range,
                                                                               GL::Buffer::AccessFlags::read));
        std::vector<Result::Element> computed_result{mapped_ptr, mapped_ptr + candidate_count};
        buffer.unmap();

        std::vector<Difference<Result::Element>> differences;

        for (std::size_t i = 0; i < candidate_count; i++)
        {
            if (expected_result[i].position == computed_result[i].position
                && expected_result[i].class_index == computed_result[i].class_index)
                continue;

            differences.emplace_back(i, expected_result[i], computed_result[i]);
        }

        CAPTURE(lower_bound, upper_bound, differences);
        CHECK(differences.empty());
    }

    {
        INFO("Densities");

        auto mapped_ptr = static_cast<const float *>(buffer.mapRange(density_range, GL::Buffer::AccessFlags::read));
        std::vector<float> computed_densities{mapped_ptr, mapped_ptr + candidate_count};
        buffer.unmap();

        std::vector<Difference<float>> differences;

        for (std::size_t i = 0; i < candidate_count; i++)
        {
            if (computed_densities[i] != 1.0f)
                differences.emplace_back(i, computed_densities[i], 1.0f);
        }

        CAPTURE(differences);
        CHECK(differences.empty());
    }
}

/**
 * This test dispatches the indexation kernel with a few hand-picked input arrays and multiple randomly generated ones,
 * checking that the retrieved indices have the expected values.
 */
TEST_CASE("IndexationKernel", "[indexation][kernel]")
{
    using Indices = std::vector<int>;
    const auto class_indices = GENERATE(
            Indices{-1}, Indices{0},
            Indices{-1, -1}, Indices{-1, 0}, Indices{0, -1}, Indices{0, 0},
            take(3, chunk(10, random(-1, 1))),
            take(3, chunk(64, random(-1, 3))),
            take(3, chunk(333, random(-1, 5))),
            take(3, chunk(1024, random(-1, 7))),
            take(3, chunk(15000, random(-1, 10))));

    using Candidate = Result::Element;

    std::vector<Candidate> candidates;

    constexpr int invalid_index = -1;

    std::vector<unsigned int> expected_class_counts;

    for (auto &class_index: class_indices)
    {
        candidates.emplace_back(Candidate{glm::vec3(0.0f), static_cast<uint>(class_index)});

        if (class_index == invalid_index)
            continue;

        if (class_index >= expected_class_counts.size())
            expected_class_counts.resize(class_index + 1);

        expected_class_counts[class_index]++;
    }

    const uint expected_total_count = std::accumulate(expected_class_counts.begin(), expected_class_counts.end(), 0u);

    CAPTURE(class_indices);

    uint computed_total_count = 0;
    std::vector<uint> computed_class_counts;
    computed_class_counts.resize(expected_class_counts.size());

    const GLsizeiptr class_count = expected_class_counts.size();
    const GLsizeiptr candidate_count = candidates.size();

    CAPTURE(class_count, candidate_count);

    constexpr GLsizeiptr candidate_size = sizeof(Candidate);
    constexpr GLsizeiptr uint_size = sizeof(GLuint);

    GL::Buffer buffer;
    const GL::BufferHandle::Range candidate_range{0, candidate_count * candidate_size};
    const GL::BufferHandle::Range index_range{candidate_range.size, candidate_count * uint_size};
    const GL::BufferHandle::Range count_range{index_range.offset + index_range.size, class_count * uint_size};

    buffer.allocateImmutable(candidate_range.size + index_range.size + count_range.size,
                             GL::BufferHandle::StorageFlags::dynamic_storage);

    buffer.write(count_range, computed_class_counts.data()); // initialize counts
    buffer.write(candidate_range, candidates.data()); // initialize candidates

    constexpr uint candidate_binding_index = 0;
    constexpr uint index_binding_index = 1;
    constexpr uint count_binding_index = 2;

    IndexationKernel kernel;

    buffer.bindRange(GL::BufferHandle::IndexedTarget::shader_storage, candidate_binding_index, candidate_range);
    buffer.bindRange(GL::BufferHandle::IndexedTarget::shader_storage, index_binding_index, index_range);
    buffer.bindRange(GL::BufferHandle::IndexedTarget::shader_storage, count_binding_index, count_range);

    const auto wg_count = IndexationKernel::calculateNumWorkGroups(candidate_count);

    kernel(wg_count, candidate_binding_index, count_binding_index, index_binding_index);
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    buffer.read(count_range, computed_class_counts.data());

    CHECK(computed_class_counts == expected_class_counts);

    computed_total_count = std::accumulate(computed_class_counts.begin(), computed_class_counts.end(), 0u);
    CHECK(computed_total_count == expected_total_count);

    std::vector<int> computed_indices(class_indices.size());
    buffer.read(index_range, computed_indices.data());

    std::vector<std::vector<uint>> computed_indices_by_class(class_count);

    for (std::size_t i = 0; i < class_indices.size(); i++)
    {
        const uint class_index = candidates[i].class_index;

        if (class_index == invalid_index)
            continue;

        computed_indices_by_class[class_index].emplace_back(computed_indices[i]);
    }

    CAPTURE(computed_indices);

    std::vector<uint> index_usage_count;
    std::vector<uint> out_of_range_indices;
    std::vector<uint> duplicate_indices;
    std::vector<uint> missing_indices;

    for (uint class_index = 0; class_index < class_count; class_index++)
    {
        CAPTURE(class_index, computed_indices_by_class[class_index]);

        const uint class_element_count = expected_class_counts[class_index];
        index_usage_count.resize(class_element_count);

        for (auto copy_index: computed_indices_by_class[class_index])
        {
            if (copy_index >= class_element_count)
                out_of_range_indices.emplace_back(copy_index);
            else if (index_usage_count[copy_index]++ > 0)
                duplicate_indices.emplace_back(copy_index);
        }

        {
            CAPTURE(out_of_range_indices);
            CHECK(out_of_range_indices.empty());
            out_of_range_indices.clear();
        }

        {
            CAPTURE(duplicate_indices);
            CHECK(duplicate_indices.empty());
            duplicate_indices.clear();
        }

        {
            for (uint index = 0; index < index_usage_count.size(); index++)
                if (index_usage_count[index] == 0)
                    missing_indices.emplace_back(index);

            CAPTURE(missing_indices);
            CHECK(missing_indices.empty());
            missing_indices.clear();
        }

        index_usage_count.clear();
    }
}

TEST_CASE("CopyKernel", "[copy][kernel]")
{
    using IndexVector = std::vector<unsigned int>;
    std::vector<unsigned int> indices{GENERATE(IndexVector{0u}, IndexVector{1u},
                                               IndexVector{0u, 0u}, IndexVector{0u, 1u},
                                               IndexVector{1u, 0u}, IndexVector{1u, 1u},
                                               take(3, chunk(10, random(0u, 2u))),
                                               take(3, chunk(64, random(0u, 3u))),
                                               take(3, chunk(333, random(0u, 5u))),
                                               take(3, chunk(1024, random(0u, 7u))),
                                               take(3, chunk(15000, random(0u, 10u))))};
    constexpr uint invalid_index = -1u;

    using Candidate = Result::Element;

    std::vector<Candidate> candidates;
    candidates.reserve(indices.size());

    std::vector<unsigned int> copy_indices;
    copy_indices.reserve(indices.size());

    std::vector<unsigned int> element_counts;

    for (auto index: indices)
    {
        if (index > element_counts.size())
            element_counts.resize(index);

        const uint class_index = index - 1;

        Candidate candidate{glm::vec3(candidates.size()), class_index};
        candidates.emplace_back(candidate);
        copy_indices.emplace_back(class_index != invalid_index ? element_counts[class_index]++ : invalid_index);
    }

    std::vector<Candidate> expected_results;
    expected_results.reserve(indices.size());

    for (uint element_class = 0; element_class < element_counts.size(); element_class++)
    {
        for (const Candidate &candidate: candidates)
            if (candidate.class_index == element_class)
                expected_results.emplace_back(candidate);
    }

    CAPTURE(candidates, element_counts, expected_results, copy_indices);

    using namespace GL;

    constexpr GLsizeiptr candidate_size = sizeof(Candidate);
    constexpr GLsizeiptr uint_size = sizeof(uint);
    const GLsizeiptr candidate_count = candidates.size();
    const GLsizeiptr class_count = element_counts.size();

    Buffer buffer;
    const BufferHandle::Range candidate_range{0, candidate_count * candidate_size};
    const BufferHandle::Range output_range{candidate_range.size, candidate_range.size};
    const BufferHandle::Range index_range{output_range.offset + output_range.size, candidate_count * uint_size};
    const BufferHandle::Range count_range{index_range.offset + index_range.size, class_count * uint_size};

    buffer.allocateImmutable(candidate_range.size + output_range.size + index_range.size + count_range.size,
                             BufferHandle::StorageFlags::dynamic_storage | Buffer::StorageFlags::map_read);

    buffer.write(candidate_range, candidates.data());
    buffer.write(count_range, element_counts.data());
    buffer.write(index_range, copy_indices.data());

    CopyKernel kernel;

    constexpr uint candidate_buffer_binding = 0;
    constexpr uint output_buffer_binding = 1;
    constexpr uint index_buffer_binding = 2;
    constexpr uint count_buffer_binding = 3;

    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, candidate_buffer_binding, candidate_range);
    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, output_buffer_binding, output_range);
    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, index_buffer_binding, index_range);
    buffer.bindRange(BufferHandle::IndexedTarget::shader_storage, count_buffer_binding, count_range);

    const uint num_work_groups = CopyKernel::calculateNumWorkGroups(candidate_count);
    CAPTURE(candidate_count);

    kernel(num_work_groups,
           candidate_buffer_binding, count_buffer_binding, index_buffer_binding, output_buffer_binding);
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    const uint total_count = std::accumulate(element_counts.begin(), element_counts.end(), 0u);

    auto output_ptr = static_cast<const Candidate *>(buffer.mapRange(output_range, GL::Buffer::AccessFlags::read));

    std::vector<Candidate> results(output_ptr, output_ptr + total_count);

    buffer.unmap();

    CHECK(results == expected_results);
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
                const glm::ivec2 tile_offset{dx, dy};
                const glm::vec2 offset = glm::vec2(tile_offset) * bounds;

                if (glm::distance(p, q + offset) >= footprint)
                    return true;
            }

        return false;
    };

    SECTION("GenerationKernel usage")
    {
        constexpr auto wg_size = GenerationKernel::work_group_size;
        CAPTURE(wg_size);

        const glm::uvec2 grid_size{glm::vec2(wg_size) * 2.5f};
        constexpr float footprint = .5f;

        DiskDistributionGenerator generator{footprint, grid_size};
        generator.setSeed(seed);
        generator.setMaxAttempts(100);

        CAPTURE(generator.getGrid().getBounds());

        for (std::size_t i = 0; i < 64; i++)
        {
            CAPTURE(i);
            REQUIRE_NOTHROW(generator.generate());
        }

        const auto &positions = generator.getPositions();

        for (auto p = positions.begin(); p != positions.end(); p++)
        {
            CAPTURE(*p, p - positions.begin());
            CHECK(p->x >= 0.0f);
            CHECK(p->y >= 0.0f);
            CHECK(p->x <= generator.getGrid().getBounds().x);
            CHECK(p->y <= generator.getGrid().getBounds().y);

            for (auto q = positions.begin(); q != p; q++)
            {
                CAPTURE(*q, q - positions.begin());
                if (p != q)
                    CHECK(checkCollision(*p, *q, generator.getGrid().getBounds(), footprint));
            }
        }
    }

    SECTION("randomized")
    {
        const unsigned int x_cell_count = GENERATE(take(3, random(10u, 100u)));
        const unsigned int y_cell_count = GENERATE(take(3, random(10u, 100u)));
        const glm::uvec2 grid_size{x_cell_count, y_cell_count};

        const float footprint = GENERATE(take(3, random(0.001f, 1.0f)));

        const glm::vec2 bounds = glm::vec2(x_cell_count, y_cell_count) * footprint / std::sqrt(2.0f);

        CAPTURE(grid_size, bounds);

        SECTION("DiskDistributionGrid::getBounds()")
        {
            DiskDistributionGrid grid{footprint, grid_size};
            CHECK(grid.getBounds() == bounds);
        }

        DiskDistributionGenerator generator(footprint, {x_cell_count, y_cell_count});
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

            for (const glm::vec2 &p: generator.getPositions())
            {
                CAPTURE(p);
                for (const glm::vec2 &q: generator.getPositions())
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

TEST_CASE("SSBO alignment")
{
    GL::Buffer buffer;

    auto compile_compute_shader = [](const char *source_code)
    {
        GL::Program program;
        GL::Shader shader{GL::Shader::Type::compute};
        shader.setSource(source_code);
        shader.compile();

        if (shader.getParameter(GL::Shader::Parameter::compile_status) != GL_TRUE)
            throw std::runtime_error(shader.getInfoLog());

        program.attachShader(shader);
        program.link();

        if (program.getParameter(GL::Program::Parameter::link_status) != GL_TRUE)
            throw std::runtime_error(program.getInfoLog());

        program.detachShader(shader);

        return program;
    };

    SECTION("struct {vec3; uint;}")
    {
        struct Candidate
        {
            glm::vec3 position;
            uint index;
        };

        buffer.allocateImmutable(16 * sizeof(Candidate), GL::Buffer::StorageFlags::none);
        buffer.bindBase(GL::Buffer::IndexedTarget::shader_storage, 0);

        GL::Program program = compile_compute_shader(
                "#version 450 core\n"
                "layout(local_size_x = 16) in;"
                "struct Candidate { vec3 position; uint index; };\n"
                "layout(std430, binding=0) buffer TransientBuffer { Candidate[] candidates; };\n"
                "void main() "
                "{"
                "   candidates[gl_GlobalInvocationID.x] = Candidate(vec3(gl_GlobalInvocationID.x),"
                "                                                   gl_GlobalInvocationID.x);"
                "}\n");

        std::vector<Candidate> candidates;
        candidates.resize(16);

        program.use();
        gl.DispatchCompute(1, 1, 1);
        gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        buffer.read(0, candidates.size() * sizeof(Candidate), candidates.data());

        for (uint i = 0; i < candidates.size(); i++)
        {
            CAPTURE(i);
            CHECK(candidates[i].position == glm::vec3(i));
            CHECK(candidates[i].index == i);
        }
    }

    SECTION("vec3")
    {
        constexpr std::size_t num_elements = 16;
        glm::vec4 results[num_elements];

        GL::Program program = compile_compute_shader(
                "#version 450 core\n"
                "layout(local_size_x=16) in;\n"
                "layout(std430, binding=0) buffer TransientBuffer { vec3 positions[]; };\n"
                "void main() { positions[gl_GlobalInvocationID.x] = vec3(gl_GlobalInvocationID.x); }");

        buffer.allocateImmutable(sizeof(results), GL::Buffer::StorageFlags::none);
        buffer.bindBase(GL::Buffer::IndexedTarget::shader_storage, 0);

        program.use();
        gl.DispatchCompute(1, 1, 1);
        gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

        buffer.read(0, sizeof(results), results);

        for (int i = 0; i < num_elements; i++)
            CHECK(glm::vec3(results[i]) == glm::vec3(i));
    }
}

using WorkGroupPattern =
        std::array<std::array<glm::vec2, GenerationKernel::work_group_size.y>, GenerationKernel::work_group_size.x>;

std::pair<glm::vec2, WorkGroupPattern> generateWorkGroupPattern(uint seed)
{
    constexpr auto wg_size = GenerationKernel::work_group_size;

    DiskDistributionGenerator generator{1.0f, wg_size * 2u};
    generator.setSeed(seed);
    generator.setMaxAttempts(100);

    std::array<std::array<glm::vec2, wg_size.y>, wg_size.x> positions;
    for (auto &column: positions)
        for (auto &cell: column)
            cell = generator.generate();

    return {generator.getGrid().getBounds(), positions};
}

struct GrayscaleImage
{
public:
    explicit GrayscaleImage(const char *filename)
            : m_data(stbi_load(filename, &m_size.x, &m_size.y, nullptr, 1), stbi_image_free)
    {
        if (!m_data)
            throw std::runtime_error(stbi_failure_reason());
    }

    [[nodiscard]]
    glm::ivec2 getSize() const
    { return m_size; }

    [[nodiscard]]
    float sample(glm::vec2 tex_coord) const
    {
        const glm::ivec2 pixel_index = glm::clamp(tex_coord, {0, 0}, {1, 1}) * glm::vec2(m_size);
        const stbi_uc value = m_data[pixel_index.y * m_size.x + pixel_index.x];
        return static_cast<float>(value) / static_cast<float>(std::numeric_limits<stbi_uc>::max());
    }

private:
    glm::ivec2 m_size{0};
    std::unique_ptr<stbi_uc[], void (*)(void *)> m_data;
};

template<class ExecutionPolicy>
[[nodiscard]]
std::vector<Result::Element> computePlacement(const ExecutionPolicy &policy,
                                              const WorldData &world_data, const LayerData &layer_data,
                                              glm::vec2 lower_bound, glm::vec2 upper_bound,
                                              glm::vec2 work_group_bounds, WorkGroupPattern work_group_pattern,
                                              const GrayscaleImage &heightmap,
                                              const std::vector<const GrayscaleImage*> &densitymaps)
{
    const glm::vec2 work_group_footprint{work_group_bounds * layer_data.footprint};
    const glm::vec2 base_offset{lower_bound / work_group_footprint};
    const glm::uvec2 num_work_groups{glm::uvec2((upper_bound - lower_bound) / work_group_footprint) + 1u};

    const glm::uvec2 wg_size{work_group_pattern.size(), work_group_pattern.front().size()};

    constexpr uint invalid_index = -1u;

    std::vector<Result::Element> candidates;
    candidates.resize(num_work_groups.x * num_work_groups.y * wg_size.x * wg_size.y, {{0, 0, 0}, invalid_index});

    std::vector<float> acc_density;
    acc_density.resize(candidates.size());

    std::vector<glm::uvec2> work_group_indices;
    for (uint i = 0; i < num_work_groups.x; i++)
        for (uint j = 0; j < num_work_groups.y; j++)
            work_group_indices.emplace_back(i, j);

    std::vector<glm::uvec2> invocation_indices;
    for (uint i = 0; i < work_group_pattern.size(); i++)
        for (uint j = 0; j < work_group_pattern.front().size(); j++)
            invocation_indices.emplace_back(i, j);

    std::for_each(policy, work_group_indices.cbegin(), work_group_indices.cend(),
                  [&, work_group_footprint, base_offset, num_work_groups, wg_size](glm::uvec2 wg_id)
                  {
                      const uint wg_array_index = (wg_id.x * num_work_groups.y + wg_id.y) * wg_size.x * wg_size.y;
                      const glm::vec2 wg_offset = base_offset + glm::vec2(wg_id) * work_group_footprint;

                      std::for_each(policy, invocation_indices.cbegin(), invocation_indices.cend(),
                                    [&, wg_array_index, wg_offset](glm::uvec2 inv_id)
                                    {
                                        const uint inv_array_index = wg_array_index + inv_id.x * wg_size.y + inv_id.y;
                                        const glm::vec2 inv_position =
                                                wg_offset + work_group_pattern[inv_id.x][inv_id.y];

                                        const glm::vec2 candidate_uv{inv_position / glm::vec2(world_data.scale)};

                                        auto &candidate = candidates[inv_array_index];
                                        candidate.position = {inv_position, heightmap.sample(candidate_uv)};

                                        if (glm::any(glm::lessThan(glm::vec2(candidate.position), lower_bound)) ||
                                            glm::any(glm::greaterThanEqual(glm::vec2(candidate.position), upper_bound)))
                                            return;

                                        for (uint i; i < layer_data.densitymaps.size(); i++)
                                        {
                                            const auto &d_map = layer_data.densitymaps[i];

                                            const float layer_density = glm::clamp(densitymaps[i]->sample(candidate_uv)
                                                                                   * d_map.scale + d_map.offset,
                                                                                   d_map.min_value, d_map.max_value);

                                            const auto threshold =
                                                    EvaluationKernel::default_dithering_matrix[inv_id.x][inv_id.y];

                                            if ((acc_density[inv_array_index] += layer_density) > threshold)
                                            {
                                                candidate.class_index = i;
                                                return;
                                            }
                                        }
                                    });
                  });

    std::sort(policy, candidates.begin(), candidates.end(),
              [](const Result::Element &l, const Result::Element &r)
              { return l.class_index < r.class_index; });

    return candidates;
}

TEST_CASE("Benchmarks")
{
    constexpr auto heightmap_filename = "assets/textures/grayscale/heightmap.png";

    placement::WorldData world_data{{10000, 10000, 1.f}, s_texture_loader[heightmap_filename]};

    const GrayscaleImage heightmap_image {heightmap_filename};

    const auto layer_random = GENERATE(take(1, chunk(20, random(0.5f, 1.5f))));

    constexpr auto densitymap_filename = "assets/textures/grayscale/radial_gradient.png";

    placement::LayerData layer_data{1.f};
    for (uint i = 0; i < 10; i++)
    {
        auto &dm = layer_data.densitymaps.emplace_back();
        dm.scale = layer_random[i * 2];
        dm.offset = layer_random[i * 2 + 1];
        dm.texture = s_texture_loader[densitymap_filename];
    }

    const GrayscaleImage densitymap_image {densitymap_filename};
    std::vector<const GrayscaleImage*> densitymaps;
    densitymaps.resize(layer_data.densitymaps.size(), &densitymap_image);

    const uint seed = 0;
    const auto work_group_pattern = generateWorkGroupPattern(seed);

    const auto single_thread_placement = [&](glm::vec2 upper_bound)
    {
        auto result = computePlacement(std::execution::seq, world_data, layer_data, {0, 0}, upper_bound,
                                       work_group_pattern.first, work_group_pattern.second,
                                       heightmap_image, densitymaps);
        CHECK(!result.empty());
        return result;
    };

    BENCHMARK("10x10 Single-thread CPU placement")
                { return single_thread_placement({10, 10}); };
    BENCHMARK("100x100 Single-thread CPU placement")
                { return single_thread_placement({100, 100}); };
    BENCHMARK("1000x1000 Single-thread CPU placement")
                { return single_thread_placement({1000, 1000}); };

#ifdef PLACEMENT_BENCHMARK_MULTITHREAD
    const auto multi_thread_placement = [&](float bounds)
    {
        auto result = computePlacement(std::execution::par_unseq, world_data, layer_data, {0, 0}, {bounds, bounds},
                                       work_group_pattern.first, work_group_pattern.second,
                                       heightmap_image, densitymaps);
        CHECK(!result.empty());
        return result;
    };

    BENCHMARK("10x10 Multi-thread CPU placement")
                { return multi_thread_placement(10); };
    BENCHMARK("100x100 Multi-thread CPU placement")
                { return multi_thread_placement(100); };
    BENCHMARK("1000x1000 Multi-thread CPU placement")
                { return multi_thread_placement(1000); };
#endif

    placement::PlacementPipeline pipeline;
    pipeline.setRandomSeed(seed);

    const auto dispatch_gpu_placement = [&](float bounds)
    {
        return pipeline.computePlacement(world_data, layer_data, {0, 0}, {bounds, bounds});
    };

    BENCHMARK("10x10 GPU placement dispatch")
                { return dispatch_gpu_placement(10); };
    BENCHMARK("100x100 GPU placement dispatch")
                { return dispatch_gpu_placement(100); };
    BENCHMARK("1000x1000 GPU placement dispatch")
                { return dispatch_gpu_placement(1000); };

    const auto gpu_placement = [&](float bounds)
    {
        auto result = dispatch_gpu_placement(bounds).readResult();
        CHECK(result.getElementArrayLength() > 0);
        return result;
    };

    BENCHMARK("10x10 GPU placement")
                { return gpu_placement(10); };
    BENCHMARK("100x100 GPU placement")
                { return gpu_placement(100); };
    BENCHMARK("1000x1000 GPU placement")
                { return gpu_placement(1000); };

    const auto poisson_placement = [&](float bounds)
    {
        DiskDistributionGenerator disk_generator{layer_data.footprint,
                                                 glm::vec2(bounds) * glm::sqrt(2.f) / layer_data.footprint};

        const auto bounds_approx = Approx(bounds).margin(layer_data.footprint / glm::sqrt(2.f));
        CHECK(disk_generator.getGrid().getBounds().x == bounds_approx);
        CHECK(disk_generator.getGrid().getBounds().y == bounds_approx);

        std::default_random_engine layer_gen(seed);
        std::uniform_int_distribution<uint> layer_dist(0, layer_data.densitymaps.size() - 1);

        std::vector<Result::Element> elements;

        const auto work_group_linear_density =
                glm::vec2(GenerationKernel::work_group_size) / work_group_pattern.first;
        const auto expected_elements_by_axis = work_group_linear_density * glm::vec2(world_data.scale);
        const std::size_t expected_elements = expected_elements_by_axis.x * expected_elements_by_axis.y;

        elements.reserve(expected_elements);

        try
        {
            while (elements.size() < expected_elements)
                elements.push_back({glm::vec3(disk_generator.generate(), 0), layer_dist(layer_gen)});
        }
        catch (std::exception &e)
        { /* ... */ }

        return elements;
    };

    BENCHMARK("10x10 Poisson placement")
                { poisson_placement(10); };
    BENCHMARK("100x100 Poisson placement")
                { poisson_placement(100); };
    BENCHMARK("1000x1000 Poisson placement")
                { poisson_placement(1000); };

    SUCCEED("Benchmarks finished");
}