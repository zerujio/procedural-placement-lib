#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP

#include "placement_result.hpp"
#include "kernel/generation_kernel.hpp"
#include "kernel/evaluation_kernel.hpp"
#include "kernel/indexation_kernel.hpp"
#include "kernel/copy_kernel.hpp"

#include "glutils/sync.hpp"
#include "glutils/buffer.hpp"

#include "glm/glm.hpp"

#include <vector>
#include <chrono>
#include <optional>

namespace placement {

/// A density map specifies the probability distribution of a single class of object over the landscape.
struct DensityMap
{
    /// name of an OpenGL texture object.
    GL::GLuint texture{0};

    /// Values in texture will be multiplied by this factor.
    float scale{1};

    /// Values in texture will be offset by this amount, after scaling.
    float offset{0};

    /// Values in texture will be clamped to the range [min_value, max_value], after scaling and offset.
    float min_value{0};
    float max_value{1};
};

/// Layer data holds information for multiple object types with the same footprint.
struct LayerData
{
    /// Minimum separation between any two placed object, i.e. a collision diameter.
    float footprint;

    /// An array of density maps, each one representing a different "object class".
    std::vector<DensityMap> densitymaps;
};

/// World data contains information about the landscape objects are placed on.
struct WorldData
{
    /// Dimensions of the world
    glm::vec3 scale;

    /// Name of an OpenGL texture object to be used as the heightmap of the terrain.
    GL::GLuint heightmap;
};

class PlacementPipeline
{
public:
    PlacementPipeline();

    /**
     * @brief Compute placement for the specified area.
     * Elements will be placed in the area specified by @p lower_bound and @p upperbound . The placement area is
     * defined by all points (x, y) such that
     *
     *      lower_bound.x <= x < upper_bound.x
     *      lower_bound.y <= y < upper_bound.y
     *
     * i.e. the half open [lower_bound, upper_bound) range. Consequently, if lower_bound is not less than upper_bound,
     * the placement region will have zero area and no elements will be placed. In that case no error will occur and an
     * empty vector will be returned.
     *
     * @param footprint collision radius for the placed objects.
     * @param lower_bound minimum x and y coordinates of the placement area
     * @param upper_bound maximum x and y coordinates of the placement area
     */
    [[deprecated("replaced by overload with explicit WorldData and LayerData arguments")]]
    void computePlacement(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound);

    /// Multiclass placement.
    [[nodiscard]]
    FutureResult computePlacement(const WorldData &world_data, const LayerData &layer_data,
                                  glm::vec2 lower_bound, glm::vec2 upper_bound);

    /**
     * @brief The heightmap texture.
     * @param tex integer name for a 2D OpenGL texture object.
     */
    [[deprecated("heightmap should be passed through WorldData")]]
    void setHeightTexture(unsigned int tex);

    [[deprecated("heightmap is now part of WorldData")]]
    [[nodiscard]]
    auto getHeightTexture() const -> unsigned int;

    /**
     * @brief The density map texture.
     * @param tex integer name for a 2D OpenGL texture object.
     */
    [[deprecated("density maps are now part of LayerData")]]
    void setDensityTexture(unsigned int tex);

    [[deprecated("density maps are now part of LayerData")]]
    [[nodiscard]]
    auto getDensityTexture() const -> unsigned int;

    /**
     * @brief The scale of the world.
     * @param scale
     */
    [[deprecated("world scale is now part of WorldData")]]
    void setWorldScale(const glm::vec3 &scale);

    [[deprecated("world scale is now part of WorldData")]]
    [[nodiscard]]
    auto getWorldScale() const -> const glm::vec3 &;

    /**
     * @brief set the seed for the random number generator.
     * For a given set of heightmap, densitymap and world scale, the random seed completely determines placement.
     */
    void setRandomSeed(uint seed);

    /// The number of different texture units used by the placement compute shaders
    static constexpr auto required_texture_units = 2u;

    /**
     * @brief Configures the texture units the pipeline will use
     * @param index the index of a texture unit such that indices in the range [index, index + required_texture_units)
     *      are all valid texture unit indices.
     */
    void setBaseTextureUnit(GL::GLuint index);

    /// The number of different shader storage buffer binding points used by the placement compute shaders.
    static constexpr auto required_shader_storage_binding_points = 6u;

    /**
     * @brief Configures the shader storage buffer binding points the pipeline will use.
     * @param index An index such that elements in the range [index, index + required_shader_storage_binding_points)
     *      are valid shader storage buffer binding points.
     */
    void setBaseShaderStorageBindingPoint(GL::GLuint index);

    /// Copy results from GPU buffer to CPU memory
    [[deprecated("replaced by PlacementResults::copyToHost()")]]
    [[nodiscard]]
    std::vector<glm::vec3> copyResultsToCPU() const;

    /// Return the number of valid positions generated by the last placement operation.
    [[deprecated("replaced by PlacementResults::getSize()")]]
    [[nodiscard]] std::size_t getResultsSize() const
    { return m_valid_count; }

    /**
     * @brief Copy results to another buffer using glCopyNamedBufferSubData()
     * @param buffer the integer name of an opengl buffer. Its size must be at
     * least @p offset + getResultsSize() * sizeof(vec4).
     * @param offset a byte offset into @p buffer
     */
    [[deprecated("replaced by PlacementResults::copyData")]]
    void copyResultsToGPUBuffer(GL::GLuint buffer, GL::GLsizeiptr offset = 0u) const;

private:

    [[nodiscard]] uint m_getCandidateBufferBindingIndex() const noexcept { return m_base_binding_index + 0; }
    [[nodiscard]] uint m_getDensityBufferBindingIndex() const noexcept { return m_base_binding_index + 1; }
    [[nodiscard]] uint m_getWorldUVBufferBindingIndex() const noexcept { return m_base_binding_index + 2; }
    [[nodiscard]] uint m_getIndexBufferBindingIndex() const noexcept { return m_base_binding_index + 3; }
    [[nodiscard]] uint m_getCountBufferBindingIndex() const noexcept { return m_base_binding_index + 4; }
    [[nodiscard]] uint m_getOutputBufferBindingIndex() const noexcept { return m_base_binding_index + 5; }

    [[nodiscard]] uint m_getHeightTexUnit() const noexcept { return m_base_tex_unit + 0; }
    [[nodiscard]] uint m_getDensityTexUnit() const noexcept { return m_base_tex_unit + 1; }

    [[nodiscard]] GL::Buffer::Range m_getResultRange() const;

    uint m_base_tex_unit {0};
    uint m_base_binding_index {0};
    GenerationKernel m_generation_kernel;
    EvaluationKernel m_evaluation_kernel;
    IndexationKernel m_indexation_kernel;
    CopyKernel m_copy_kernel;

    static constexpr glm::vec2 s_wg_scale_factor {2.5f};

    /// number of valid candidates. Total candidates are equal to m_buffer.getSize()
    [[deprecated]] GL::GLsizeiptr m_valid_count = 0;

    class Buffer
    {
    public:
        void resize(GL::GLsizeiptr candidate_count);

        void reserve(GL::GLsizeiptr candidate_count);

        [[nodiscard]] GL::GLsizeiptr getSize() const
        { return m_size; }

        [[nodiscard]] GL::GLsizeiptr getCapacity() const
        { return m_capacity; }

        [[nodiscard]] GL::BufferHandle getBuffer() const { return m_buffer; }

        [[nodiscard]] GL::Buffer::Range getCandidateRange() const { return m_candidate_range; }

        [[nodiscard]] GL::Buffer::Range getDensityRange() const { return m_density_range; }

        [[nodiscard]] GL::Buffer::Range getWorldUVRange() const { return m_world_uv_range; }

        [[nodiscard]] GL::Buffer::Range getIndexRange() const { return m_index_range; }

    private:
        GL::Buffer m_buffer{};
        GL::GLsizeiptr m_capacity{0};
        GL::GLsizeiptr m_size{0};

        GL::Buffer::Range m_candidate_range;
        GL::Buffer::Range m_density_range;
        GL::Buffer::Range m_world_uv_range;
        GL::Buffer::Range m_index_range;

        static constexpr GL::GLsizeiptr s_min_capacity = 64;

        static GL::GLsizeiptr s_calculateSize(GL::GLsizeiptr capacity);

        class Allocator;
    } m_buffer;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP
