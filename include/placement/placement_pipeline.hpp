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
    GLuint texture{0};

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
    GLuint heightmap;
};

class PlacementPipeline
{
public:
    PlacementPipeline();

    /// Multiclass placement.
    [[nodiscard]]
    FutureResult computePlacement(const WorldData &world_data, const LayerData &layer_data,
                                  glm::vec2 lower_bound, glm::vec2 upper_bound);

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
    void setBaseTextureUnit(GLuint index);

    /// The number of different shader storage buffer binding points used by the placement compute shaders.
    static constexpr auto required_shader_storage_binding_points = 6u;

    /**
     * @brief Configures the shader storage buffer binding points the pipeline will use.
     * @param index An index such that elements in the range [index, index + required_shader_storage_binding_points)
     *      are valid shader storage buffer binding points.
     */
    void setBaseShaderStorageBindingPoint(GLuint index);

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

    class Buffer
    {
    public:
        void resize(GLsizeiptr candidate_count);

        void reserve(GLsizeiptr candidate_count);

        [[nodiscard]] GLsizeiptr getSize() const
        { return m_size; }

        [[nodiscard]] GLsizeiptr getCapacity() const
        { return m_capacity; }

        [[nodiscard]] GL::BufferHandle getBuffer() const { return m_buffer; }

        [[nodiscard]] GL::Buffer::Range getCandidateRange() const { return m_candidate_range; }

        [[nodiscard]] GL::Buffer::Range getDensityRange() const { return m_density_range; }

        [[nodiscard]] GL::Buffer::Range getWorldUVRange() const { return m_world_uv_range; }

        [[nodiscard]] GL::Buffer::Range getIndexRange() const { return m_index_range; }

    private:
        GL::Buffer m_buffer{};
        GLsizeiptr m_capacity{0};
        GLsizeiptr m_size{0};

        GL::Buffer::Range m_candidate_range;
        GL::Buffer::Range m_density_range;
        GL::Buffer::Range m_world_uv_range;
        GL::Buffer::Range m_index_range;

        static constexpr GLsizeiptr s_min_capacity = 64;

        static GLsizeiptr s_calculateSize(GLsizeiptr capacity);

        class Allocator;
    } m_buffer;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP
