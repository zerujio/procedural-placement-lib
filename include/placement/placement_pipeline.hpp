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
#include "density_map.hpp"

#include <vector>
#include <chrono>
#include <optional>

namespace placement {

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
    [[nodiscard]] static ResultBuffer s_makeResultBuffer(uint candidate_count, uint class_count);
    [[nodiscard]] uint m_getBindingIndex(uint buffer_index) const;

    uint m_base_tex_unit {0};
    uint m_base_binding_index {0};
    glm::vec2 m_work_group_scale;
    GenerationKernel m_generation_kernel;
    EvaluationKernel m_evaluation_kernel;
    IndexationKernel m_indexation_kernel;
    CopyKernel m_copy_kernel;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP
