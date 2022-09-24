#include "placement_pipeline.hpp"

#include "glutils/gl.hpp"

#include "glm/glm.hpp"

#include <stdexcept>

namespace placement {
    using glutils::gl;

    void PlacementPipeline::setHeightTexture(unsigned int tex)
    {
        m_world_data.height_tex = tex;
    }

    auto PlacementPipeline::getHeightTexture() const -> unsigned int
    {
        return m_world_data.height_tex;
    }

    void PlacementPipeline::setDensityTexture(unsigned int tex)
    {
        m_world_data.density_tex = tex;
    }

    auto PlacementPipeline::getDensityTexture() const -> unsigned int
    {
        return m_world_data.density_tex;
    }

    void PlacementPipeline::setWorldScale(const glm::vec3 &scale)
    {
        m_world_data.world_scale = scale;
    }

    auto PlacementPipeline::getWorldScale() const -> const glm::vec3 &
    {
        return m_world_data.world_scale;
    }

    auto PlacementPipeline::computePlacement(float footprint, glm::vec2 lower_bound,
                                             glm::vec2 upper_bound) const -> std::vector<glm::vec3>
    {
        // empty area
        if (! glm::all(glm::lessThanEqual(lower_bound, upper_bound)))
            return {};


    }
} // placement