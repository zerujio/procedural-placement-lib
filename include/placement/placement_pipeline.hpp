#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP

#include "generation_kernel.hpp"
#include "reduction_kernel.hpp"

#include "glm/glm.hpp"

#include <vector>

namespace placement {

    class PlacementPipeline
    {
    public:
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
         * @return an array of positions within the placement area, separated from each other by at least 2x @p footprint.
         */
         [[nodiscard]]
         auto computePlacement(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound) const -> std::vector<glm::vec3>;

        /**
         * @brief The heightmap texture.
         * @param tex integer name for a 2D OpenGL texture object.
         */
        void setHeightTexture(unsigned int tex);

        [[nodiscard]]
        auto getHeightTexture() const -> unsigned int;

        /**
         * @brief The density map texture.
         * @param tex integer name for a 2D OpenGL texture object.
         */
        void setDensityTexture(unsigned int tex);

        [[nodiscard]]
        auto getDensityTexture() const -> unsigned int;

        /**
         * @brief The scale of the world.
         * @param scale
         */
        void setWorldScale(const glm::vec3& scale);

        [[nodiscard]]
        auto getWorldScale() const -> const glm::vec3&;

        /**
         * @brief set the seed for the random number generator.
         * For a given set of heightmap, densitymap and world scale, the random seed completely determines placement.
         */
        void setRandomSeed() const;

    private:

        struct WorldData
        {
            unsigned int height_tex;
            unsigned int density_tex;
            glm::vec3 world_scale;
        } m_world_data;

        GenerationKernel m_generation_kernel;
        ReductionKernel m_reduction_kernel;
        glutils::GLuint m_position_binding {};
        glutils::GLuint m_index_binding;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_HPP
