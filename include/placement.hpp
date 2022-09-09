#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_HPP


#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

#include <vector>

namespace placement {

    using GLuint = unsigned int;

    /// Specifies world texture data and grid dimensions.
    struct WorldData
    {
        /// A 2D OpenGL texture. Determines the height of the terrain.
        GLuint height_texture{0};

        /// A 2D OpenGL texture. Determines the density with which objects are placed.
        GLuint density_texture{0};

        /// Dimensions of the world.
        glm::vec3 scale{10.0f, 10.0f, 10.0f};
    };

    using GLproc = void (*)();

    /**
     * @brief Compute placement for the specified area.
     * Elements will be placed in the area specified by @p lower_bound and @p upperbound . The placement area is
     * defined by all points (x, y) such that
     *
     *      lower_bound.x < x < upper_bound.x
     *      lower_bound.y < y < upper_bound.y
     *
     * If lower_bound is not less than upper_bound, the placement region will have zero area and no elements will be
     * placed.
     *
     * @param world_data world height, density and scale.
     * @param footprint collision radius for the placed objects.
     * @param lower_bound minimum x and y coordinates of the placement area
     * @param upper_bound maximum x and y coordinates of the placement area
     * @return an array of positions within the placement area, separated from each other by at least 2x @p footprint.
     */
    std::vector<glm::vec3> computePlacement(const WorldData& world_data, float footprint,
                                            glm::vec2 lower_bound, glm::vec2 upper_bound);

    using GLloader = GLproc (*)(const char*);

    /// Load OpenGL functions for the current context and thread.
    void loadGL(GLloader gl_loader);

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_HPP
