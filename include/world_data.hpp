#ifndef PROCEDURAL_PLACEMENT_LIB_WORLD_DATA_HPP
#define PROCEDURAL_PLACEMENT_LIB_WORLD_DATA_HPP

#include "gl_types.hpp"

namespace placement {

    struct Vec2
    {
        float x;
        float y;
    };

    /// Specifies world data textures and grid parameters.
    class WorldData
    {
    public:
        /**
         * @brief Construct world data specification.
         * @param height_texture the OpenGL name of a 2D texture object. Determines the height of the terrain.
         * @param density_texture the OpenGL name of a 2D texture object. Determines the density of the placed objects.
         * @param world_size size of the terrain represented in the height and density textures.
         * @param grid_cell_size size of the grid cells, in world space.
         * @param world_origin_uv UV coordinates of the world origin.
         */
        WorldData(GLuint height_texture, GLuint density_texture, Vec2 world_size = {1.0f, 1.0f},
                  Vec2 grid_cell_size = {1.0f, 1.0f}, Vec2 world_origin_uv = {0.0f, 0.0f});

        /// Heightmap texture for the terrain.
        auto getHeightMap() const -> GLuint;
        void setHeightMap(GLuint texture);

        /// Density texture for placement.
        auto getDensityMap() const -> GLuint;
        void setDensityMap(GLuint texture);

        /// Dimensions of the world.
        auto getWorldSize() const -> Vec2;
        void setWorldSize(Vec2 size);

        /// Dimensions of the grid cells; placement is computed in a per-cell basis.
        auto getGridCellSize() const -> Vec2;
        void setGridCellSize(Vec2 size);

        /// Coordinates of the world origin in UV space.
        auto getWorldOriginUV() const -> Vec2;
        void setWorldOriginUV(Vec2 size);

    private:
        GLuint m_heightmap_name;
        GLuint m_densitymap_name;
        Vec2 m_world_size;
        Vec2 m_grid_cell_size;
        Vec2 m_world_origin_uv;
    };

} // pp

#endif //PROCEDURAL_PLACEMENT_LIB_WORLD_DATA_HPP
