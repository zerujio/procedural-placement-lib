#ifndef PROCEDURALPLACEMENTLIB_DISK_DISTRIBUTION_GENERATOR_HPP
#define PROCEDURALPLACEMENTLIB_DISK_DISTRIBUTION_GENERATOR_HPP

#include "glm/vec2.hpp"

#include <vector>
#include <random>
#include <limits>
#include <optional>

namespace placement {

class DiskDistributionGrid
{
public:
    /**
     * @brief Construct a new grid.
     * @param diameter The collision diameter of elements. Minimum distance between any two points is 2 * diameter.
     * @param size The number of cells in the grid. Each cell is a square with side equal to 2 * diameter / sqrt(2).
     */
    DiskDistributionGrid(float diameter, glm::uvec2 size);

    /// attempt to insert a new position into the grid, checking for collisions.
    bool tryInsert(glm::vec2 position);

    /// Access to all the values in the grid.
    [[nodiscard]] const std::vector<glm::vec2>& getPositions() const {return m_positions;}

    /// Determine the cell index that the given position falls into. Said index may fall outside the boundaries of the grid.
    [[nodiscard]] glm::uvec2 getCellIndex(glm::vec2 position) const;

    /// Get the position contained in the given cell.
    [[nodiscard]] std::optional<glm::vec2> get(glm::uvec2 cell_index) const;

    /// Check if an object at @p position collides with the one possibly contained in cell_index + index_offset.
    [[nodiscard]] bool collides(glm::vec2 position, glm::uvec2 cell_index, glm::ivec2 index_offset = {0, 0}) const;

    /// Get the number of cells in the grid
    [[nodiscard]] glm::uvec2 getSize() const {return m_grid_size;}

    /// dimensions of the square region covered by the grid
    [[nodiscard]] glm::vec2 getBounds() const {return glm::vec2(m_grid_size) * m_disk_diameter / std::sqrt(2.0f);}

private:
    static constexpr std::size_t invalid_index = std::numeric_limits<std::size_t>::max();

    float m_disk_diameter;
    glm::uvec2 m_grid_size;
    std::vector<std::size_t> m_grid;
    std::vector<glm::vec2> m_positions;

    [[nodiscard]] std::size_t& m_gridCell(glm::uvec2 index);
    [[nodiscard]] const std::size_t& m_gridCell(glm::uvec2 index) const;
};

class DiskDistributionGenerator
{
public:
    /**
     * @brief Create a new generator.
     * @param diameter Collision diameter for objects.
     * @param size of the placement area, in grid cells. Each cell is square with side = 2 * diameter / sqrt(2).
     */
    DiskDistributionGenerator(float diameter, glm::uvec2 size) :
        m_grid(diameter, size),
        m_dist_x(0.0f, m_grid.getBounds().x),
        m_dist_y(0.0f, m_grid.getBounds().y)
    {}

    glm::vec2 generate();

    [[nodiscard]] const std::vector<glm::vec2>& getPositions() const { return m_grid.getPositions(); }

    void setMaxAttempts(std::size_t n) { m_max_attempts = n; }
    [[nodiscard]] std::size_t getMaxAttempts() const { return m_max_attempts; }

    void setSeed(uint s) { m_rand.seed(s); }

    [[nodiscard]] const DiskDistributionGrid& getGrid() const { return m_grid; }
private:
    DiskDistributionGrid m_grid;
    std::size_t m_max_attempts = 25;
    std::default_random_engine m_rand;
    std::uniform_real_distribution<float> m_dist_x;
    std::uniform_real_distribution<float> m_dist_y;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_DISK_DISTRIBUTION_GENERATOR_HPP
