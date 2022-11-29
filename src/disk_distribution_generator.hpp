#ifndef PROCEDURALPLACEMENTLIB_DISK_DISTRIBUTION_GENERATOR_HPP
#define PROCEDURALPLACEMENTLIB_DISK_DISTRIBUTION_GENERATOR_HPP

#include "glm/vec2.hpp"

#include <vector>
#include <random>

namespace placement {

class DiskDistributionGrid
{
public:
    /**
     * @brief Construct a new grid.
     * @param radius The collision radius of elements. Minimum distance between any two points is 2 * radius.
     * @param size The dimensions of the grid. Must be positive.
     */
    DiskDistributionGrid(float radius, glm::vec2 size);

    bool tryInsert(glm::vec2 position);

    [[nodiscard]] const std::vector<glm::vec2>& getVector() const {return m_grid;}

private:
    static constexpr float sqrt2 = 1.41421f;

    float m_diameter;
    glm::vec2 m_bounds;
    glm::uvec2 m_grid_size;
    std::vector<glm::vec2> m_grid;

    [[nodiscard]] glm::uvec2 m_findCellIndex(glm::vec2) const;
    [[nodiscard]] glm::vec2& m_getCellValue(glm::uvec2 index);
    [[nodiscard]] const glm::vec2& m_getCellValue(glm::uvec2 index) const;
    [[nodiscard]] bool m_collides(glm::vec2 position, glm::uvec2 cell) const;
};

class DiskDistributionGenerator
{
public:
    DiskDistributionGenerator(float radius, glm::vec2 bounds) :
        m_grid(radius, bounds),
        m_dist_x(0.0f, bounds.x),
        m_dist_y(0.0f, bounds.y)
    {}

    glm::vec2 generate();

    [[nodiscard]] const std::vector<glm::vec2>& getGridVector() const {return m_grid.getVector();}

    void setMaxAttempts(std::size_t n) {m_max_attempts = n;}
    [[nodiscard]] std::size_t getMaxAttempts() const {return m_max_attempts;}

    void setSeed(uint s) {m_rand.seed(s);}
private:
    std::default_random_engine m_rand;
    std::uniform_real_distribution<float> m_dist_x;
    std::uniform_real_distribution<float> m_dist_y;
    DiskDistributionGrid m_grid;
    std::size_t m_max_attempts = 10;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_DISK_DISTRIBUTION_GENERATOR_HPP
