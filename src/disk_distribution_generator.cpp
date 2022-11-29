#include "disk_distribution_generator.hpp"

#include "glm/glm.hpp"

#include <stdexcept>
#include <array>

namespace placement {

DiskDistributionGrid::DiskDistributionGrid(float radius, glm::vec2 size) :
    m_diameter(2 * radius),
    m_bounds(size),
    m_grid_size(glm::uvec2 (size * sqrt2 / m_diameter) + 1u)
{
    if (size.x <= 0.0f || size.y <= 0.0f)
        throw std::logic_error("size must be >= 0"); // TODO: swap this for some other error type

    m_grid.resize(m_grid_size.x * m_grid_size.y, glm::vec2(-1.0f));
}

glm::uvec2 DiskDistributionGrid::m_findCellIndex(glm::vec2 position) const
{
    if (glm::any(glm::lessThan(position, glm::vec2(0.0f))) || glm::any(glm::greaterThan(position, m_bounds)))
        throw std::logic_error("position out of bounds");

    return {position * sqrt2 / m_diameter};
}

glm::vec2 &DiskDistributionGrid::m_getCellValue(glm::uvec2 index)
{
    return m_grid[index.x + index.y * m_grid_size.x];
}

const glm::vec2 &DiskDistributionGrid::m_getCellValue(glm::uvec2 index) const
{
    return m_grid[index.x + index.y * m_grid_size.x];
}

bool DiskDistributionGrid::m_collides(glm::vec2 position, glm::uvec2 cell) const
{
    const auto& cell_position = m_getCellValue(cell);

    if (cell_position.x < 0.0f || cell_position.y < 0.0f)
        return false;

    return glm::distance(cell_position, position) <= m_diameter;
}

bool DiskDistributionGrid::tryInsert(glm::vec2 position)
{
    const auto cell = m_findCellIndex(position);

    for (int dx = -1; dx <= 1; dx++)
    {
        const uint x = dx >= 0 ? (cell.x + dx) % m_grid_size.x : (m_grid_size.x + cell.x - 1) % m_grid_size.x;
        for (int dy = -1; dy <= 1; dy++)
        {
            const uint y = dy >= 0 ? (cell.y + dy) % m_grid_size.y : (m_grid_size.y + cell.y - 1) % m_grid_size.y;
            if (m_collides(position, {x, y}))
                return false;
        }
    }

    m_getCellValue(cell) = position;
    return true;
}

glm::vec2 DiskDistributionGenerator::generate()
{
    for (std::size_t i = 0; i < m_max_attempts; i++)
    {
        const glm::vec2 candidate {m_dist_x(m_rand), m_dist_y(m_rand)};
        if (m_grid.tryInsert(candidate))
            return candidate;
    }

    throw std::runtime_error("maximum insertion attempts exceeded");
}
} // placement