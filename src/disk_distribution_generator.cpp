#include "disk_distribution_generator.hpp"

#include "glm/glm.hpp"

#include <stdexcept>
#include <array>
#include <algorithm>

namespace placement {

DiskDistributionGrid::DiskDistributionGrid(float radius, glm::uvec2 size) :
        m_disk_diameter(2 * radius),
        m_grid_size(size)
{
    m_grid.resize(m_grid_size.x * m_grid_size.y, invalid_index);
}

glm::uvec2 DiskDistributionGrid::getCellIndex(glm::vec2 position) const
{
    return {position * glm::sqrt(2.0f) / m_disk_diameter};
}

std::optional<glm::vec2> DiskDistributionGrid::get(glm::uvec2 cell_index) const
{
    if (glm::any(glm::greaterThanEqual(cell_index, m_grid_size)))
        throw std::logic_error("cell index out of bounds");

    const auto index = m_gridCell(cell_index);

    if (index == invalid_index)
        return {};
    else
        return {m_positions[index]};
}

std::size_t& DiskDistributionGrid::m_gridCell(glm::uvec2 index)
{
    return m_grid[index.x + index.y * m_grid_size.x];
}

const std::size_t& DiskDistributionGrid::m_gridCell(glm::uvec2 index) const
{
    return m_grid[index.x + index.y * m_grid_size.x];
}

bool DiskDistributionGrid::collides(glm::vec2 position, glm::uvec2 cell_index, glm::ivec2 index_offset) const
{
    const glm::ivec2 offset_cell = glm::ivec2(cell_index) + index_offset;

    const glm::ivec2 wrapped_cell = offset_cell % glm::ivec2(m_grid_size);
    const glm::uvec2 collision_cell = glm::uvec2(wrapped_cell) + glm::uvec2(glm::lessThan(wrapped_cell, glm::ivec2(0))) * m_grid_size;

    const auto other_position = get(collision_cell);

    // check if cell is empty
    if (!other_position)
        return false;

    // the grid repeats. (grid_size, 1) is the same as (0, 1) offset by the length of the grid along the X dimension.
    const glm::vec2 position_offset = glm::floor(glm::vec2(offset_cell) / glm::vec2(m_grid_size))
                                    * glm::vec2(m_grid_size) * m_disk_diameter / glm::sqrt(2.0f);

    return glm::distance(other_position.value() + position_offset, position) <= m_disk_diameter;
}

bool DiskDistributionGrid::tryInsert(glm::vec2 position)
{
    const auto cell_index = getCellIndex(position);

    // insertion fails if cell is not empty
    if (get(cell_index))
        return false;

    for (int dx = -2; dx <= 2; dx++)
        for (int dy = -2; dy <= 2; dy++)
        {
            const glm::ivec2 offset {dx, dy};

            if (offset == glm::ivec2(0) || glm::abs(offset) == glm::ivec2(2))
                continue;

            if (collides(position, cell_index, offset))
                return false;
        }

    // insert element
    m_gridCell(cell_index) = m_positions.size();
    m_positions.emplace_back(position);

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