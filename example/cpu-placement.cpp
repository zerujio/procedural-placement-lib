#include "cpu-placement.hpp"

#include "placement/kernel/generation_kernel.hpp"
#include "placement/kernel/evaluation_kernel.hpp"
#include "../src/disk_distribution_generator.hpp"

#include "stb_image.h"

#include <array>
#include <algorithm>
#include <execution>

namespace placement {

GrayscaleImage::GrayscaleImage(const char *filename) : m_data(stbi_load(filename, &m_size.x, &m_size.y, nullptr, 1))
{
    if (!m_data)
        throw std::runtime_error(stbi_failure_reason());
}

float GrayscaleImage::sample(glm::vec2 tex_coord) const
{
    const glm::ivec2 pixel_index = glm::clamp(tex_coord, {0, 0}, {1, 1}) * glm::vec2(m_size);
    const stbi_uc value = m_data[pixel_index.y * m_size.x + pixel_index.x];
    return static_cast<float>(value) / static_cast<float>(std::numeric_limits<stbi_uc>::max());
}

void GrayscaleImage::DataDeleter::operator()(void *ptr) const
{
    stbi_image_free(ptr);
}

Result::Result(std::vector<Element> elements) : m_elements(std::move(elements))
{
    const auto findLayerStart = [](uint layer_index, auto begin, auto end)
    {
        return std::find_if(begin, end,
                            [layer_index](const Element &e)
                            { return e.class_index == layer_index; });
    };

    for (auto iter = m_elements.cbegin(); iter != m_elements.cend();
         iter = findLayerStart(m_layer_iters.size(), iter, m_elements.cend()))
    {
        m_layer_iters.emplace_back(iter);
    }

    m_layer_iters.emplace_back(m_elements.cend());
}

auto Result::getClassElements(uint layer_index) const -> std::pair<ConstElementIterator, ConstElementIterator>
{
    if (layer_index >= m_layer_iters.size())
        throw std::out_of_range("layer index out of range");

    return std::make_pair(m_layer_iters[layer_index], m_layer_iters[layer_index + 1]);
}

uint Result::getClassElementCount(uint layer_index) const
{
    auto [begin, end] = getClassElements(layer_index);

    return end - begin;
}

const Result::Element *Result::getClassElementData(uint layer_index) const
{
    if (layer_index >= m_layer_iters.size())
        throw std::out_of_range("layer index out of range");

    return &(*m_layer_iters[layer_index]);
}

FutureResult::FutureResult(std::shared_ptr<ResultBuffer> result_buffer) : m_buffer(std::move(result_buffer))
{
    if (!m_buffer)
        throw std::logic_error("result_buffer is null");
}

Result FutureResult::readResult()
{
    std::unique_lock<std::mutex> lock(m_buffer->m_mutex);

    if (!isReady())
        m_buffer->m_cond.wait(lock, [this]
        { return isReady(); });

    return Result(std::move(m_buffer->m_values));
}

PlacementPipeline::PlacementPattern PlacementPipeline::generatePlacementPattern(uint seed)
{
    WorkGroupPattern pattern;

    placement::DiskDistributionGenerator generator{1.0f, glm::uvec2(16)};
    generator.setSeed(seed);
    generator.setMaxAttempts(100);

    for (auto &column: pattern)
        for (auto &cell: column)
            cell = generator.generate();

    //auto [bounds, pattern] = generateWorkGroupPattern(seed);
    return {generator.getGrid().getBounds(), pattern};
}

PlacementPipeline::~PlacementPipeline()
{
    m_destructor_flag = true;
    m_cond.notify_all();
    m_thread.join();
}

FutureResult PlacementPipeline::computePlacement(WorldData world_data, LayerData layer_data, glm::vec2 lower_bound,
                                                 glm::vec2 upper_bound)
{
    auto result_buffer = std::make_shared<ResultBuffer>();

    {
        std::lock_guard<std::mutex> queue_lock {m_mutex};
        m_queue.emplace_back(Request{world_data, std::move(layer_data), lower_bound, upper_bound, result_buffer});
    }
    m_cond.notify_one();

    return FutureResult(std::move(result_buffer));
}

void PlacementPipeline::threadLoop()
{
    while (not m_destructor_flag)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (m_queue.empty())
            m_cond.wait(lock, [this] { return !m_queue.empty() || m_destructor_flag; });

        if (m_destructor_flag)
            return;

        auto request = m_queue.back();
        m_queue.pop_back();

        lock.unlock();

#ifdef CPU_PLACEMENT_PARALLEL
        constexpr auto execution_policy = std::execution::par_unseq;
#else
        constexpr auto execution_policy = std::execution::seq;
#endif

        auto result_values = computePlacement(execution_policy,
                                              request.world_data, request.layer_data,
                                              request.lower_bound, request.upper_bound);

        auto &result_buffer = *request.result_buffer;
        {
            std::lock_guard<std::mutex> r_lock(result_buffer.m_mutex);
            result_buffer.m_values = std::move(result_values);
            result_buffer.m_ready = true;
        }
        result_buffer.m_cond.notify_all();
    }
}

template<class ExecutionPolicy>
std::vector<placement::Result::Element>
PlacementPipeline::computePlacement(const ExecutionPolicy &policy, const WorldData &world_data,
                                    const LayerData &layer_data, glm::vec2 lower_bound, glm::vec2 upper_bound)
{
    if (!world_data.heightmap)
        throw std::logic_error("invalid world height map");

    const glm::vec2 work_group_footprint{m_pattern.bounds * layer_data.footprint};
    const glm::uvec2 base_offset{lower_bound / work_group_footprint};
    const glm::uvec2 num_work_groups{glm::uvec2((upper_bound - lower_bound) / work_group_footprint) + 1u};

    const glm::uvec2 wg_size{m_pattern.array.size(), m_pattern.array.front().size()};

    constexpr uint invalid_index = -1u;

    std::vector<placement::Result::Element> candidates;
    candidates.resize(num_work_groups.x * num_work_groups.y * wg_size.x * wg_size.y, {{0, 0, 0}, invalid_index});

    std::vector<glm::uvec2> work_group_indices;
    for (uint i = 0; i < num_work_groups.x; i++)
        for (uint j = 0; j < num_work_groups.y; j++)
            work_group_indices.emplace_back(i, j);

    std::vector<glm::uvec2> invocation_indices;
    for (uint i = 0; i < m_pattern.array.size(); i++)
        for (uint j = 0; j < m_pattern.array.front().size(); j++)
            invocation_indices.emplace_back(i, j);

    std::for_each(policy, work_group_indices.cbegin(), work_group_indices.cend(),
                  [&, work_group_footprint, base_offset, num_work_groups, wg_size](glm::uvec2 wg_id)
                  {
                      const uint wg_array_index = (wg_id.x * num_work_groups.y + wg_id.y) * wg_size.x * wg_size.y;
                      const glm::vec2 wg_offset = glm::vec2(base_offset + wg_id) * work_group_footprint;

                      std::for_each(policy, invocation_indices.cbegin(), invocation_indices.cend(),
                                    [&, wg_array_index, wg_offset](glm::uvec2 inv_id)
                                    {
                                        const uint inv_array_index = wg_array_index + inv_id.x * wg_size.y + inv_id.y;
                                        const glm::vec2 inv_position = wg_offset + m_pattern[inv_id.x][inv_id.y] * layer_data.footprint;

                                        const glm::vec2 candidate_uv{inv_position / glm::vec2(world_data.scale)};

                                        auto &candidate = candidates[inv_array_index];
                                        candidate.position = {inv_position,
                                                              world_data.heightmap->sample(candidate_uv) * world_data.scale.z};

                                        if (glm::any(glm::lessThan(glm::vec2(candidate.position), lower_bound)) ||
                                            glm::any(glm::greaterThanEqual(glm::vec2(candidate.position),
                                                                           upper_bound)))
                                            return;

                                        float acc_density = 0.0f;

                                        const auto& dithering_matrix = placement::EvaluationKernel::default_dithering_matrix;
                                        const auto threshold = dithering_matrix[inv_id.x][inv_id.y];

                                        for (uint i = 0; i < layer_data.densitymaps.size(); i++)
                                        {
                                            const auto &d_map = layer_data.densitymaps[i];

                                            const auto *density_map = layer_data.densitymaps[i].texture;

                                            if (!density_map)
                                                throw std::logic_error("invalid density map!");

                                            const float layer_density = glm::clamp(density_map->sample(candidate_uv)
                                                                                   * d_map.scale + d_map.offset,
                                                                                   d_map.min_value,
                                                                                   d_map.max_value);

                                            acc_density += layer_density;
                                            if (acc_density > threshold && candidate.class_index == invalid_index)
                                            {
                                                candidate.class_index = i;
                                                return;
                                            }
                                        }
                                    });
                  });

    std::sort(policy, candidates.begin(), candidates.end(),
              [](const placement::Result::Element &l, const placement::Result::Element &r)
              { return l.class_index < r.class_index; });

    const auto invalid_begin = std::find_if(candidates.begin(), candidates.end(), [](const auto &candidate)
    {
        return candidate.class_index == invalid_index;
    });
    candidates.erase(invalid_begin, candidates.end());

    return candidates;
}
} // namespace placement
