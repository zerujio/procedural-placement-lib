#ifndef PROCEDURALPLACEMENTLIB_CPU_PLACEMENT_HPP
#define PROCEDURALPLACEMENTLIB_CPU_PLACEMENT_HPP

#include "glm/vec2.hpp"
#include "glm/vec3.hpp"

#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <array>

namespace placement
{

struct ResultElement
{
    glm::vec3 position;
    uint class_index;
};

class GrayscaleImage
{
public:
    explicit GrayscaleImage(const char* filename);

    [[nodiscard]] glm::uvec2 getSize() const { return m_size; }

    [[nodiscard]] float sample(glm::vec2 tex_coord) const;

private:
    using uchar = unsigned char;
    struct DataDeleter { void operator() (void*) const; };

    glm::ivec2 m_size {0};
    std::unique_ptr<uchar[], DataDeleter> m_data;
};

struct WorldData
{
    glm::vec3 scale;
    const GrayscaleImage *heightmap;
};

struct DensityMap
{
    const GrayscaleImage *texture{nullptr};
    float scale{1};
    float offset{0};
    float min_value{0};
    float max_value{0};
};

struct LayerData
{
    float footprint;
    std::vector<DensityMap> densitymaps;
};

struct ResultBuffer
{
    bool m_ready = false;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::vector<ResultElement> m_values;
};

class Result
{
public:
    using Element = ResultElement;
    using ElementIterator = std::vector<Element>::iterator;
    using ConstElementIterator = std::vector<Element>::const_iterator;

    explicit Result(std::vector<Element> elements);

    [[nodiscard]] std::pair<ConstElementIterator, ConstElementIterator> getClassElements(uint layer_index) const;

    [[nodiscard]] uint getClassElementCount(uint layer_index) const;

    [[nodiscard]] const Element* getClassElementData(uint layer_index) const;

    [[nodiscard]] uint getNumClasses() const { return m_layer_iters.size() - 1; }

    [[nodiscard]] const std::vector<Element>& getElements() const { return m_elements; }

    [[nodiscard]] uint getElementArrayLength() const { return m_elements.size(); }

private:
    std::vector<Element> m_elements;
    std::vector<std::vector<Element>::const_iterator> m_layer_iters;
};

class FutureResult
{
public:
    explicit FutureResult(std::shared_ptr<ResultBuffer> result_buffer);

    [[nodiscard]] bool isReady() const
    { return m_buffer->m_ready; }

    Result readResult();

private:
    std::shared_ptr<ResultBuffer> m_buffer;
};


class PlacementPipeline final
{
public:
    ~PlacementPipeline();

    FutureResult computePlacement(WorldData world_data, LayerData layer_data,
                                  glm::vec2 lower_bound, glm::vec2 upper_bound);

private:
    void threadLoop();

    template<class ExecutionPolicy>
    [[nodiscard]]
    std::vector<placement::Result::Element> computePlacement(const ExecutionPolicy &policy,
                                                             const WorldData &world_data,
                                                             const LayerData &layer_data,
                                                             glm::vec2 lower_bound, glm::vec2 upper_bound);

    struct Request
    {
        WorldData world_data;
        LayerData layer_data;
        glm::vec2 lower_bound;
        glm::vec2 upper_bound;
        std::shared_ptr<ResultBuffer> result_buffer;
    };

    using WorkGroupPattern = std::array<std::array<glm::vec2, 8>, 8>;

    struct PlacementPattern
    {
        glm::vec2 bounds;
        WorkGroupPattern array;

        auto& operator[] (std::size_t i)
        {
            return array[i];
        }
    };

    static PlacementPattern generatePlacementPattern(uint seed);

    bool m_destructor_flag = false;
    std::vector<Request> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond;
    std::thread m_thread{&PlacementPipeline::threadLoop, this};
    PlacementPattern m_pattern = generatePlacementPattern(123);
};


} // placement

#endif //PROCEDURALPLACEMENTLIB_CPU_PLACEMENT_HPP