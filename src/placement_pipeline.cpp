#include "placement/placement_pipeline.hpp"
#include "gl_context.hpp"
#include "disk_distribution_generator.hpp"

#include "glutils/guard.hpp"
#include "glutils/buffer.hpp"

#include <stdexcept>

namespace placement {

using Candidate = Result::Element;

PlacementPipeline::PlacementPipeline()
{
    setBaseTextureUnit(0);
    setBaseShaderStorageBindingPoint(0);
    setRandomSeed(0);
}

ResultBuffer PlacementPipeline::s_makeResultBuffer(uint candidate_count, uint class_count)
{
    constexpr GLsizeiptr result_element_size = sizeof(glm::vec4);
    constexpr GLsizeiptr uint_size = sizeof(uint);

    const auto size = class_count * uint_size + candidate_count * result_element_size;

    GL::Buffer buffer;
    buffer.allocateImmutable(size, GL::Buffer::StorageFlags::map_read, nullptr);
    gl.ClearNamedBufferSubData(buffer.getName(), GL_R8, 0, class_count * uint_size, GL_RED, GL_UNSIGNED_BYTE, nullptr);

    return {class_count, size, std::move(buffer)};
}

enum BufferIndex
{
    candidate_buffer_index,
    world_uv_buffer_index,
    density_buffer_index,
    index_buffer_index,
    count_buffer_index,
    element_buffer_index
};

auto PlacementPipeline::s_makeBindingArray(const PlacementPipeline::TransientBuffer &transient_buffer,
                                           const ResultBuffer &result_buffer)
{
    std::array<std::pair<GL::BufferHandle, GL::Buffer::Range>, 6> array;

    array[candidate_buffer_index] = {transient_buffer.getBuffer(), transient_buffer.getCandidateRange()};
    array[world_uv_buffer_index] = {transient_buffer.getBuffer(), transient_buffer.getWorldUVRange()};
    array[density_buffer_index] = {transient_buffer.getBuffer(), transient_buffer.getDensityRange()};
    array[index_buffer_index] = {transient_buffer.getBuffer(), transient_buffer.getIndexRange()};
    array[count_buffer_index] = {result_buffer.gl_object, result_buffer.getCountRange()};
    array[element_buffer_index] = {result_buffer.gl_object, result_buffer.getElementRange()};

    return array;
}

uint PlacementPipeline::m_getBindingIndex(uint buffer_index) const
{
    return m_base_binding_index + buffer_index;
}

//DEBUG
template<typename T>
std::vector<T> dumpBufferRange(GL::BufferHandle buffer, GL::Buffer::Range range,
                               GLenum barrier_bits = GL_BUFFER_UPDATE_BARRIER_BIT)
{
    gl.MemoryBarrier(barrier_bits);

    const GLsizeiptr size = range.size / sizeof(T);

    std::vector<T> vector;
    vector.reserve(size);

    auto ptr = static_cast<const T *>(buffer.mapRange(range, GL::Buffer::AccessFlags::read));
    vector.insert(vector.end(), ptr, ptr + size);
    buffer.unmap();

    return vector;
}

//DEBUG

FutureResult PlacementPipeline::computePlacement(const WorldData &world_data, const LayerData &layer_data,
                                                 glm::vec2 lower_bound, glm::vec2 upper_bound)
{
    constexpr glm::uvec2 wg_size{GenerationKernel::work_group_size};
    const glm::vec2 wg_bounds = m_work_group_scale * layer_data.footprint;

    const glm::uvec2 work_group_offset{lower_bound / wg_bounds};
    const glm::uvec3 num_work_groups = {1u + glm::uvec2((upper_bound - lower_bound) / wg_bounds), 1u};

    const uint candidate_count = num_work_groups.x * num_work_groups.y * wg_size.x * wg_size.y;

    m_buffer.resize(candidate_count);

    ResultBuffer result_buffer = s_makeResultBuffer(candidate_count, layer_data.densitymaps.size());

    const auto buffer_bindings = s_makeBindingArray(m_buffer, result_buffer);

    GL::Buffer::bindRanges(GL::Buffer::IndexedTarget::shader_storage, m_base_binding_index,
                           buffer_bindings.begin(), buffer_bindings.end());

    // generation
    gl.BindTextureUnit(m_base_tex_unit, world_data.heightmap);
    m_generation_kernel(num_work_groups, work_group_offset, layer_data.footprint, world_data.scale, m_base_tex_unit,
                        m_getBindingIndex(candidate_buffer_index), m_getBindingIndex(world_uv_buffer_index),
                        m_getBindingIndex(density_buffer_index));
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // evaluation
    const uint class_count = layer_data.densitymaps.size();
    for (std::size_t i = 0; i < class_count; i++)
    {
        gl.BindTextureUnit(m_base_tex_unit, layer_data.densitymaps[i].texture);
        m_evaluation_kernel(num_work_groups, work_group_offset, i, lower_bound, upper_bound, m_base_tex_unit,
                            layer_data.densitymaps[i],
                            m_getBindingIndex(candidate_buffer_index),
                            m_getBindingIndex(world_uv_buffer_index),
                            m_getBindingIndex(density_buffer_index));
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // indexation
    m_indexation_kernel(IndexationKernel::calculateNumWorkGroups(candidate_count),
                        m_getBindingIndex(candidate_buffer_index), m_getBindingIndex(count_buffer_index),
                        m_getBindingIndex(index_buffer_index));
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // copy
    m_copy_kernel(CopyKernel::calculateNumWorkGroups(candidate_count), m_getBindingIndex(candidate_buffer_index),
                  m_getBindingIndex(count_buffer_index), m_getBindingIndex(index_buffer_index),
                  m_getBindingIndex(element_buffer_index));
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // fence
    auto fence = GL::createFenceSync();
    gl.Flush();

    return {std::move(result_buffer), std::move(fence)};
}

void PlacementPipeline::setBaseTextureUnit(GLuint index)
{
    m_base_tex_unit = index;
}

void PlacementPipeline::setBaseShaderStorageBindingPoint(GLuint index)
{
    m_base_binding_index = index;
}

void PlacementPipeline::setRandomSeed(uint seed)
{
    constexpr auto wg_size = GenerationKernel::work_group_size;

    DiskDistributionGenerator generator{1.0f, wg_size * 2u};
    generator.setSeed(seed);
    generator.setMaxAttempts(100);
    m_work_group_scale = generator.getGrid().getBounds();
    m_generation_kernel.setWorkGroupPatternBoundaries(m_work_group_scale);

    std::array<std::array<glm::vec2, wg_size.y>, wg_size.x> positions;
    for (auto &column: positions)
        for (auto &cell: column)
            cell = generator.generate();

    m_generation_kernel.setWorkGroupPatternColumns(positions);
}

// PlacementPipeline::TransientBuffer
constexpr GLsizeiptr candidate_size = 4 * sizeof(GLfloat);
constexpr GLsizeiptr density_size = sizeof(GLfloat);
constexpr GLsizeiptr world_uv_size = 2 * sizeof(GLfloat);
constexpr GLsizeiptr index_size = sizeof(unsigned int);

GLsizeiptr PlacementPipeline::TransientBuffer::s_calculateSize(GLsizeiptr capacity)
{
    return (candidate_size + density_size + world_uv_size + index_size) * capacity;
}

class PlacementPipeline::TransientBuffer::Allocator
{
public:
    GL::BufferHandle::Range allocate(GLsizeiptr size)
    {
        const GL::BufferHandle::Range r{offset, size};
        offset += size;
        return r;
    }

private:
    GLintptr offset = 0;
};

void PlacementPipeline::TransientBuffer::resize(GLsizeiptr candidate_count)
{
    reserve(candidate_count);

    Allocator a;

    m_candidate_range = a.allocate(candidate_size * candidate_count);
    m_density_range = a.allocate(density_size * candidate_count);
    m_world_uv_range = a.allocate(world_uv_size * candidate_count);
    m_index_range = a.allocate(index_size * candidate_count);
}

void PlacementPipeline::TransientBuffer::reserve(GLsizeiptr candidate_count)
{
    const GLsizeiptr required_size = s_calculateSize(candidate_count);

    if (required_size <= m_capacity)
        return;

    GLsizeiptr new_buffer_size = std::max(m_capacity, s_calculateSize(s_min_capacity));

    while (new_buffer_size < required_size)
        new_buffer_size <<= 1;

    m_buffer.allocate(new_buffer_size, GL::BufferHandle::Usage::dynamic_read);
    m_capacity = new_buffer_size;
}

} // placement