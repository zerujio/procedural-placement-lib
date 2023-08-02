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

    ResultBuffer result_buffer {class_count, size, GL::Buffer(), nullptr};

    using SFlags = GL::Buffer::StorageFlags;

    GL::BufferHandle buffer = result_buffer.gl_object;
    buffer.allocateImmutable(size, SFlags::map_read | SFlags::map_persistent | SFlags::map_coherent, nullptr);

    using AFlags = GL::Buffer::AccessFlags;
    result_buffer.mapped_ptr = static_cast<const std::byte*>(buffer.mapRange(0, size, AFlags::read | AFlags::coherent | AFlags::persistent));

    if (!result_buffer.mapped_ptr)
        throw std::runtime_error("GL memory mapping error!");

    gl.ClearNamedBufferSubData(buffer.getName(), GL_R8, 0, class_count * uint_size, GL_RED,  GL_UNSIGNED_BYTE, nullptr);

    return result_buffer;
}

uint PlacementPipeline::m_getBindingIndex(uint buffer_index) const
{
    return m_base_binding_index + buffer_index;
}

namespace {

struct TransientBuffer
{
public:
    explicit TransientBuffer(uint candidate_count)
    {
        constexpr GLsizeiptr candidate_size = sizeof(float) * 4;
        m_candidate_range = allocate(candidate_count * candidate_size);

        constexpr GLsizeiptr density_size = sizeof(float);
        m_density_range = allocate(candidate_count * density_size);

        constexpr GLsizeiptr world_uv_size = sizeof(float) * 2;
        m_world_uv_range = allocate(candidate_count * world_uv_size);

        constexpr GLsizeiptr index_size = sizeof(uint);
        m_index_range = allocate(candidate_count * index_size);

        m_buffer.allocateImmutable(m_size, GL::Buffer::StorageFlags::none);
    }

    [[nodiscard]] GL::BufferHandle getBuffer() const { return m_buffer; }

    [[nodiscard]] GL::Buffer::Range getCandidateRange() const { return m_candidate_range; }

    [[nodiscard]] GL::Buffer::Range getDensityRange() const { return m_density_range; }

    [[nodiscard]] GL::Buffer::Range getWorldUVRange() const { return m_world_uv_range; }

    [[nodiscard]] GL::Buffer::Range getIndexRange() const { return m_index_range; }

private:
    GL::Buffer m_buffer;
    GL::Buffer::Range m_candidate_range;
    GL::Buffer::Range m_density_range;
    GL::Buffer::Range m_world_uv_range;
    GL::Buffer::Range m_index_range;
    GLsizeiptr m_size {0};

    GL::Buffer::Range allocate(GLsizeiptr alloc_size)
    {
        const auto offset = m_size;
        m_size += alloc_size;
        return { offset, alloc_size };
    }
};


enum BufferIndex
{
    candidate_buffer_index,
    world_uv_buffer_index,
    density_buffer_index,
    index_buffer_index,
    count_buffer_index,
    element_buffer_index
};

auto makeBindingArray(const TransientBuffer &transient_buffer, const ResultBuffer &result_buffer)
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

void bindBuffers(uint base_index, const TransientBuffer& transient_buffer, const ResultBuffer& result_buffer)
{
    const auto bindings = makeBindingArray(transient_buffer, result_buffer);

    GL::Buffer::bindRanges(GL::Buffer::IndexedTarget::shader_storage, base_index, bindings.begin(), bindings.end());
}

} // namespace

FutureResult PlacementPipeline::computePlacement(const WorldData &world_data, const LayerData &layer_data,
                                                 glm::vec2 lower_bound, glm::vec2 upper_bound)
{
    constexpr glm::uvec2 wg_size{GenerationKernel::work_group_size};
    const glm::vec2 wg_bounds = m_work_group_scale * layer_data.footprint;

    const glm::uvec2 work_group_offset{lower_bound / wg_bounds};
    const glm::uvec3 num_work_groups = {1u + glm::uvec2((upper_bound - lower_bound) / wg_bounds), 1u};

    const uint candidate_count = num_work_groups.x * num_work_groups.y * wg_size.x * wg_size.y;

    TransientBuffer transient_buffer {candidate_count};

    ResultBuffer result_buffer = s_makeResultBuffer(candidate_count, layer_data.densitymaps.size());

    bindBuffers(m_base_binding_index, transient_buffer, result_buffer);

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

} // placement