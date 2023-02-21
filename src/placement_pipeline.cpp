#include "placement/placement_pipeline.hpp"
#include "gl_context.hpp"
#include "disk_distribution_generator.hpp"

#include "glutils/guard.hpp"
#include "glutils/buffer.hpp"

#include <stdexcept>

namespace placement {

PlacementPipeline::PlacementPipeline()
{
    setBaseTextureUnit(0);
    setBaseShaderStorageBindingPoint(0);
    setRandomSeed(0);
}

FutureResult PlacementPipeline::computePlacement(const WorldData &world_data, const LayerData &layer_data,
                                                 glm::vec2 lower_bound, glm::vec2 upper_bound)
{
    constexpr glm::uvec2 wg_size{GenerationKernel::work_group_size};
    constexpr auto wg_scale = glm::vec2(wg_size) * s_wg_scale_factor;

    const glm::uvec2 work_group_offset{lower_bound / wg_scale};
    const glm::uvec3 num_work_groups = {1u + glm::uvec2((upper_bound - lower_bound) / wg_scale), 1u};

    const uint candidate_count = num_work_groups.x * num_work_groups.y * wg_size.x * wg_size.y;

    m_buffer.resize(candidate_count);

    constexpr auto ssb_binding = GL::Buffer::IndexedTarget::shader_storage;

    m_buffer.getBuffer().bindRange(ssb_binding, m_getCandidateBufferBindingIndex(), m_buffer.getCandidateRange());
    m_buffer.getBuffer().bindRange(ssb_binding, m_getDensityBufferBindingIndex(), m_buffer.getDensityRange());
    m_buffer.getBuffer().bindRange(ssb_binding, m_getWorldUVBufferBindingIndex(), m_buffer.getWorldUVRange());
    m_buffer.getBuffer().bindRange(ssb_binding, m_getIndexBufferBindingIndex(), m_buffer.getIndexRange());

    // generation
    gl.BindTextureUnit(m_getHeightTexUnit(), world_data.heightmap);
    m_generation_kernel.setWorldScale(world_data.scale);
    m_generation_kernel.setFootprint(layer_data.footprint);
    m_generation_kernel.setWorkGroupOffset(work_group_offset);

    m_generation_kernel.useProgram();
    ComputeKernel::dispatch(num_work_groups);
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // evaluation
    m_evaluation_kernel.useProgram();
    const uint class_count = layer_data.densitymaps.size();
    for (std::size_t i = 0; i < class_count; i++)
    {
        gl.BindTextureUnit(m_getDensityTexUnit(), layer_data.densitymaps[i].texture);
        m_evaluation_kernel.setClassIndex(i);

        ComputeKernel::dispatch(num_work_groups);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    constexpr GLsizeiptr candidate_size = 4 * sizeof(std::uint32_t);

    // indexation
    const GL::Buffer::Range count_range{0, IndexationKernel::getCountBufferMemoryRequirement(class_count)};
    const GL::Buffer::Range element_range{count_range.size, candidate_count * candidate_size};

    ResultBuffer result_buffer{class_count, count_range.size + element_range.size, GL::Buffer()};
    result_buffer.gl_object.allocateImmutable(result_buffer.size,
                                              GL::Buffer::StorageFlags::map_read |
                                              GL::Buffer::StorageFlags::dynamic_storage);
    result_buffer.gl_object.bindRange(ssb_binding, m_getCountBufferBindingIndex(), count_range);
    result_buffer.gl_object.bindRange(ssb_binding, m_getOutputBufferBindingIndex(), element_range);

    {
        const std::vector<GLuint> count_buffer_initializer{class_count, 0u};
        result_buffer.gl_object.write(count_range, count_buffer_initializer.data());
    }

    m_indexation_kernel.useProgram();
    ComputeKernel::dispatch(IndexationKernel::calculateNumWorkGroups(candidate_count));
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // copy

    m_copy_kernel.useProgram();
    ComputeKernel::dispatch(CopyKernel::calculateNumWorkGroups(candidate_count));
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);

    // fence
    auto fence = GL::createFenceSync();
    gl.Flush();

    return {std::move(result_buffer), std::move(fence)};
}

void PlacementPipeline::setBaseTextureUnit(GLuint index)
{
    m_base_tex_unit = index;
    m_generation_kernel.setHeightmapTextureUnit(m_getHeightTexUnit());
    m_evaluation_kernel.setDensityMapTextureUnit(m_getDensityTexUnit());
}

void PlacementPipeline::setBaseShaderStorageBindingPoint(GLuint index)
{
    m_base_binding_index = index;

    // candidate buffer
    {
        const auto i = m_getCandidateBufferBindingIndex();
        m_generation_kernel.setCandidateBufferBindingIndex(i);
        m_evaluation_kernel.setCandidateBufferBindingIndex(i);
        m_indexation_kernel.setCandidateBufferBindingIndex(i);
        m_copy_kernel.setCandidateBufferBindingIndex(i);
    }

    // density buffer
    {
        const auto j = m_getDensityBufferBindingIndex();
        m_generation_kernel.setDensityBufferBindingIndex(j);
        m_evaluation_kernel.setDensityBufferBindingIndex(j);
    }

    // WorldUV buffer
    {
        const auto k = m_getWorldUVBufferBindingIndex();
        m_generation_kernel.setWorldUVBufferBindingIndex(k);
        m_evaluation_kernel.setWorldUVBufferBindingIndex(k);
    }

    // count buffer
    {
        const auto m = m_getCountBufferBindingIndex();
        m_indexation_kernel.setCountBufferBindingIndex(m);
        m_copy_kernel.setCountBufferBindingIndex(m);
    }

    // Index buffer
    {
        const auto l = m_getIndexBufferBindingIndex();
        m_indexation_kernel.setIndexBufferBindingIndex(l);
        m_copy_kernel.setIndexBufferBindingIndex(l);
    }

    // output buffer
    m_copy_kernel.setOutputBufferBindingIndex(m_getOutputBufferBindingIndex());
}

void PlacementPipeline::setRandomSeed(uint seed)
{
    constexpr auto wg_size = GenerationKernel::work_group_size;
    // dart throwing algorithm for poisson disk distribution
    const glm::vec2 wg_scale = glm::vec2(wg_size) * glm::vec2(wg_size);
    m_generation_kernel.setWorkGroupScale(wg_scale);

    DiskDistributionGenerator generator{1.0f, wg_scale};
    generator.setSeed(seed);
    generator.setMaxAttempts(100);

    std::array<std::array<glm::vec2, wg_size.y>, wg_size.x> positions;

    for (uint i = 0; i < wg_size.x; i++)
        for (uint j = 0; j < wg_size.y; j++)
            positions[i][j] = generator.generate();

    m_generation_kernel.setWorkGroupPatternColumns(positions);
}


// PlacementPipeline::Buffer
constexpr GLsizeiptr candidate_size = 4 * sizeof(GLfloat);
constexpr GLsizeiptr density_size = sizeof(GLfloat);
constexpr GLsizeiptr world_uv_size = 2 * sizeof(GLfloat);
constexpr GLsizeiptr index_size = sizeof(unsigned int);

GLsizeiptr PlacementPipeline::Buffer::s_calculateSize(GLsizeiptr capacity)
{
    return (candidate_size + density_size + world_uv_size + index_size) * capacity;
}

class PlacementPipeline::Buffer::Allocator
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

void PlacementPipeline::Buffer::resize(GLsizeiptr candidate_count)
{
    reserve(candidate_count);

    Allocator a;

    m_candidate_range = a.allocate(candidate_size * candidate_count);
    m_density_range = a.allocate(density_size * candidate_count);
    m_world_uv_range = a.allocate(world_uv_size * candidate_count);
    m_index_range = a.allocate(index_size * candidate_count);
}

void PlacementPipeline::Buffer::reserve(GLsizeiptr candidate_count)
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