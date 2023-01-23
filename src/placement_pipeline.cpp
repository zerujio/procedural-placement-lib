#include "placement/placement_pipeline.hpp"
#include "gl_context.hpp"
#include "disk_distribution_generator.hpp"

#include "glutils/guard.hpp"
#include "glutils/buffer.hpp"
#include "glutils/error.hpp"

#include <stdexcept>

namespace placement {


// PlacementPipeline

PlacementPipeline::PlacementPipeline()
{
    setBaseTextureUnit(0);
    setBaseShaderStorageBindingPoint(0);
    setRandomSeed(0);
}

void PlacementPipeline::setHeightTexture(unsigned int tex)
{
    m_world_data.height_tex = tex;
}

auto PlacementPipeline::getHeightTexture() const -> unsigned int
{
    return m_world_data.height_tex;
}

void PlacementPipeline::setDensityTexture(unsigned int tex)
{
    m_world_data.density_tex = tex;
}

auto PlacementPipeline::getDensityTexture() const -> unsigned int
{
    return m_world_data.density_tex;
}

void PlacementPipeline::setWorldScale(const glm::vec3 &scale)
{
    m_world_data.scale = scale;
}

auto PlacementPipeline::getWorldScale() const -> const glm::vec3 &
{
    return m_world_data.scale;
}

FutureResult PlacementPipeline::computePlacement(const WorldData &world_data, const LayerData &layer_data,
                                                 glm::vec2 lower_bound, glm::vec2 upper_bound)
{
    constexpr glm::uvec2 wg_size {GenerationKernel::work_group_size};
    constexpr auto wg_scale = glm::vec2(wg_size) * s_wg_scale_factor;

    const glm::uvec2 work_group_offset {lower_bound / wg_scale};
    const glm::uvec2 num_work_groups = 1u + glm::uvec2((upper_bound - lower_bound) / wg_scale);

    const uint candidate_count = num_work_groups.x * num_work_groups.y * wg_size.x * wg_size.y;

    m_buffer.resize(candidate_count);

    constexpr auto ssbo_target = GL::Buffer::IndexedTarget::shader_storage;

    m_buffer.getBuffer().bindRange(ssbo_target, m_getCandidateBufferBindingIndex(), m_buffer.getCandidateRange());
    m_buffer.getBuffer().bindRange(ssbo_target, m_getDensityBufferBindingIndex(), m_buffer.getDensityRange());
    m_buffer.getBuffer().bindRange(ssbo_target, m_getWorldUVBufferBindingIndex(), m_buffer.getWorldUVRange());

    gl.BindTextureUnit(m_getHeightTexUnit(), world_data.heightmap);

    m_generation_kernel.setWorldScale(world_data.scale);
    m_generation_kernel.setFootprint(layer_data.footprint);
    m_generation_kernel.setWorkGroupOffset(work_group_offset);
    m_generation_kernel.useProgram();

    gl.DispatchCompute(num_work_groups.x, num_work_groups.y, 1);
    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_evaluation_kernel.useProgram();
    for (std::size_t i = 0; i < layer_data.densitymaps.size(); i++)
    {
        gl.BindTextureUnit(m_getDensityTexUnit(), layer_data.densitymaps[i].texture);
        m_evaluation_kernel.setClassIndex(i);

        gl.DispatchCompute(num_work_groups.x, num_work_groups.y, 1);
        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    // TODO: call indexation and copy kernel, create output buffer, return result buffer.
}

void PlacementPipeline::computePlacement(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound)
{
    using namespace GL;

    // check if empty area
    if (!glm::all(glm::lessThan(lower_bound, upper_bound)))
    {
        m_valid_count = 0;
        return;
    }

    gl.BindTextureUnit(m_generation_kernel.getHeightTextureUnit(), m_world_data.height_tex);
    gl.BindTextureUnit(m_generation_kernel.getDensitytextureUnit(), m_world_data.density_tex);

    const auto candidate_count = m_generation_kernel.setArgs(m_world_data.scale, footprint, lower_bound, upper_bound);
    m_buffer.resize(candidate_count);

    // generate positions
    m_buffer.getBuffer().bindRange(GL::BufferHandle::IndexedTarget::shader_storage,
                                   m_generation_kernel.getCandidateBufferBindingIndex(),
                                   m_buffer.getCandidateRange());

    m_generation_kernel.dispatchCompute();

    // index valid candidates
    auto count_range = m_buffer.getIndexRange();
    count_range.size = static_cast<GLsizeiptr>(sizeof(unsigned int));
    m_valid_count = 0;
    m_buffer.getBuffer().write(count_range, &m_valid_count);

    m_buffer.getBuffer().bindRange(GL::BufferHandle::IndexedTarget::shader_storage,
                                   m_assignment_kernel.getIndexBufferBindingIndex(),
                                   m_buffer.getIndexRange());

    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_assignment_kernel.dispatchCompute(candidate_count);

    // copy valid candidates
    m_buffer.getBuffer().bindRange(GL::BufferHandle::IndexedTarget::shader_storage,
                                   m_copy_kernel.getPositionBufferBindingIndex(),
                                   m_buffer.getPositionRange());

    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_copy_kernel.dispatchCompute(candidate_count);

    // read valid candidate count
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    m_buffer.getBuffer().read(count_range, &m_valid_count);
}

void PlacementPipeline::setBaseTextureUnit(GL::GLuint index)
{
    m_base_tex_unit = index;
    m_generation_kernel.setHeightmapTextureUnit(m_getHeightTexUnit());
    m_evaluation_kernel.setDensityMapTextureUnit(m_getDensityTexUnit());
}

void PlacementPipeline::setBaseShaderStorageBindingPoint(GL::GLuint index)
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

    // Index buffer
    {
        const auto l = m_getIndexBufferBindingIndex();
        m_indexation_kernel.setIndexBufferBindingIndex(l);
        m_copy_kernel.setIndexBufferBindingIndex(l);
    }

    // count buffer
    m_indexation_kernel.setCountBufferBindingIndex(m_getCountBufferBindingIndex());

    // output buffer
    m_copy_kernel.setOutputBufferBindingIndex(m_getOutputBufferBindingIndex());
}

std::vector<glm::vec3> PlacementPipeline::copyResultsToCPU() const
{
    if (m_valid_count == 0)
        return {};

    std::vector<glm::vec3> positions;
    positions.reserve(m_valid_count);

    auto gpu_positions = static_cast<const glm::vec4 *>(m_buffer.getBuffer().mapRange(m_buffer.getPositionRange(),
                                                                                      GL::BufferHandle::AccessFlags::read));
    for (GLintptr i = 0; i < m_valid_count; i++)
        positions.emplace_back(gpu_positions[i]);

    m_buffer.getBuffer().unmap();

    return positions;
}

void PlacementPipeline::copyResultsToGPUBuffer(GL::GLuint buffer, GL::GLsizeiptr offset) const
{
    const auto range = m_getResultRange();
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    gl.CopyNamedBufferSubData(m_buffer.getBuffer().getName(), buffer, range.offset, offset, range.size);
}

GL::BufferHandle::Range PlacementPipeline::m_getResultRange() const
{
    return {m_buffer.getPositionRange().offset,
            static_cast<GLsizeiptr>(m_valid_count * sizeof(glm::vec4))};
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

GL::GLsizeiptr PlacementPipeline::Buffer::s_calculateSize(GL::GLsizeiptr capacity)
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

void PlacementPipeline::Buffer::resize(GL::GLsizeiptr candidate_count)
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