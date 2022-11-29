#include "placement/placement_pipeline.hpp"
#include "gl_context.hpp"
#include "disk_distribution_generator.hpp"

#include "glutils/guard.hpp"
#include "glutils/buffer.hpp"

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

void PlacementPipeline::computePlacement(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound)
{
    using namespace glutils;

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
    m_buffer.getBuffer().bindRange(glutils::Buffer::IndexedTarget::shader_storage,
                                   m_generation_kernel.getCandidateBufferBindingIndex(),
                                   m_buffer.getCandidateRange());

    m_generation_kernel.dispatchCompute();

    // index valid candidates
    auto count_range = m_buffer.getIndexRange();
    count_range.size = static_cast<GLsizeiptr>(sizeof(unsigned int));
    m_valid_count = 0;
    m_buffer.getBuffer().write(count_range, &m_valid_count);

    m_buffer.getBuffer().bindRange(glutils::Buffer::IndexedTarget::shader_storage,
                                   m_assignment_kernel.getIndexBufferBindingIndex(),
                                   m_buffer.getIndexRange());

    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_assignment_kernel.dispatchCompute(candidate_count);

    // copy valid candidates
    m_buffer.getBuffer().bindRange(glutils::Buffer::IndexedTarget::shader_storage,
                                   m_copy_kernel.getPositionBufferBindingIndex(),
                                   m_buffer.getPositionRange());

    gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    m_copy_kernel.dispatchCompute(candidate_count);

    // read valid candidate count
    gl.MemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    m_buffer.getBuffer().read(count_range, &m_valid_count);
}

void PlacementPipeline::setBaseTextureUnit(glutils::GLuint index)
{
    m_generation_kernel.setHeightTextureUnit(index);
    m_generation_kernel.setDensityTextureUnit(index);
}

void PlacementPipeline::setBaseShaderStorageBindingPoint(glutils::GLuint index)
{
    m_setCandidateBufferBindingIndex(index);
    m_setIndexBufferBindingIndex(index + 1);
    m_setPositionBufferBindingIndex(index + 2);
}

void PlacementPipeline::m_setCandidateBufferBindingIndex(glutils::GLuint index)
{
    m_generation_kernel.setCandidateBufferBindingIndex(index);
    m_assignment_kernel.setCandidateBufferBindingIndex(index);
    m_copy_kernel.setCandidateBufferBindingIndex(index);
}

void PlacementPipeline::m_setIndexBufferBindingIndex(glutils::GLuint index)
{
    m_assignment_kernel.setIndexBufferBindingIndex(index);
    m_copy_kernel.setIndexBufferBindingIndex(index);
}

void PlacementPipeline::m_setPositionBufferBindingIndex(glutils::GLuint index)
{
    m_copy_kernel.setPositionBufferBindingIndex(index);
}

std::vector<glm::vec3> PlacementPipeline::copyResultsToCPU() const
{
    if (m_valid_count == 0)
        return {};

    std::vector<glm::vec3> positions;
    positions.reserve(m_valid_count);

    auto gpu_positions = static_cast<const glm::vec4 *>(m_buffer.getBuffer().mapRange(m_buffer.getPositionRange(),
                                                                                      glutils::Buffer::AccessFlags::read));
    for (GLintptr i = 0; i < m_valid_count; i++)
        positions.emplace_back(gpu_positions[i]);

    m_buffer.getBuffer().unmap();

    return positions;
}

void PlacementPipeline::copyResultsToGPUBuffer(glutils::GLuint buffer, glutils::GLsizeiptr offset) const
{
    const auto range = m_getResultRange();
    gl.CopyNamedBufferSubData(m_buffer.getBuffer().getName(), buffer, range.offset, offset, range.size);
}

glutils::Buffer::Range PlacementPipeline::m_getResultRange() const
{
    return {m_buffer.getPositionRange().offset,
            static_cast<GLsizeiptr>(m_valid_count * sizeof(glm::vec4))};
}

void PlacementPipeline::setRandomSeed(uint seed)
{
    constexpr auto wg_size = GenerationKernel::work_group_size;
    // dart throwing algorithm for poisson disk distribution
    DiskDistributionGenerator generator {1.0f, glm::vec2(wg_size) * GenerationKernel::s_work_group_scale};
    generator.setSeed(seed);

    std::array<std::array<glm::vec2, wg_size.y>, wg_size.x> positions;

    for (uint i = 0; i < wg_size.x; i++)
        for (uint j = 0; j < wg_size.y; j++)
            positions[i][j] = generator.generate();

    m_generation_kernel.setPositionStencil(positions);
}


// PlacementPipeline::Buffer

glutils::GLsizeiptr PlacementPipeline::Buffer::s_calculateSize(glutils::GLsizeiptr capacity)
{
    return GenerationKernel::calculateCandidateBufferSize(capacity)
           + IndexAssignmentKernel::calculateIndexBufferSize(capacity)
           + IndexedCopyKernel::calculatePositionBufferSize(capacity);
}

glutils::Buffer::Range PlacementPipeline::Buffer::getCandidateRange() const
{
    return m_candidate_range;
}

glutils::Buffer::Range PlacementPipeline::Buffer::getIndexRange() const
{
    return m_index_range;
}

glutils::Buffer::Range PlacementPipeline::Buffer::getPositionRange() const
{
    return m_position_range;
}

class PlacementPipeline::Buffer::Allocator
{
public:
    glutils::Buffer::Range allocate(GLsizeiptr size)
    {
        const glutils::Buffer::Range r{offset, size};
        offset += size;
        return r;
    }

private:
    GLintptr offset = 0;
};

void PlacementPipeline::Buffer::resize(glutils::GLsizeiptr candidate_count)
{
    reserve(candidate_count);

    Allocator a;

    m_index_range = a.allocate(IndexAssignmentKernel::calculateIndexBufferSize(candidate_count));
    m_candidate_range = a.allocate(GenerationKernel::calculateCandidateBufferSize(candidate_count));
    m_position_range = a.allocate(IndexedCopyKernel::calculatePositionBufferSize(candidate_count));
}

void PlacementPipeline::Buffer::reserve(GLsizeiptr candidate_count)
{
    const GLsizeiptr required_size = s_calculateSize(candidate_count);

    if (required_size <= m_capacity)
        return;

    GLsizeiptr new_buffer_size = std::max(m_capacity, s_calculateSize(s_min_capacity));

    while (new_buffer_size < required_size)
        new_buffer_size <<= 1;

    m_buffer->allocate(new_buffer_size, glutils::Buffer::Usage::dynamic_read);
    m_capacity = new_buffer_size;
}

glutils::Buffer PlacementPipeline::Buffer::getBuffer() const
{
    return m_buffer.getHandle();
}

} // placement