#include "placement/placement_pipeline.hpp"
#include "gl_context.hpp"

#include "glutils/guard.hpp"
#include "glutils/buffer.hpp"

#include "glm/glm.hpp"

#include <stdexcept>

namespace placement {

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
        m_world_data.world_scale = scale;
    }

    auto PlacementPipeline::getWorldScale() const -> const glm::vec3 &
    {
        return m_world_data.world_scale;
    }

    auto PlacementPipeline::computePlacement(float footprint, glm::vec2 lower_bound,
                                             glm::vec2 upper_bound) const -> std::vector<glm::vec3>
    {
        using namespace glutils;

        // empty area
        if (! glm::all(glm::lessThan(lower_bound, upper_bound)))
            return {};

        gl.BindTextureUnit(GenerationKernel::s_default_heightmap_tex_unit, m_world_data.height_tex);
        gl.BindTextureUnit(GenerationKernel::s_default_densitymap_tex_unit, m_world_data.density_tex);

        const auto num_invocations = GenerationKernel::computeNumInvocations(footprint, lower_bound, upper_bound);
        const auto candidate_count = num_invocations.x * num_invocations.y;
        const GLsizeiptr position_buffer_size = PlacementPipelineKernel::computePositionBufferSize(candidate_count);
        const GLsizeiptr index_buffer_size = PlacementPipelineKernel::computeIndexBufferSize(candidate_count);

        Guard<Buffer> buffer;
        buffer->allocateImmutable(position_buffer_size + index_buffer_size,
                                  Buffer::StorageFlags::dynamic_storage | Buffer::StorageFlags::map_read);
        buffer->bindRange(Buffer::IndexedTarget::shader_storage,
                          PlacementPipelineKernel::s_default_position_buffer_binding,
                          0, position_buffer_size);
        buffer->bindRange(Buffer::IndexedTarget::shader_storage,
                          PlacementPipelineKernel::s_default_index_buffer_binding,
                          position_buffer_size, index_buffer_size);

        m_generation_kernel.dispatchCompute(footprint, m_world_data.world_scale, lower_bound, upper_bound);

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        const auto result_offset = static_cast<GLintptr>(m_reduction_kernel.dispatchCompute(candidate_count));

        // read results
        auto mapped_ptr = buffer->map(Buffer::AccessMode::read_only);
        if (!mapped_ptr)
            throw std::runtime_error("failed to map output buffer");

        auto position_count = *reinterpret_cast<const unsigned int*>(static_cast<const GLbyte*>(mapped_ptr)
                                                                    + position_buffer_size + result_offset);
        std::vector<glm::vec3> positions;
        positions.reserve(position_count);

        auto position_buffer = static_cast<const glm::vec3*>(mapped_ptr);
        for (auto position_ptr = position_buffer; position_ptr < position_buffer + position_count; position_ptr++)
            positions.emplace_back(*position_ptr);

        buffer->unmap();

        return positions;
    }
} // placement