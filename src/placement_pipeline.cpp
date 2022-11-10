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
        m_world_data.scale = scale;
    }

    auto PlacementPipeline::getWorldScale() const -> const glm::vec3 &
    {
        return m_world_data.scale;
    }

    auto PlacementPipeline::computePlacement(float footprint, glm::vec2 lower_bound,
                                             glm::vec2 upper_bound) const -> std::vector<glm::vec3>
    {
        using namespace glutils;

        // check if empty area
        if (! glm::all(glm::lessThan(lower_bound, upper_bound)))
            return {};

        gl.BindTextureUnit(GenerationKernel::s_default_heightmap_tex_unit, m_world_data.height_tex);
        gl.BindTextureUnit(GenerationKernel::s_default_densitymap_tex_unit, m_world_data.density_tex);

        const auto num_workgroups = m_generation_kernel.setArgs(m_world_data.scale, footprint, lower_bound, upper_bound);
        const auto num_invocations = num_workgroups * GenerationKernel::work_group_size;
        const auto candidate_count = num_invocations.x * num_invocations.y;

        Guard<Buffer> buffer;

        const BufferRange position_range {*buffer, 0,
                                          PlacementPipelineKernel::getPositionBufferRequiredSize(candidate_count)};
        const BufferRange index_range {*buffer, position_range.size,
                                       PlacementPipelineKernel::getIndexBufferRequiredSize(candidate_count)};
        const GLsizeiptr buffer_size = position_range.size + index_range.size;

        buffer->allocateImmutable(buffer_size,
                                  Buffer::StorageFlags::dynamic_storage | Buffer::StorageFlags::map_read);


        position_range.bindRange(Buffer::IndexedTarget::shader_storage,
                                 PlacementPipelineKernel::default_position_ssb_binding);
        index_range.bindRange(Buffer::IndexedTarget::shader_storage,
                              PlacementPipelineKernel::default_index_ssb_binding);

        // generate positions
        m_generation_kernel.dispatchCompute(num_workgroups);

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // discard invalid candidates
        auto count_offset = static_cast<GLintptr>(m_reduction_kernel(num_workgroups.x * num_workgroups.y));

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // read results
        GLuint valid_candidate_count = 0;
        buffer->read(index_range.offset + count_offset, sizeof(GLuint), &valid_candidate_count);

        if (valid_candidate_count == 0)
            return {};


        auto position_ptr = static_cast<const glm::vec4*>(position_range.map(Buffer::AccessFlags::read));

        if (!position_ptr)
            // TODO: replace this for some other error class.
            throw std::runtime_error("OpenGL buffer access error");

        std::vector<glm::vec3> positions;
        positions.reserve(valid_candidate_count);
        while (positions.size() < valid_candidate_count)
            positions.emplace_back(*position_ptr++);

        return positions;
    }
} // placement