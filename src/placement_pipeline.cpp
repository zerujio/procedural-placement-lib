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

    auto PlacementPipeline::computePlacement(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound)
    -> std::vector<glm::vec3>
    {
        using namespace glutils;

        // check if empty area
        if (! glm::all(glm::lessThan(lower_bound, upper_bound)))
            return {};

        gl.BindTextureUnit(GenerationKernel::s_default_heightmap_tex_unit, m_world_data.height_tex);
        gl.BindTextureUnit(GenerationKernel::s_default_densitymap_tex_unit, m_world_data.density_tex);

        const auto candidate_count = m_generation_kernel.setArgs(m_world_data.scale, footprint, lower_bound, upper_bound);

        Guard<Buffer> buffer;

        const BufferRange position_range {*buffer, 0,
                                          static_cast<GLsizeiptr>(PlacementPipelineKernel::calculatePositionBufferSize(candidate_count))};
        const BufferRange index_range {*buffer, position_range.size,
                                       static_cast<GLsizeiptr>(PlacementPipelineKernel::calculateIndexBufferSize(candidate_count))};
        const GLsizeiptr buffer_size = position_range.size + index_range.size;

        buffer->allocateImmutable(buffer_size,Buffer::StorageFlags::map_read);

        position_range.bindRange(Buffer::IndexedTarget::shader_storage,
                                 PlacementPipelineKernel::default_position_ssb_binding);
        index_range.bindRange(Buffer::IndexedTarget::shader_storage,
                              PlacementPipelineKernel::default_index_ssb_binding);

        // generate positions
        m_generation_kernel.dispatchCompute();

        // discard invalid candidates
        auto count_offset = static_cast<GLintptr>(m_reduction_kernel.dispatchCompute(candidate_count));

        // read results
        GLuint valid_candidate_count = 0;
        buffer->read(index_range.offset + count_offset, sizeof(GLuint), &valid_candidate_count);

        if (valid_candidate_count == 0)
            return {};

        auto position_ptr = static_cast<const glm::vec4*>(position_range.map(Buffer::AccessFlags::read));

        if (!position_ptr)
            // TODO: replace this with some other error class.
            throw std::runtime_error("OpenGL buffer access error");

        std::vector<glm::vec3> positions;
        positions.reserve(valid_candidate_count);
        while (positions.size() < valid_candidate_count)
            positions.emplace_back(*position_ptr++);

        buffer->unmap();

        return positions;
    }
} // placement