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
        using Candidate = PlacementPipelineKernel::Candidate;

        // empty area
        if (! glm::all(glm::lessThan(lower_bound, upper_bound)))
            return {};

        gl.BindTextureUnit(GenerationKernel::s_default_heightmap_tex_unit, m_world_data.height_tex);
        gl.BindTextureUnit(GenerationKernel::s_default_densitymap_tex_unit, m_world_data.density_tex);

        const auto num_workgroups = m_generation_kernel.setArgs(m_world_data.scale, footprint, lower_bound, upper_bound);
        const auto num_invocations = num_workgroups * GenerationKernel::work_group_size;
        const auto candidate_count = num_invocations.x * num_invocations.y;
        const GLsizeiptr candidate_buffer_size = static_cast<GLsizeiptr>(sizeof(Candidate)) * candidate_count;

        Guard<Buffer> buffer;
        buffer->allocateImmutable(candidate_buffer_size,
                                  Buffer::StorageFlags::dynamic_storage | Buffer::StorageFlags::map_read);
        buffer->bindRange(Buffer::IndexedTarget::shader_storage, PlacementPipelineKernel::default_ssb_binding,
                          0 /*bytes*/, candidate_buffer_size);

        m_generation_kernel.dispatchCompute(num_workgroups);

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        static_cast<GLintptr>(m_reduction_kernel(num_workgroups.x * num_workgroups.y));

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        // read results
        auto candidates = static_cast<const Candidate*>(buffer->map(Buffer::AccessMode::read_only));

        if (!candidates)
            throw std::runtime_error("failed to map output buffer");

        const auto valid_candidate_count = candidates[candidate_count - 1].index;

        std::vector<glm::vec3> positions;
        positions.reserve(valid_candidate_count);

        while (positions.size() < valid_candidate_count)
            positions.emplace_back(candidates++->position);

        buffer->unmap();

        return positions;
    }
} // placement