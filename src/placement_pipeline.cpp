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
        // empty area
        if (! glm::all(glm::lessThan(lower_bound, upper_bound)))
            return {};

        gl.BindTextureUnit(GenerationKernel::heightmap_tex_unit, m_world_data.height_tex);
        gl.BindTextureUnit(GenerationKernel::densitymap_tex_unit, m_world_data.density_tex);

        const auto num_work_groups = GenerationKernel::getNumWorkGroups(footprint, lower_bound, upper_bound);

        using namespace glutils;

        Guard<Buffer> output_buffer;
        output_buffer->allocateImmutable(GenerationKernel::getRequiredBufferSize(num_work_groups),
                                         Buffer::StorageFlags::dynamic_storage | Buffer::StorageFlags::map_read);
        gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, GenerationKernel::output_buffer_binding, output_buffer->getName());

        m_generation_kernel.dispatchCompute(footprint, m_world_data.world_scale, lower_bound, upper_bound);

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        struct Candidate
        {
            glm::vec3 position;
            uint index;
        };

        const auto num_invocations = GenerationKernel::getNumInvocations(num_work_groups);
        auto candidates = static_cast<const Candidate*>(output_buffer->map(Buffer::AccessMode::read_only));
        if (!candidates)
            throw std::runtime_error("failed to map output buffer");

        std::vector<glm::vec3> positions;
        positions.reserve(num_invocations.x * num_invocations.y);

        for (int i = 0; i < num_invocations.x * num_invocations.y; i++)
        {
            const auto candidate = candidates[i];
            if (candidate.index > 0)
                positions.push_back(candidate.position);
        }

        output_buffer->unmap();

        return positions;
    }
} // placement