#ifndef PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP

#include "placement_pipeline_kernel.hpp"

#include "glutils/buffer.hpp"
#include "glutils/program.hpp"
#include "glutils/guard.hpp"

#include "glm/glm.hpp"

namespace placement {

    /// Wrapper for the candidate position generation compute shader.
    class GenerationKernel final : public PlacementPipelineKernel
    {
    public:
        GenerationKernel();

        /// The texture unit the heightmap uniform is bound to by default.
        static constexpr int s_default_heightmap_tex_unit = 0;

        /// Set the texture unit the heightmap sampler uniform is bound to.
        void setHeightmapTexUnit(glutils::GLuint index) const
        {
            m_heightmap_tex.setTextureUnit(*this, index);
        }

        /// Query from the GL the texture unit the heightmap sampler uniform is currently bound to.
        [[nodiscard]]
        auto getHeightmapTexUnit() const -> glutils::GLuint
        {
            return m_heightmap_tex.getTextureUnit(*this);
        }

        /// The texture unit the densitymap uniform is bound to by default.
        static constexpr int s_default_densitymap_tex_unit = 1;

        /// Set the texture unit the densitymap sampler uniform is bound to.
        void setDensitymapTexUnit(glutils::GLuint index) const
        {
            m_densitymap_tex.setTextureUnit(*this, index);
        }

        /// Query the texture unit the densitymap sampler uniform is currently bound to.
        [[nodiscard]]
        auto getDensitymapTexUnit() const -> glutils::GLuint
        {
            return m_densitymap_tex.getTextureUnit(*this);
        }

        /// The work group size of this kernel.
        static constexpr glm::uvec2 work_group_size {4, 4};

        /// compute the required number of workgroups for a given placement area and footprint
        [[nodiscard]]
        static constexpr auto computeNumWorkGroups(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound)
        -> glm::uvec2
        {
            const glm::uvec2 min_invocations {(upper_bound - lower_bound) / (2.0f * footprint)};
            return min_invocations / work_group_size + 1u;
        }

        /// compute the number of invocations the execution of the kernel with the given parameters will result in.
        [[nodiscard]]
        static constexpr auto computeNumInvocations(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound)
        -> glm::uvec2
        {
            return computeNumInvocations(computeNumWorkGroups(footprint, lower_bound, upper_bound));
        }

        [[nodiscard]]
        static constexpr auto computeNumInvocations(glm::uvec2 num_work_groups) -> glm::uvec2
        {
            return num_work_groups * work_group_size;
        }

        /**
         * @brief Execute the generation kernel.
         * This function requires that the index buffer, position bufer, height texture and density texture be correctly
         * bound.
         * @param footprint the collision radius for placed objects
         * @param world_scale dimensions of the world
         * @param lower_bound lower corner of the placement area
         * @param upper_bound upper corner of the placement area
         * @return the number of work groups executed.
         */
        auto dispatchCompute(float footprint, glm::vec3 world_scale, glm::vec2 lower_bound, glm::vec2 upper_bound) const
            -> glm::uvec2;

    private:
        static const std::string source_string;

        TextureSampler m_heightmap_tex;
        TextureSampler m_densitymap_tex;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
