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

        /// The texture unit the heightmap sampler uniform is bound to by default.
        static constexpr int s_default_heightmap_tex_unit = 0;

        /// Get a proxy object to configure the heightmap texture sampler.
        [[nodiscard]] auto getHeightmapSampler() const -> TextureSampler {return {*this, m_heightmap_loc};}

        /// The texture unit the densitymap sampler uniform is bound to by default.
        static constexpr int s_default_densitymap_tex_unit = 1;

        /// Get a proxy object to configure the densitymap texture sampler.
        [[nodiscard]] auto getDensitymapSampler() const -> TextureSampler {return {*this, m_densitymap_loc};}

        /// The work group size of this kernel.
        static constexpr glm::uvec2 work_group_size {8, 8};

        /**
         * @brief Calculate the number of work groups required to cover the given area.
         * The number of invocations and, consequently, the number of candidates generated, can be obtained by
         * multiplying this value with work_group_size.
         * @param footprint Placement footprint.
         * @param lower_bound Lower bound of the placement area.
         * @param upper_bound Upper bound of the placement area.
         * @return The required number of work groups in each dimension.
         * @see setArgs()
         */
        [[nodiscard]]
        static auto calculateNumWorkGroups(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound) -> glm::uvec2;

        /// Set input parameters. @return the number of workgroups required for this set of arguments.
        glm::uvec2 setArgs(const glm::vec3& world_scale, float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound) const;

        /// Execute the kernel.
        void dispatchCompute(glm::uvec2 num_work_groups) const;

        /**
         * @brief Execute the generation kernel.
         * This function requires that the index buffer, position buffer, height texture and density texture be correctly
         * bound.
         * @param footprint the collision radius for placed objects
         * @param world_scale dimensions of the world
         * @param lower_bound lower corner of the placement area
         * @param upper_bound upper corner of the placement area
         * @return the number of work groups executed.
         */
        auto operator() (glm::vec3 world_scale, float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound) const
            -> glm::uvec2;

    private:
        static const std::string source_string;

        UniformLocation m_heightmap_loc;
        UniformLocation m_densitymap_loc;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
