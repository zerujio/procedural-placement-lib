#ifndef PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP

#include "glutils/program.hpp"
#include "glutils/guard.hpp"

#include "glm/glm.hpp"

namespace placement {

    /// Wrapper for the candidate generation compute shader.
    class GenerationKernel
    {
    public:
        /// Set the lower bound uniform.
        void setLowerBound(glm::vec2 lower_bound) const;

        /// Set the upper bound uniform.
        void setUpperBound(glm::vec2 upper_bound) const;

        /// Set the world scale uniform.
        void setWorldScale(glm::vec3 world_scale) const;

        /// Set the index offset uniform.
        void setIndexOffset(glm::uvec2 index_offset) const;

        /**
         * @brief set the footprint uniform.
         * @param radius footprint radius in world space.
         */
        void setFootprint(float radius) const;

        auto getHeightTextureBindingPoint() const -> int;

        auto getDensityTextureBindingPoint() const -> int;

        /// Execute the kernel with the currently set uniform values.
        void dispatchCompute(uint x, uint y) const;

        void dispatchCompute(glm::uvec2 workgroups) const
        {
            dispatchCompute(workgroups.x, workgroups.y);
        }

    private:
        glutils::Guard<glutils::Program> m_program;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
