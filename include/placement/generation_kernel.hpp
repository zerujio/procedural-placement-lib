#ifndef PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP

#include "ssb_proxy.hpp"

#include "glutils/program.hpp"
#include "glutils/guard.hpp"

#include "glm/glm.hpp"

namespace placement {

    /// Wrapper for the candidate generation compute shader.
    class GenerationKernel
    {
    public:

        static constexpr int heightmap_tex_unit = 0;

        static constexpr int densitymap_tex_unit = 1;

        static constexpr int output_buffer_binding = 0;

        static constexpr glm::uvec2 work_group_size {4, 4};

        GenerationKernel();

        /// compute the required number of workgroups for a given area
        [[nodiscard]]
        static constexpr auto getNumWorkGroups(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound) -> glm::uvec2
        {
            const glm::uvec2 min_invocations {(upper_bound - lower_bound) / (2.0f * footprint)};
            return min_invocations / work_group_size + 1u;
        }

        [[nodiscard]]
        static constexpr auto getNumInvocations(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound) -> glm::uvec2
        {
            return getNumInvocations(getNumWorkGroups(footprint, lower_bound, upper_bound));
        }

        [[nodiscard]]
        static constexpr auto getNumInvocations(glm::uvec2 num_work_groups) -> glm::uvec2
        {
            return num_work_groups * work_group_size;
        }

        /// compute the minimum size for the output buffer, in bytes.
        [[nodiscard]]
        static constexpr auto getRequiredBufferSize(glm::uvec2 num_work_groups) -> glutils::GLsizeiptr
        {
            const auto num_elements = num_work_groups * work_group_size;
            return num_elements.x * num_elements.y * static_cast<glutils::GLsizeiptr>(sizeof(glm::vec4));
        }

        [[nodiscard]]
        static constexpr auto getMinRequiredBufferSize(float footprint, glm::vec2 lower_bound, glm::vec2 upper_bound) -> glutils::GLsizeiptr
        {
            return getRequiredBufferSize(getNumWorkGroups(footprint, lower_bound, upper_bound));
        }

        void dispatchCompute(float footprint, glm::vec3 world_scale, glm::vec2 lower_bound, glm::vec2 upper_bound) const;

    private:
        glutils::Guard<glutils::Program> m_program;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_GENERATION_KERNEL_HPP
