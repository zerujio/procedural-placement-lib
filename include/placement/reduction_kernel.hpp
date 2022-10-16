#ifndef PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP

#include "placement_pipeline_kernel.hpp"

#include "glutils/guard.hpp"
#include "glutils/program.hpp"
#include "glutils/buffer.hpp"
#include "glutils/gl_types.hpp"

namespace placement {

    /// Reduces the position buffer, discarding invalid positions.
    class ReductionKernel final : public PlacementPipelineKernel
    {
    public:
        ReductionKernel();

        /**
         * @brief execute the compute kernel with @p num_work_groups.
         * The reduced number of elements can be found in the last position of the index buffer. In order for the
         * changes made by this function to be visible, its necessary to insert a call to glMemoryBarrier with
         * GL_SHADER_STORAGE_BARRIER_BIT before reading the position or index buffers.
         *
         * The compute program will remain bound after this operation.
         *
         * @param num_work_groups the number of work groups to dispatch. The kernel requires half as many invocations as
         * there are elements in the candidate buffer. The buffer should be zero-padded to ensure that it contains
         * exactly 2 * num_work_groups * work_group_size elements. Behavior is undefined otherwise.
         * @return offset into the index buffer at which the reduced count (an unsigned int) is located.
         */
        auto operator() (std::size_t num_work_groups) const -> std::size_t;

        /// Number of invocations per work group
        static constexpr unsigned int work_group_size = 64;

    private:
        static const std::string source_string;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
