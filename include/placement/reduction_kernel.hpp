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
         * @brief Invoke the reduction kernel on a pair of position and index buffers.
         * and must use std430 layout.
         * @param position_buffer Offset into a buffer containing at least @p count vec3 positions.
         * @param index_buffer Offset into a buffer containing at least @p count uint indices. The values must be either
         * 0, for invalid positions, or 1, for valid positions.
         * @param count Number of elements in the reduced position buffer.
         */
        auto operator() (glutils::BufferOffset position_buffer, glutils::BufferOffset index_buffer,
                std::size_t count) const -> std::size_t;

        /**
         * @brief execute the compute kernel for @p count elements.
         * The number of valid positions in the reduced position buffer can be found at index @p count - 1 of the index
         * buffer. In order for the changes made by this function to be visible, its necessary to insert a call to
         * glMemoryBarrier with GL_SHADER_STORAGE_BARRIER_BIT before reading the position or index buffers.
         *
         * The compute program will remain bound after this operation.
         * @param count the number of elements contained in the position and index buffers.
         * @return offset into the index buffer at which the reduced count (an unsigned int) is located.
         */
        [[nodiscard]]
        auto dispatchCompute(std::size_t count) const -> std::size_t;

    private:
        static const std::string source_string;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
