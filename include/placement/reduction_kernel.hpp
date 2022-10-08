#ifndef PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP

#include "ssb_proxy.hpp"

#include "glutils/guard.hpp"
#include "glutils/program.hpp"
#include "glutils/buffer.hpp"
#include "glutils/gl_types.hpp"

namespace placement {

    using glutils::GLuint;
    using glutils::GLintptr;
    using glutils::GLsizeiptr;

    class ReductionKernel
    {
    public:
        ReductionKernel();

        /**
         * @brief Invoke the reduction kernel on a pair of position and index buffers.
         * and must use std430 layout.
         * @param position_buffer Offset into a buffer containing at least @p count vec3 positions.
         * @param index_buffer Offset into a buffer containing at least @p count uint indices. The values must be either
         * 0, for invalid positions, or 1, for valid positions.
         * @param count Number of positions in the buffers.
         */
        std::size_t operator() (glutils::BufferOffset position_buffer, glutils::BufferOffset index_buffer,
                std::size_t count) const;

        /// the shader storage block binding for the position buffer.
        void setPositionBufferBinding(GLuint index);
        [[nodiscard]] GLuint getPositionBufferBinding() const;

        /// the shader storage block binding for the index buffer;
        void setIndexBufferBinding(GLuint index);
        [[nodiscard]] GLuint getIndexBufferBinding() const;

        /**
         * @brief execute the compute kernel for @p count elements.
         * The compute kernel requieres that a position buffer and an index buffer be bound in the appropriate shader
         * storage binding points (see set/getPositionBufferBinding() and set/getIndexBufferBinding()). The buffer
         * ranges bound must contain exactly @p elements each.
         *
         * The number of valid positions in the reduced position buffer can be found at index @p count - 1 of the index
         * buffer.
         *
         * The compute program will remain bound after this operation.
         *
         * @param count the number of elements contained in the position and index buffers.
         */
        void dispatchCompute(std::size_t count) const;

        /// Bind a buffer range as the position buffer.
        void bindPositionBuffer(glutils::BufferOffset buffer_offset, std::size_t count) const;

        /// Bind a buffer range as the index buffer.
        void bindIndexBuffer(glutils::BufferOffset buffer_offset, std::size_t count) const;

        /// compute the expected size for a buffer range containing @p count positions.
        static auto computePositionBufferSize(std::size_t count) { return count * 3 * sizeof(float); }

        /// compute the expected size for a buffer range containing @p count indices.
        static auto computeIndexBufferSize(std::size_t count) { return count * sizeof(unsigned int); }

    private:
        glutils::Guard<glutils::Program> m_program;

        ShaderStorageBlockInfo m_position_ssb;
        ShaderStorageBlockInfo m_index_ssb;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
