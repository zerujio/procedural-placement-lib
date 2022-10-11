#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_KERNEL_HPP

#include "compute_kernel.hpp"

#include "glutils/buffer.hpp"

namespace placement {

    /// Base class for kernels that are part of a the placement pipeline.
    class PlacementPipelineKernel : public ComputeKernel
    {
    public:
        using ComputeKernel::ComputeKernel;

        static constexpr glutils::GLuint s_default_position_buffer_binding = 0;

        /// get/set the shader storage block binding for the position buffer.
        void setPositionBufferBinding(glutils::GLuint index)
        {
            m_position_ssb.setBinding(*this, index);
        }

        [[nodiscard]]
        auto getPositionBufferBinding() const -> glutils::GLuint
        {
            return m_position_ssb.getBinding(*this);
        }

        /// compute the expected size for a buffer range containing @p count positions.
        static auto computePositionBufferSize(std::size_t count) -> std::size_t
        {
            return count * 3 * sizeof(float);
        }

        /// Bind a buffer range as the position buffer.
        void bindPositionBuffer(glutils::BufferOffset buffer_offset, std::size_t count) const
        {
            buffer_offset.bindRange(glutils::Buffer::IndexedTarget::shader_storage,
                                    getPositionBufferBinding(),
                                    static_cast<glutils::GLsizeiptr>(computePositionBufferSize(count)));
        }


        static constexpr glutils::GLuint s_default_index_buffer_binding = 1;

        /// get/set the shader storage block binding for the index buffer;
        void setIndexBufferBinding(glutils::GLuint index)
        {
            m_index_ssb.setBinding(*this, index);
        }

        [[nodiscard]]
        auto getIndexBufferBinding() const -> glutils::GLuint
        {
            return m_index_ssb.getBinding(*this);
        }

        /// compute the expected size for a buffer range containing @p count indices.
        static auto computeIndexBufferSize(std::size_t count) -> std::size_t
        {
            return count * sizeof(unsigned int);
        }

        /// Bind a buffer range as the index buffer.
        void bindIndexBuffer(glutils::BufferOffset buffer_offset, std::size_t count) const
        {
            buffer_offset.bindRange(glutils::Buffer::IndexedTarget::shader_storage,
                                    getIndexBufferBinding(),
                                    static_cast<glutils::GLsizeiptr>(computeIndexBufferSize(count)));
        }

    protected:
        static constexpr auto s_position_ssb_name = "PositionBuffer";
        static constexpr auto s_index_ssb_name = "IndexBuffer";

    private:
        ShaderStorageBlock m_position_ssb {*this, s_position_ssb_name};
        ShaderStorageBlock m_index_ssb {*this, s_index_ssb_name};
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_KERNEL_HPP
