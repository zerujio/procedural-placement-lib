#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_KERNEL_HPP

#include "compute_kernel.hpp"

#include "glutils/buffer.hpp"

#include "glm/vec3.hpp"

namespace placement {

    /**
     * @brief Base class for compute kernels in the procedural placement pipeline.
     * All such kernels must define a shader storage block with the name indicated by PlacementPipelinKernel::s_ssb_name.
     * The block must contain a single member: and unbound array of candidates or arrays of candidates, each of which is
     * defined as a struct containing a vec3 and a uint.
     */
    class PlacementPipelineKernel : public ComputeKernel
    {
    public:
        using ComputeKernel::ComputeKernel;

        /// get the position buffer's binding index (used in glBindBufferBase and glBindBufferRange)
        [[nodiscard]]
        auto getPositionBufferBindingIndex() const -> glutils::GLuint { return m_position_ssb.getBindingIndex(); }

        /// get the index buffer's binding index (used in glBindBufferBase and glBindBufferRange)
        [[nodiscard]]
        auto getIndexBufferBindingIndex() const -> glutils::GLuint { return m_index_ssb.getBindingIndex(); }

        /// Set the position buffer's binding index, which must be different to the index buffer.
        void setPositionBufferBindingIndex(glutils::GLuint new_index) {m_position_ssb.setBindingIndex(*this, new_index); }

        /// set the index buffer's binding index, which must be different from the position buffer's index.
        void setIndexBufferBindingIndex(glutils::GLuint new_index) {m_index_ssb.setBindingIndex(*this, new_index);}

        /// Get the required size of a shader storage block for a given number of elements.
        [[nodiscard]]
        static auto calculatePositionBufferSize(std::size_t element_count) -> std::size_t
        {
            // alignment of vec3 is that of vec4 in shader storage blocks
            return element_count * sizeof(glm::vec4);
        }

        [[nodiscard]]
        static auto calculateIndexBufferSize(std::size_t element_count) -> std::size_t
        {
            return element_count * sizeof(glutils::GLuint);
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
