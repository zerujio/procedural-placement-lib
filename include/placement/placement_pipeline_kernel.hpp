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

        /// Default shader storage block binding index
        static constexpr glutils::GLuint default_position_ssb_binding = 0;
        static constexpr glutils::GLuint default_index_ssb_binding = 1;

        /// Access the shader storage block for the buffers.
        [[nodiscard]]
        auto getPositionShaderStorageBlock() const -> ShaderStorageBlock {return {*this, m_position_ssb};}

        [[nodiscard]]
        auto getIndexShaderStorageBlock() const -> ShaderStorageBlock {return {*this, m_index_ssb};}

        /// Get the required size of a shader storage block for a given number of elements.
        [[nodiscard]]
        static auto getPositionBufferRequiredSize(glutils::GLsizeiptr element_count) -> glutils::GLsizeiptr
        {
            // alignment of vec3 is that of vec4 in shader storage blocks
            return element_count * static_cast<glutils::GLsizeiptr>(sizeof(glm::vec4));
        }

        [[nodiscard]]
        static auto getIndexBufferRequiredSize(glutils::GLsizeiptr element_count) -> glutils::GLsizeiptr
        {
            return element_count * static_cast<glutils::GLsizeiptr>(sizeof(glutils::GLuint));
        }

    protected:
        static constexpr auto s_position_ssb_name = "PositionBuffer";
        static constexpr auto s_index_ssb_name = "IndexBuffer";

    private:
        using ShaderStorageBlockIndex = ProgramResourceIndex<glutils::Program::Interface::shader_storage_block>;

        ShaderStorageBlockIndex m_position_ssb {*this, s_position_ssb_name};
        ShaderStorageBlockIndex m_index_ssb {*this, s_index_ssb_name};
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_KERNEL_HPP
