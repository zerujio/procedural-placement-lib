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
        static constexpr glutils::GLuint default_ssb_binding = 0;

        /// Access the shader storage block for the candidate buffer.
        [[nodiscard]]
        auto getShaderStorageBlock() const -> ShaderStorageBlock {return {*this, m_ssb};}

        /// An struct with identical layout and alignment to the elements of the candidate buffer.
        struct Candidate
        {
            glm::vec3 position;
            unsigned int index;
        };

    protected:
        static constexpr auto s_ssb_name = "CandidateBuffer";

    private:
        using ShaderStorageBlockIndex = ProgramResourceIndex<glutils::Program::Interface::shader_storage_block>;
        ShaderStorageBlockIndex m_ssb {*this, s_ssb_name};
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_KERNEL_HPP
