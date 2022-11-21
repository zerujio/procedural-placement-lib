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

        [[nodiscard]]
        glutils::GLuint getCandidateBufferBindingIndex() const {return m_candidate_ssb.getBindingIndex();}

        void setCandidateBufferBindingIndex(glutils::GLuint index) {m_candidate_ssb.setBindingIndex(*this, index);}

        static glutils::GLsizeiptr calculateCandidateBufferSize(glutils::GLsizeiptr element_count)
        {
            return element_count * static_cast<glutils::GLsizeiptr>(sizeof(glm::vec4));
        }

    protected:
        static constexpr auto s_candidate_ssb_name = "CandidateBuffer";

    private:
        ShaderStorageBlock m_candidate_ssb {*this, s_candidate_ssb_name};
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_PIPELINE_KERNEL_HPP
