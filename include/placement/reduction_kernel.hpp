#ifndef PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP

#include "placement_pipeline_kernel.hpp"

#include "glutils/guard.hpp"
#include "glutils/program.hpp"
#include "glutils/buffer.hpp"
#include "glutils/gl_types.hpp"

#include "glm/vec2.hpp"

namespace placement {

    class ReductionKernel : public PlacementPipelineKernel
    {
        using PlacementPipelineKernel::PlacementPipelineKernel;

    public:
        [[nodiscard]]
        GLuint getIndexBufferBindingIndex() const
        {
            return m_index_buffer_ssb.getBindingIndex();
        }

        void setIndexBufferBindingIndex(GLuint index)
        {
            m_index_buffer_ssb.setBindingIndex(*this, index);
        }

        [[nodiscard]]
        static GLsizeiptr calculateIndexBufferSize(GLsizeiptr element_count)
        {
            return (element_count + 1) * static_cast<GLsizeiptr>(sizeof(unsigned int));
        }

    protected:
        static constexpr auto s_index_ssb_name = "IndexBuffer";
        ShaderStorageBlock m_index_buffer_ssb {*this, s_index_ssb_name};
    };

    class IndexAssignmentKernel final : public ReductionKernel
    {
    public:
        IndexAssignmentKernel();

        /**
         * @brief Execute this compute kernel, assigning a unique zero-based index to each valid candidate.
         * @param candidate_count The number of candidates. That is, the exact number of elements in the position buffer.
         */
        void dispatchCompute(std::size_t candidate_count) const;

    private:
        static constexpr unsigned int s_work_group_size = 32;

        static const std::string s_source_string;

        /// determine the required number of workgroups given the number of elements.
        [[nodiscard]]
        static auto m_calculateNumWorkGroups(std::size_t element_count) -> std::size_t;
    };

    class IndexedCopyKernel final : public ReductionKernel
    {
    public:
        IndexedCopyKernel();

        /// Execute this kernel, copying all valid positions in the candidate buffer to the position buffer.
        void dispatchCompute(std::size_t candidate_count) const;

        [[nodiscard]]
        GLuint getPositionBufferBindingIndex() const
        {
            return m_position_ssb.getBindingIndex();
        }

        void setPositionBufferBindingIndex(GLuint index)
        {
            m_position_ssb.setBindingIndex(*this, index);
        }

        /// position buffer is an array of vec3, but with vec4 alignment
        static GLsizeiptr calculatePositionBufferSize(GLsizeiptr reduced_count)
        {
            return reduced_count * static_cast<GLsizeiptr>(sizeof(glm::vec4));
        }

    private:
        static const std::string s_source_string;
        static constexpr auto s_position_ssb_name = "PositionBuffer";
        static constexpr auto s_workgroup_size = 64;
        ShaderStorageBlock m_position_ssb {*this, s_position_ssb_name};
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
