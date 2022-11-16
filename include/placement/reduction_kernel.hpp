#ifndef PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP

#include "compute_kernel.hpp"

#include "glutils/guard.hpp"
#include "glutils/program.hpp"
#include "glutils/buffer.hpp"
#include "glutils/gl_types.hpp"

#include "glm/vec2.hpp"

namespace placement {

    // TODO: rewrite this to solve the workgroup ordering problem with an atomically incremented variable:
    /*
     * IndexReductionKernel:
     *  buffer CandidateIndexBuffer
     *  {
     *      uint index_count;
     *      uint index_buffer[]; // alignment is as expected
     *  };
     *
     *  shared uint index_offset;
     *  shared uint indices[gl_WorkGroupSize * 2];
     *
     *  void main()
     *  {
     *      // copy to shared memory
     *      indices[... first] = index_buffer[...];
     *      indices[... second] = index_buffer[...];
     *
     *      for (uint group_size = 1; group_size < gl_WorkGroupSize.x; group_size <<= 1)
     *      {
     *          barrier();
     *          memoryBarrierShared();
     *
     *          ... // same code
     *          indices[write_index] += indices[read_index];
     *      }
     *
     *      if (gl_LocalInvocationID.x == gl_WorkGroupSize.x - 1)
     *          index_offset = atomicAdd(index_count, indices[gl_WokGroupSize.x * 2 - 1]);
     *
     *      barrier();
     *      memoryBarrierShared();
     *
     *      // copy to buffer from shared memory
     *      index_buffer[... first] = indices[...] + index_offset;
     *      index_buffer[... second] = indices[...] + index_offset;
     *  }
     *
     * CopyKernel:
     *  buffer IndexBuffer
     *  {
     *      uint count;
     *      uint indices[];
     *  };
     *
     *  buffer From
     *  {
     *      vec3 from[];
     *  };
     *
     *  buffer To
     *  {
     *      vec3 to[];
     *  };
     *
     *  void main()
     *  {
     *      // check validity of indices
     *      to[indices[gl_GlobalInvocationID.x]] = from[gl_GlobalInvocationID.x];
     *  }
     */

    /// Reduces the position buffer, discarding invalid positions.
    class ReductionKernel final : public PlacementPipelineKernel
    {
    public:
        ReductionKernel();

        /**
         * @brief Execute this compute kernel, discarding all invalid elements in the position buffer.
         * All valid positions will be moved to adjacent locations within the position buffer, like an array. The array
         * begins at the start of the buffer, and the number of elements it contains is equal to the value of the last
         * element in the index buffer.
         * @param candidate_count The number of candidates. That is, the exact number of elements in the position and
         * index buffers.
         * @return The byte offset of the last element of the index buffer. This is measured from the start of the
         * memory range mapped to the index buffer binding point, which may or may not coincide with the start of the
         * buffer that contains it.
         */
        auto dispatchCompute(std::size_t candidate_count) const -> std::size_t;

        /// set/get the binding indices used for the auxiliary buffers, which must be different from the position and index buffers as well as each other.
        void setAuxiliaryBufferBindingIndices(glm::uvec2 indices);

        [[nodiscard]]
        auto getAuxiliaryBufferBindingIndices() const -> glm::uvec2;

    private:
        static const std::string s_source_string;

        /// Number of invocations per work group
        static constexpr unsigned int s_work_group_size = 64;

        /// determine the required number of workgroups given the number of elements.
        [[nodiscard]]
        static auto m_calculateNumWorkGroups(std::size_t element_count) -> std::size_t;

        class AuxiliaryKernel : public ComputeKernel
        {
        public:
            void setInputBindingIndex(glutils::GLuint i) {m_input_block.setBindingIndex(*this, i);}
            [[nodiscard]] auto getInputBindingIndex() const {return m_input_block.getBindingIndex();}
            void setOutputBindingIndex(glutils::GLuint i) {m_output_block.setBindingIndex(*this, i);}
            [[nodiscard]] auto getOutputBindingIndex() const {return m_output_block.getBindingIndex();}
        protected:
            explicit AuxiliaryKernel(std::string source_string);
        private:
            ShaderStorageBlock m_input_block;
            ShaderStorageBlock m_output_block;
        };

        class Forward final : public AuxiliaryKernel
        {
        public:
            Forward();

            void dispatchCompute(std::size_t index_count) const;

        private:
            static const std::string s_source_string;
        } m_forward_kernel;

        class Backward final : public AuxiliaryKernel
        {
            Backward();

            void dispatchCompute(std::size)
        private:
            static const std::string s_source_string;
        } m_backward_kernel;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_REDUCTION_KERNEL_HPP
