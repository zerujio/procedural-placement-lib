#ifndef PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP

#include "glutils/program.hpp"
#include "glutils/guard.hpp"
#include "glutils/gl_types.hpp"

#include <vector>
#include <array>

namespace placement {

    class ComputeKernel
    {
    public:
        /// Compile and link a compute shader from source code.
        explicit ComputeKernel(const char* source_string) : ComputeKernel(1, &source_string) {}

        explicit ComputeKernel(const std::string& source_string) : ComputeKernel(source_string.c_str()) {}

        explicit ComputeKernel(const std::vector<const char*>& strings)
        : ComputeKernel(strings.size(), const_cast<const char**>(strings.data())) {}

        template<auto N>
        explicit ComputeKernel(const std::array<const char*, N>& strings)
        : ComputeKernel(strings.size, const_cast<const char**>(strings.data())) {}

        ComputeKernel(unsigned int count, const char** source_strings);

        /// Equivalent to calling glUseProgram with this kernel's program name.
        void useProgram() const;

        // Various utility classes
    protected:

        class ProgramResourceIndexBase
        {
        public:
            [[nodiscard]] auto get() const {return m_index;}

        protected:
            ProgramResourceIndexBase(const ComputeKernel& kernel, glutils::Program::Interface program_interface,
                                     const char* resource_name);

        private:
            glutils::GLuint m_index;
        };

        /// Queries and stores the index for some program resource.
        template<glutils::Program::Interface Interface>
        class ProgramResourceIndex : public ProgramResourceIndexBase
        {
        public:
            ProgramResourceIndex(const ComputeKernel& kernel, const char* resource_name) :
                ProgramResourceIndexBase(kernel, Interface, resource_name) {}
        };

        /// Queries and stores the location of a uniform.
        struct UniformLocation
        {
        public:
            UniformLocation(const ComputeKernel& kernel, const char* uniform_name);

            [[nodiscard]] auto get() const -> glutils::GLint {return m_location;}

            [[nodiscard]] auto isValid() const -> bool {return m_location >= 0;}

            operator bool() const {return isValid();}

        private:
            glutils::GLint m_location;
        };

    public:

        // These are various utility classes

        /// Lightweight object for manipulating uniform and shader storage interface block binding points.
        class InterfaceBlockBase
        {
        protected:
            using Type = glutils::Program::Interface;

            InterfaceBlockBase(const ComputeKernel& kernel, glutils::GLuint resource_index)
            : m_program(*kernel.m_program), m_block_index(resource_index) {}

            [[nodiscard]] auto m_getBindingIndex(Type type) const -> glutils::GLuint;

            glutils::Program m_program;
            glutils::GLuint m_block_index;
        };

        template<glutils::Program::Interface InterfaceType>
        class InterfaceBlock : public InterfaceBlockBase
        {
            static_assert(InterfaceType == Type::uniform_block || InterfaceType == Type::shader_storage_block,
                    "An interface block must be either a uniform block or a shader storage block");
        public:
            InterfaceBlock(const ComputeKernel& kernel, ProgramResourceIndex<InterfaceType> resource_index) :
                InterfaceBlockBase(kernel, resource_index.get())
            {}

            /// Query (from the GL) the index of the current binding point.
            [[nodiscard]] auto getBindingIndex() const -> glutils::GLuint {return m_getBindingIndex(InterfaceType);}

            /// Change the binding point.
            void setBindingIndex(glutils::GLuint index) const
            {
                if constexpr (InterfaceType == Type::uniform_block)
                    m_program.setUniformBlockBinding(m_block_index, index);
                else if constexpr (InterfaceType == Type::shader_storage_block)
                    m_program.setShaderStorageBlockBinding(m_block_index, index);
            }
        };

        using UniformBlock = InterfaceBlock<glutils::Program::Interface::uniform_block>;
        using ShaderStorageBlock = InterfaceBlock<glutils::Program::Interface::shader_storage_block>;


        /// Lightweight for setting and querying the texture unit of a sampler uniform
        class TextureSampler
        {
        public:
            TextureSampler(const ComputeKernel& kernel, UniformLocation location)
            : m_program(*kernel.m_program), m_location(location.get()) {}

            /// Set the texture unit for this sampler
            void setTextureUnit(glutils::GLuint index) const;

            /**
             * @brief Query the texture unit this sampler is currently bound to.
             * Note that this performs a "glGet*" call, which may cause synchronization (i.e. a stall) of the GL pipeline.
             */
            [[nodiscard]] auto getTextureUnit() const -> glutils::GLuint;

        private:
            glutils::Program m_program;
            glutils::GLint m_location;
        };

    protected:

        template<typename T>
        void setUniform(glutils::GLint location, T value) const
        {
            m_program->setUniform(location, value);
        }

        template<typename T>
        void setUniform(glutils::GLint location, glutils::GLsizei count, const T* values) const
        {
            m_program->setUniform(location, count, values);
        }

        template<typename T>
        void setUniformMatrix(glutils::GLint location, glutils::GLsizei count, glutils::GLboolean transpose,
                              const T* values) const
        {
            m_program->setUniformMatrix(location, count, transpose, values);
        }

    private:
        glutils::Guard<glutils::Program> m_program;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP
