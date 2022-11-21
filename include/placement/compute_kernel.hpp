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
    protected:
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
        void m_useProgram() const;

        // Various utility classes

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

        class InterfaceBlockBase
        {
        public:
            /**
             * @brief get the current binding index of this interface block.
             * This value is cached. If the binding is modified by any means other than InterfaceBlock::setBindingIndex()
             * the value returned by this function will become outdated. To update it, call queryBindingIndex().
             * @return the current binding index.
             */
            [[nodiscard]]
            auto getBindingIndex() const -> glutils::GLuint { return m_binding_index; }

        protected:
            using Type = glutils::Program::Interface;

            explicit InterfaceBlockBase(const ComputeKernel& kernel, ProgramResourceIndexBase resource_index, Type type)
            : m_resource_index(resource_index)
            {
                m_queryBindingIndex(kernel, type);
            }

            auto m_queryBindingIndex(const ComputeKernel& kernel, Type type) -> glutils::GLuint;

            ProgramResourceIndexBase m_resource_index;
            glutils::GLuint m_binding_index {0};
        };

        /// Lightweight object for manipulating uniform and shader storage interface block binding points.
        template<glutils::Program::Interface InterfaceType>
        class InterfaceBlock : public InterfaceBlockBase
        {
            static_assert(InterfaceType == Type::uniform_block || InterfaceType == Type::shader_storage_block,
                          "An interface block must be either a uniform block or a shader storage block");
        public:
            InterfaceBlock(const ComputeKernel& kernel, ProgramResourceIndex<InterfaceType> resource_index) :
                    InterfaceBlockBase(kernel, resource_index, InterfaceType)
            {}

            /// Construct from the name of the interface block.
            InterfaceBlock(const ComputeKernel& kernel, const char* name) : InterfaceBlock(kernel, {kernel, name}) {}

            /// Change the binding point.
            void setBindingIndex(const ComputeKernel& kernel, glutils::GLuint index)
            {
                if constexpr (InterfaceType == Type::uniform_block)
                    kernel.m_program->setUniformBlockBinding(m_resource_index.get(), index);
                else if constexpr (InterfaceType == Type::shader_storage_block)
                    kernel.m_program->setShaderStorageBlockBinding(m_resource_index.get(), index);

                m_binding_index = index;
            }
        };

        using UniformBlock = InterfaceBlock<glutils::Program::Interface::uniform_block>;
        using ShaderStorageBlock = InterfaceBlock<glutils::Program::Interface::shader_storage_block>;

        /// Queries and stores the location of a uniform.
        struct UniformLocation
        {
        public:
            UniformLocation(const ComputeKernel& kernel, const char* uniform_name);

            [[nodiscard]] auto get() const -> glutils::GLint {return m_location;}

            [[nodiscard]] auto isValid() const -> bool {return m_location >= 0;}

            explicit operator bool() const {return isValid();}

        private:
            glutils::GLint m_location {-1};
        };


        /// Lightweight object for setting and querying the texture unit of a sampler uniform
        class TextureSampler
        {
        public:
            TextureSampler(const ComputeKernel& kernel, UniformLocation location) : m_location(location)
            {
                queryTextureUnit(kernel);
            }

            TextureSampler(const ComputeKernel& kernel, const char* name)
            : TextureSampler(kernel, UniformLocation(kernel, name)) {}

            /// Set the texture unit for this sampler
            void setTextureUnit(const ComputeKernel& kernel, glutils::GLuint index);

            /**
             * @brief Get the cached texture unit index.
             * If the texture unit binding has been changed in any way other than calling setTextureUnit() on this
             * instance, then the value returned by this function will be outdated. To update it, call queryTextureUnit().
             * @return the current texture unit
             */
            [[nodiscard]]
            auto getTextureUnit() const -> glutils::GLuint { return m_tex_unit; }

            /// Query the GL for the current texture unit index (and update the cached value).
            auto queryTextureUnit(const ComputeKernel& kernel) -> glutils::GLuint;
        private:
            UniformLocation m_location;
            glutils::GLuint m_tex_unit {0};
        };

        template<typename T>
        void setUniform(glutils::GLint location, T value) const
        {
            m_program->setUniform(location, value);
        }

        template<typename T>
        void setUniform(UniformLocation location, T value) const
        {
            setUniform(location.get(), value);
        }

        template<typename T>
        void setUniform(glutils::GLint location, glutils::GLsizei count, const T* values) const
        {
            m_program->setUniform(location, count, values);
        }

        template<typename T>
        void setUniform(UniformLocation location, glutils::GLsizei count, const T* values) const
        {
            setUniform(location.get(), count, values);
        }

        template<typename T>
        void setUniformMatrix(glutils::GLint location, glutils::GLsizei count, glutils::GLboolean transpose,
                              const T* values) const
        {
            m_program->setUniformMatrix(location, count, transpose, values);
        }

        template<typename T>
        void setUniformMatrix(UniformLocation location, glutils::GLsizei count, glutils::GLboolean transpose,
                              const T* values) const
        {
            setUniformMatrix(location.get(), count, transpose, values);
        }

    private:
        glutils::Guard<glutils::Program> m_program;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP
