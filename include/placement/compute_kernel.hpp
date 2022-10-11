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
        ComputeKernel(const char* source_string) : ComputeKernel(1, &source_string) {}

        ComputeKernel(const std::string& source_string) : ComputeKernel(source_string.c_str()) {}

        ComputeKernel(const std::vector<const char*>& strings)
        : ComputeKernel(strings.size(), const_cast<const char**>(strings.data())) {}

        template<auto N>
        ComputeKernel(const std::array<const char*, N>& strings)
        : ComputeKernel(strings.size, const_cast<const char**>(strings.data())) {}

        ComputeKernel(unsigned int count, const char** source_strings);

        /// Equivalent to calling glUseProgram with this kernel's program name.
        void useProgram() const;

    protected:
        [[nodiscard]] auto m_getProgram() const {return *m_program;}

        enum class InterfaceBlockType : glutils::GLenum;

        class InterfaceBlock
        {
        protected:
            InterfaceBlock(const ComputeKernel& kernel, InterfaceBlockType type, const std::string& name);
            [[nodiscard]] auto m_getBinding(const ComputeKernel& kernel, InterfaceBlockType type) const -> glutils::GLuint;
            [[nodiscard]] auto m_getResourceIndex() const {return m_resource_index;}
        private:
            glutils::GLuint m_resource_index;
        };

        /// wrapper for a uniform interface block
        class UniformBlock final : public InterfaceBlock
        {
        public:
            UniformBlock(const ComputeKernel& kernel, const std::string& name);

            [[nodiscard]]
            auto getBinding(const ComputeKernel& kernel) const -> glutils::GLuint;

            void setBinding(const ComputeKernel& kernel, glutils::GLuint index) const;
        };

        /// wrapper for a shader storage interface block
        class ShaderStorageBlock final : public InterfaceBlock
        {
        public:
            ShaderStorageBlock(const ComputeKernel& kernel, const std::string& name);

            [[nodiscard]]
            auto getBinding(const ComputeKernel& kernel) const -> glutils::GLuint;

            void setBinding(const ComputeKernel& kernel, glutils::GLuint index) const;
        };

        /// wraps texture uniform configuration.
        class TextureSampler final
        {
        public:
            /// Construct from uniform name.
            TextureSampler(const ComputeKernel& kernel, const char* uniform_name);
            TextureSampler(const ComputeKernel& kernel, const std::string& uniform_name)
            : TextureSampler(kernel, uniform_name.c_str()) {}

            /// set the texture unit this sampler is bound to
            void setTextureUnit(const ComputeKernel& kernel, glutils::GLuint index) const;

            /// query the texture unit this sampler is currently bound to
            [[nodiscard]]
            auto getTextureUnit(const ComputeKernel& kernel) const -> glutils::GLuint;

        private:
            glutils::GLint m_location;
        };

    private:
        glutils::Guard<glutils::Program> m_program;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP
