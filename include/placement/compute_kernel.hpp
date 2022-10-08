#ifndef PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP
#define PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP

#include "glutils/program.hpp"
#include "glutils/guard.hpp"

namespace placement {

    class ComputeKernel
    {
    public:
        ComputeKernel(const char** source_strings);

    protected:
        [[nodiscard]] auto m_getProgram() const {return *m_program;}

        enum class InterfaceBlockType : glutils::GLenum;

        class InterfaceBlock
        {
        public:
            InterfaceBlock(const ComputeKernel& kernel, InterfaceBlockType type, const std::string& name);

            [[nodiscard]] auto getBinding() const {return m_binding_index;}
            void setBinding(const ComputeKernel& kernel, glutils::GLuint index);
        protected:
            virtual void m_updateBinding(glutils::GLuint uint)
            [[nodiscard]] auto m_getResourceIndex() const {return m_resource_index;}
        private:
            using BindingSetter = void (*) (glutils::GLuint, glutils::GLuint, glutils::GLuint);
            GladGLContext::* m_setter;
            glutils::GLuint m_resource_index;
            glutils::GLuint m_binding_index;
        };

        class UniformBlock final : public InterfaceBlock
        {
        public:
            UniformBlock(const ComputeKernel& kernel, const std::string& name);
            void setBinding(const ComputeKernel& kernel, glutils::GLuint index) override;
        };

        class ShaderStorageBlock final : public InterfaceBlock
        {
        public:
            ShaderStorageBlock(const ComputeKernel& kernel, const std::string& name);
            void setBinding(const ComputeKernel& kernel, glutils::GLuint index) override;
        };

    private:
        glutils::Guard<glutils::Program> m_program;

    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP
