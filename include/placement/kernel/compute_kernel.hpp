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
    /// Equivalent to calling glUseProgram with this kernel's program name.
    void useProgram() const;

    static void dispatch(glm::uvec3 num_work_groups);

protected:

    /// Compile and link a compute shader from source code.
    explicit ComputeKernel(const char *source_string) : ComputeKernel(1, &source_string)
    {}

    explicit ComputeKernel(const std::string &source_string) : ComputeKernel(source_string.c_str())
    {}

    explicit ComputeKernel(const std::vector<const char *> &strings)
            : ComputeKernel(strings.size(), const_cast<const char **>(strings.data()))
    {}

    template<auto N>
    explicit ComputeKernel(const std::array<const char *, N> &strings)
            : ComputeKernel(strings.size, const_cast<const char **>(strings.data()))
    {}

    ComputeKernel(unsigned int count, const char **source_strings);

    // Various utility classes

    using Interface = GL::Program::Interface;

    [[nodiscard]]
    GLuint m_getResourceIndex(Interface interface, const char* name) const;

    template<Interface I>
    struct ResourceIndex
    {
        GLuint value;
    };

    template<Interface I>
    [[nodiscard]]
    ResourceIndex<I> m_getResourceIndex(const char* name) const
    {
        return {m_getResourceIndex(I, name)};
    }

    class [[deprecated("Use resource query functions.")]] ProgramResourceIndexBase
    {
    public:
        [[nodiscard]] auto get() const
        { return m_index; }

    protected:
        ProgramResourceIndexBase(const ComputeKernel &kernel, GL::ProgramHandle::Interface program_interface,
                                 const char *resource_name);

    private:
        GLuint m_index;
    };

    /// Queries and stores the index for some program resource.
    template<GL::ProgramHandle::Interface Interface>
    class [[deprecated]] ProgramResourceIndex : public ProgramResourceIndexBase
    {
    public:
        ProgramResourceIndex(const ComputeKernel &kernel, const char *resource_name) :
                ProgramResourceIndexBase(kernel, Interface, resource_name)
        {}
    };


    using UniformBlockIndex = ResourceIndex<Interface::uniform_block>;

    [[nodiscard]]
    UniformBlockIndex m_getUniformBlockIndex(const char* name) const
    {
        return m_getResourceIndex<Interface::uniform_block>(name);
    }

    void m_setUniformBlockBinding(UniformBlockIndex resource_index, uint binding_index) const
    {
        m_program.setUniformBlockBinding(resource_index.value, binding_index);
    }

    using ShaderStorageBlockIndex = ResourceIndex<Interface::shader_storage_block>;

    [[nodiscard]]
    ShaderStorageBlockIndex m_getShaderStorageBlockIndex(const char* name) const
    {
        return m_getResourceIndex<Interface::shader_storage_block>(name);
    }

    void m_setShaderStorageBlockBinding(ResourceIndex<Interface::shader_storage_block> resource_index, uint binding_index) const
    {
        m_program.setShaderStorageBlockBinding(resource_index.value, binding_index);
    }

    struct UniformLocation
    {
        GLint value {-1};

        [[nodiscard]] operator bool() const { return value > -1; }
    };

    [[nodiscard]]
    UniformLocation m_getUniformLocation(const char* name) const;

    template<typename T>
    void m_setUniform(GLint location, T value) const
    {
        m_program.setUniform(location, value);
    }

    template<typename T>
    void m_setUniform(UniformLocation location, T value) const
    {
        m_setUniform(location.value, value);
    }

    template<typename T>
    void m_setUniform(GLint location, GLsizei count, const T *values) const
    {
        m_program.setUniform(location, count, values);
    }

    template<typename T>
    void m_setUniform(UniformLocation location, GLsizei count, const T *values) const
    {
        m_setUniform(location.value, count, values);
    }

    template<typename T>
    void m_setUniformMatrix(GLint location, GLsizei count, GLboolean transpose,
                            const T *values) const
    {
        m_program.setUniformMatrix(location, count, transpose, values);
    }

    template<typename T>
    void m_setUniformMatrix(UniformLocation location, GLsizei count, GLboolean transpose,
                            const T *values) const
    {
        m_setUniformMatrix(location.value, count, transpose, values);
    }

private:
    GL::Program m_program;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP
