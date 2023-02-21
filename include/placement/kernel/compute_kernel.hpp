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

    class [[deprecated]] InterfaceBlockBase
    {
    public:
        /**
         * @brief get the current binding index of this interface block.
         * This value is cached. If the binding is modified by any means other than InterfaceBlock::setBindingIndex()
         * the value returned by this function will become outdated. To update it, call queryBindingIndex().
         * @return the current binding index.
         */
        [[nodiscard]]
        auto getBindingIndex() const -> GLuint
        { return m_binding_index; }

    protected:
        using Type = GL::ProgramHandle::Interface;

        explicit InterfaceBlockBase(const ComputeKernel &kernel, ProgramResourceIndexBase resource_index, Type type)
                : m_resource_index(resource_index)
        {
            m_queryBindingIndex(kernel, type);
        }

        auto m_queryBindingIndex(const ComputeKernel &kernel, Type type) -> GLuint;

        ProgramResourceIndexBase m_resource_index;
        GLuint m_binding_index{0};
    };

    /// Lightweight object for manipulating uniform and shader storage interface block binding points.
    template<GL::ProgramHandle::Interface InterfaceType>
    class [[deprecated]] InterfaceBlock : public InterfaceBlockBase
    {
        static_assert(InterfaceType == Type::uniform_block || InterfaceType == Type::shader_storage_block,
                      "An interface block must be either a uniform block or a shader storage block");
    public:
        InterfaceBlock(const ComputeKernel &kernel, ProgramResourceIndex<InterfaceType> resource_index) :
                InterfaceBlockBase(kernel, resource_index, InterfaceType)
        {}

        /// Construct from the name of the interface block.
        InterfaceBlock(const ComputeKernel &kernel, const char *name) : InterfaceBlock(kernel, {kernel, name})
        {}

        /// Change the binding point.
        void setBindingIndex(const ComputeKernel &kernel, GLuint index)
        {
            if constexpr (InterfaceType == Type::uniform_block)
                kernel.m_program.setUniformBlockBinding(m_resource_index.get(), index);
            else if constexpr (InterfaceType == Type::shader_storage_block)
                kernel.m_program.setShaderStorageBlockBinding(m_resource_index.get(), index);

            m_binding_index = index;
        }
    };

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

    /// Lightweight object for setting and querying the texture unit of a sampler uniform
    class [[deprecated]] TextureSampler
    {
    public:
        TextureSampler(const ComputeKernel &kernel, UniformLocation location) : m_location(location)
        {
            queryTextureUnit(kernel);
        }

        TextureSampler(const ComputeKernel &kernel, const char *name)
                : TextureSampler(kernel, kernel.m_getUniformLocation(name))
        {}

        /// Set the texture unit for this sampler
        void setTextureUnit(const ComputeKernel &kernel, GLuint index);

        /**
         * @brief Get the cached texture unit index.
         * If the texture unit binding has been changed in any way other than calling setTextureUnit() on this
         * instance, then the value returned by this function will be outdated. To update it, call queryTextureUnit().
         * @return the current texture unit
         */
        [[nodiscard]]
        auto getTextureUnit() const -> GLuint
        { return m_tex_unit; }

        /// Query the GL for the current texture unit index (and update the cached value).
        auto queryTextureUnit(const ComputeKernel &kernel) -> GLuint;

    private:
        UniformLocation m_location;
        GLuint m_tex_unit{0};
    };

private:
    GL::Program m_program;
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_COMPUTE_KERNEL_HPP
