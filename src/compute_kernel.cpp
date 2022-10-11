#include "placement/compute_kernel.hpp"

#include "glutils/shader.hpp"
#include "glutils/error.hpp"
#include "gl_context.hpp"

namespace placement {

    // =============================================== InterfaceBlock =============================================== //
    enum class ComputeKernel::InterfaceBlockType : GLenum
    {
        uniform         = GL_UNIFORM_BLOCK,
        shader_storage  = GL_SHADER_STORAGE_BLOCK
    };

    ComputeKernel::InterfaceBlock::InterfaceBlock(const ComputeKernel &kernel, InterfaceBlockType type,
                                                  const std::string &name)
    {
        // resource index
        m_resource_index = kernel.m_program->getResourceIndex(static_cast<GLenum>(type), name.c_str());
        if (m_resource_index == GL_INVALID_INDEX)
            throw glutils::GLError("No interface block named " + name);
    }

    auto ComputeKernel::InterfaceBlock::m_getBinding(const ComputeKernel& kernel, InterfaceBlockType type) const -> GLuint
    {
        // binding index
        const GLenum prop = GL_BUFFER_BINDING;
        int binding = -1;
        kernel.m_program->getResource(static_cast<GLenum>(type), m_resource_index, 1, &prop, 1, nullptr, &binding);
        if (binding < 0)
            throw glutils::GLError("Invalid interface block buffer binding index");
        return binding;
    }


    // ================================================ UniformBlock ================================================ //

    ComputeKernel::UniformBlock::UniformBlock(const ComputeKernel &kernel, const std::string &name) :
            InterfaceBlock(kernel, InterfaceBlockType::uniform, name)
    {}

    auto ComputeKernel::UniformBlock::getBinding(const ComputeKernel &kernel) const -> GLuint
    {
        return m_getBinding(kernel, InterfaceBlockType::uniform);
    }

    void ComputeKernel::UniformBlock::setBinding(const ComputeKernel &kernel, glutils::GLuint index) const
    {
        kernel.m_program->setUniformBlockBinding(m_getResourceIndex(), index);
    }


    // ============================================= ShaderStorageBlock ============================================= //

    ComputeKernel::ShaderStorageBlock::ShaderStorageBlock(const ComputeKernel &kernel, const std::string &name) :
            InterfaceBlock(kernel, InterfaceBlockType::shader_storage, name)
    {}

    auto ComputeKernel::ShaderStorageBlock::getBinding(const ComputeKernel &kernel) const -> GLuint
    {
        return m_getBinding(kernel, InterfaceBlockType::shader_storage);
    }

    void ComputeKernel::ShaderStorageBlock::setBinding(const ComputeKernel &kernel, glutils::GLuint index) const
    {
        kernel.m_program->setShaderStorageBlockBinding(m_getResourceIndex(), index);
    }


    // =============================================== TextureSampler =============================================== //

    ComputeKernel::TextureSampler::TextureSampler(const ComputeKernel &kernel, const char *uniform_name)
    {
        m_location = kernel.m_program->getResourceLocation(GL_UNIFORM, uniform_name);
        if (m_location < 0)
            throw glutils::GLError("invalid uniform location");
    }

    void ComputeKernel::TextureSampler::setTextureUnit(const ComputeKernel &kernel, GLuint index) const
    {
        gl.ProgramUniform1ui(kernel.m_program->getName(), m_location, index);
    }

    auto ComputeKernel::TextureSampler::getTextureUnit(const ComputeKernel &kernel) const -> GLuint
    {
        GLuint unit;
        gl.GetUniformuiv(kernel.m_program->getName(), m_location, &unit);
        return unit;
    }


    // =============================================== ComputeKernel ================================================ //

    ComputeKernel::ComputeKernel(unsigned int count, const char **source_strings)
    {
        using namespace glutils;

        Guard<Shader> shader {Shader::Type::compute};
        shader->setSource(static_cast<GLsizei>(count), source_strings);
        shader->compile();
        if (!shader->getParameter(Shader::Parameter::compile_status))
            throw GLError(shader->getInfoLog());

        m_program->attachShader(*shader);
        m_program->link();
        if (!m_program->getParameter(Program::Parameter::link_status))
            throw GLError(m_program->getInfoLog());
        m_program->detachShader(*shader);
    }

    void ComputeKernel::useProgram() const
    {
        gl.UseProgram(m_program->getName());
    }
} // placement