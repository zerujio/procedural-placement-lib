#include "placement/compute_kernel.hpp"

#include "glutils/shader.hpp"
#include "glutils/error.hpp"
#include "gl_context.hpp"

namespace placement {
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

    ComputeKernel::ProgramResourceIndexBase::ProgramResourceIndexBase(const ComputeKernel &kernel,
                                                                      glutils::Program::Interface program_interface,
                                                                      const char *resource_name) :
        m_index(kernel.m_program->getResourceIndex(program_interface, resource_name))
    {
        if (m_index == GL_INVALID_INDEX)
            throw glutils::GLError("glGetProgramResourceIndex() returned GL_INVALID_INDEX");
    }

    ComputeKernel::UniformLocation::UniformLocation(const ComputeKernel &kernel, const char *uniform_name) :
        m_location(kernel.m_program->getResourceLocation(glutils::Program::Interface::uniform, uniform_name))
    {}

    auto ComputeKernel::InterfaceBlockBase::m_queryBindingIndex(const ComputeKernel& kernel,
                                                                ComputeKernel::InterfaceBlockBase::Type type)
                                                                -> glutils::GLuint
    {
        const GLenum prop = GL_BUFFER_BINDING;
        kernel.m_program->getResource(static_cast<GLenum>(type), m_resource_index.get(), 1, &prop, 1, nullptr,
                                      reinterpret_cast<GLint*>(&m_binding_index));
        return m_binding_index;
    }

    void ComputeKernel::TextureSampler::setTextureUnit(const ComputeKernel& kernel, glutils::GLuint texture_unit) const
    {
        kernel.setUniform<GLint>(m_location, texture_unit);
    }

    auto ComputeKernel::TextureSampler::queryTextureUnit(const placement::ComputeKernel &kernel) -> glutils::GLuint
    {
        gl.GetUniformuiv(kernel.m_program->getName(), m_location, &m_tex_unit);
        return m_tex_unit;
    }
} // placement