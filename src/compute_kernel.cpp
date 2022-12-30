#include "placement/compute_kernel.hpp"

#include "glutils/shader.hpp"
#include "glutils/error.hpp"
#include "gl_context.hpp"

#include "glm/gtc/type_ptr.hpp"

namespace placement {
    ComputeKernel::ComputeKernel(unsigned int count, const char **source_strings)
    {
        using namespace GL;

        Shader shader {ShaderHandle::Type::compute};
        shader.setSource(static_cast<GLsizei>(count), source_strings);
        shader.compile();
        if (!shader.getParameter(ShaderHandle::Parameter::compile_status))
            throw GLError(shader.getInfoLog());

        m_program.attachShader(shader);
        m_program.link();
        if (!m_program.getParameter(ProgramHandle::Parameter::link_status))
            throw GLError(m_program.getInfoLog());
        m_program.detachShader(shader);
    }

    void ComputeKernel::m_useProgram() const
    {
        gl.UseProgram(m_program.getName());
    }

    ComputeKernel::ProgramResourceIndexBase::ProgramResourceIndexBase(const ComputeKernel &kernel,
                                                                      GL::ProgramHandle::Interface program_interface,
                                                                      const char *resource_name) :
        m_index(kernel.m_program.getResourceIndex(program_interface, resource_name))
    {
        if (m_index == GL_INVALID_INDEX)
            throw GL::GLError("glGetProgramResourceIndex() returned GL_INVALID_INDEX");
    }

    ComputeKernel::UniformLocation::UniformLocation(const ComputeKernel &kernel, const char *uniform_name) :
        m_location(kernel.m_program.getResourceLocation(GL::ProgramHandle::Interface::uniform, uniform_name))
    {
        if (m_location < 0)
            throw std::runtime_error("failed to retrieve uniform location");
    }

    auto ComputeKernel::InterfaceBlockBase::m_queryBindingIndex(const ComputeKernel& kernel,
                                                                ComputeKernel::InterfaceBlockBase::Type type)
                                                                -> GL::GLuint
    {
        const GLenum prop = GL_BUFFER_BINDING;
        kernel.m_program.getResource(static_cast<GLenum>(type), m_resource_index.get(), 1, &prop, 1, nullptr,
                                      reinterpret_cast<GLint*>(&m_binding_index));
        return m_binding_index;
    }

    void ComputeKernel::TextureSampler::setTextureUnit(const ComputeKernel& kernel, GL::GLuint texture_unit)
    {
        kernel.setUniform(m_location, static_cast<GLint>(texture_unit));
        m_tex_unit = texture_unit;
    }

    auto ComputeKernel::TextureSampler::queryTextureUnit(const placement::ComputeKernel &kernel) -> GL::GLuint
    {
        glm::uvec2 values {0};  // querying a sampler returns an uvec2 for some reason
        gl.GetUniformuiv(kernel.m_program.getName(), m_location.get(), glm::value_ptr(values));
        m_tex_unit = values.x;
        return m_tex_unit;
    }
} // placement