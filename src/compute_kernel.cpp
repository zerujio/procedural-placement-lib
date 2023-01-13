#include "placement/compute_kernel.hpp"

#include "glutils/shader.hpp"
#include "glutils/error.hpp"
#include "gl_context.hpp"

#include "glm/gtc/type_ptr.hpp"

namespace placement {
ComputeKernel::ComputeKernel(unsigned int count, const char **source_strings)
{
    using namespace GL;

    Shader shader{ShaderHandle::Type::compute};
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

void ComputeKernel::useProgram() const
{
    gl.UseProgram(m_program.getName());
}

GL::GLuint ComputeKernel::m_getResourceIndex(Interface interface, const char *name) const
{
    const GLuint value = m_program.getResourceIndex(interface, name);

    if (value == GL_INVALID_INDEX)
        throw GL::GLError("glGetProgramResourceIndex() return GL_INVALID_INDEX");

    return value;
}

ComputeKernel::UniformLocation ComputeKernel::m_getUniformLocation(const char *name) const
{
    const UniformLocation value {m_program.getResourceLocation(Interface::uniform, name)};
    if (!value)
        throw GL::GLError("failed to retrieve uniform location");
    return value;
}

ComputeKernel::ProgramResourceIndexBase::ProgramResourceIndexBase(const ComputeKernel &kernel,
                                                                  GL::ProgramHandle::Interface program_interface,
                                                                  const char *resource_name) :
        m_index(kernel.m_program.getResourceIndex(program_interface, resource_name))
{
    if (m_index == GL_INVALID_INDEX)
        throw GL::GLError("glGetProgramResourceIndex() returned GL_INVALID_INDEX");
}

auto ComputeKernel::InterfaceBlockBase::m_queryBindingIndex(const ComputeKernel &kernel,
                                                            ComputeKernel::InterfaceBlockBase::Type type)
-> GL::GLuint
{
    const GLenum prop = GL_BUFFER_BINDING;
    kernel.m_program.getResource(type, m_resource_index.get(), 1, &prop, 1, nullptr,
                                 reinterpret_cast<GLint *>(&m_binding_index));
    return m_binding_index;
}

void ComputeKernel::TextureSampler::setTextureUnit(const ComputeKernel &kernel, GL::GLuint texture_unit)
{
    kernel.m_setUniform(m_location, static_cast<GLint>(texture_unit));
    m_tex_unit = texture_unit;
}

auto ComputeKernel::TextureSampler::queryTextureUnit(const placement::ComputeKernel &kernel) -> GL::GLuint
{
    glm::uvec2 values{0};  // querying a sampler returns an uvec2 for some reason
    gl.GetUniformuiv(kernel.m_program.getName(), m_location.get(), glm::value_ptr(values));
    m_tex_unit = values.x;
    return m_tex_unit;
}
} // placement