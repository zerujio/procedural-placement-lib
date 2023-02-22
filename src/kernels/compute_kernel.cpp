#include "placement/kernel/compute_kernel.hpp"

#include "glutils/shader.hpp"
#include "glutils/error.hpp"
#include "../gl_context.hpp"

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

void ComputeKernel::dispatch(glm::uvec3 num_work_groups)
{
    gl.DispatchCompute(num_work_groups.x, num_work_groups.y, num_work_groups.z);
}

GLuint ComputeKernel::m_getResourceIndex(Interface interface, const char *name) const
{
    const GLuint value = m_program.getResourceIndex(interface, name);

    if (value == GL_INVALID_INDEX)
        throw GL::GLError("glGetProgramResourceIndex() returned GL_INVALID_INDEX");

    return value;
}

ComputeKernel::UniformLocation ComputeKernel::m_getUniformLocation(const char *name) const
{
    const UniformLocation value {m_program.getResourceLocation(Interface::uniform, name)};
    if (!value)
        throw GL::GLError(std::string("failed to retrieve uniform location for ") + name);
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

} // placement