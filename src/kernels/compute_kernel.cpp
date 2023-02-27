#include "placement/kernel/compute_kernel.hpp"

#include "glutils/shader.hpp"
#include "glutils/error.hpp"
#include "../gl_context.hpp"

#include "glm/gtc/type_ptr.hpp"

namespace placement {
ComputeShaderProgram::ComputeShaderProgram(unsigned int count, const char **source_strings)
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

void ComputeShaderProgram::useProgram() const
{
    gl.UseProgram(m_program.getName());
}

void ComputeShaderProgram::dispatch(glm::uvec3 num_work_groups) const
{
    useProgram();
    gl.DispatchCompute(num_work_groups.x, num_work_groups.y, num_work_groups.z);
}

GLuint ComputeShaderProgram::getResourceIndex(Interface interface, const char *name) const
{
    const GLuint value = m_program.getResourceIndex(interface, name);

    if (value == GL_INVALID_INDEX)
        throw GL::GLError("glGetProgramResourceIndex() returned GL_INVALID_INDEX");

    return value;
}

ComputeShaderProgram::UniformLocation ComputeShaderProgram::getUniformLocation(const char *name) const
{
    const UniformLocation value {m_program.getResourceLocation(Interface::uniform, name)};
    if (!value)
        throw GL::GLError(std::string("failed to retrieve uniform location for ") + name);
    return value;
}

GLuint
ComputeShaderProgram::m_queryInterFaceBlockBindingIndex(InterfaceBlockType block_type, GLuint resource_index) const
{
    GLint index;
    GLenum prop {GL_BUFFER_BINDING};
    m_program.getResource(static_cast<Interface>(block_type), resource_index, 1, &prop, 1, nullptr, &index);
    return index;
}

} // placement