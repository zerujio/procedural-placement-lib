#include "placement/reduction_kernel.hpp"

#include "glutils/buffer.hpp"
#include "gl_context.hpp"

#include "glm/glm.hpp"

#include <string>
#include <sstream>

namespace placement {

    constexpr ShaderStorageBlockInfo g_position_block_info {"PositionBuffer", GL_INVALID_INDEX, 0};
    constexpr ShaderStorageBlockInfo g_index_block_info {"IndexBuffer", GL_INVALID_INDEX, 1};

    static const std::string source_string {
        (std::ostringstream() << R"glsl(
#version 430 core

layout(std430, binding=)glsl" << g_position_block_info.binding_index << R"glsl()
restrict coherent
buffer )glsl" << g_position_block_info.name << R"glsl(
{
    vec3 position_buffer[];
};

layout(std430, binding=)glsl" << g_index_block_info.binding_index << R"glsl()
restrict coherent
buffer  )glsl" << g_index_block_info.name << R"glsl(
{
    uint index_buffer[];
};

struct Candidate
{
    vec3 position;
    bool is_valid;
};

void main()
{
    const uint invocation_index = gl_GlobalInvocationID.x * 2;
    const Candidate c0 = {position_buffer[invocation_index], index_buffer[invocation_index]};
    const Candidate c1 = {position_buffer[invocation_index + 1], index_buffer[invocation_index + 1]};

    for (uint group_size = 1; group_size < candidate_buffer.length(); group_size <<= 1)
    {
        const uint group_index = (gl_GlobalInvocationID.x / group_size) * 2 + 1;
        const uint base_index = group_index * group_size;
        const uint write_index = base_index + gl_GlobalInvocationID.x % group_size;
        const uint read_index = base_index - 1;

        if (write_index < candidate_buffer.length())
            index_buffer[write_index] += index_buffer[read_index];

        memoryBarrierBuffer();
    }

    if (c0.is_valid)
    {
        const uint c_index = index_buffer[invocation_index];
        position_buffer[c_index] = c0.position;
    }

    if (c1.is_valid)
    {
        const uint c_index = index_buffer[invocation_index + 1];
        position_buffer[c_index] = c1.position;
    }
}
)glsl"
        ).str()
    };

    using namespace glutils;

    ReductionKernel::ReductionKernel() :
            m_position_ssb(g_position_block_info),
            m_index_ssb(g_index_block_info)
    {
        Guard<Shader> shader {Shader::Type::compute};
        shader->setSource(source_string);
        shader->compile();
        if (!shader->getParameter(Shader::Parameter::compile_status))
            throw std::runtime_error(shader->getInfoLog());

        m_program->attachShader(*shader);
        m_program->link();
        if (!m_program->getParameter(Program::Parameter::link_status))
            throw std::runtime_error(m_program->getInfoLog());
        m_program->detachShader(*shader);

        // retrieve ssbo info
        m_position_ssb.resource_index = m_program->getResourceIndex(GL_SHADER_STORAGE_BLOCK,
                                                                    m_position_ssb.name);
        if (m_position_ssb.resource_index == GL_INVALID_INDEX)
            throw std::runtime_error("couldn't retrieve resource index for position buffer");

        m_index_ssb.resource_index = m_program->getResourceIndex(GL_SHADER_STORAGE_BLOCK,
                                                                 m_index_ssb.name);
        if (m_index_ssb.resource_index == GL_INVALID_INDEX)
            throw std::runtime_error("couldn't retrieve resource index for index buffer");
    }

    std::size_t ReductionKernel::operator()(glutils::BufferOffset position_buffer, glutils::BufferOffset index_buffer,
                                            std::size_t count) const
    {
        bindPositionBuffer(position_buffer, count);
        bindIndexBuffer(index_buffer, count);

        dispatchCompute(count);

        gl.MemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

        index_buffer.offset += (count - 1) * sizeof(GLuint);
        uint reduced_count = 0;
        index_buffer.read(sizeof(GLuint), &reduced_count);

        return reduced_count;
    }

    void ReductionKernel::setPositionBufferBinding(GLuint index)
    {
        m_program->setShaderStorageBlockBinding(m_position_ssb.resource_index, index);
        m_position_ssb.binding_index = index;
    }

    GLuint ReductionKernel::getPositionBufferBinding() const
    {
        return m_position_ssb.binding_index;
    }

    void ReductionKernel::setIndexBufferBinding(GLuint index)
    {
        m_program->setShaderStorageBlockBinding(m_index_ssb.resource_index, index);
        m_index_ssb.binding_index = index;
    }

    GLuint ReductionKernel::getIndexBufferBinding() const
    {
        return m_index_ssb.binding_index;
    }

    void ReductionKernel::dispatchCompute(std::size_t count) const
    {
        const unsigned int num_work_groups = count / 32 + (count % 32 != 0); // one wgroup for two 4x4 blocks
        m_program->use();
        gl.DispatchCompute(num_work_groups, 1, 1);
    }

    void ReductionKernel::bindPositionBuffer(glutils::BufferOffset buffer_offset, std::size_t count) const
    {
        gl.BindBufferRange(GL_SHADER_STORAGE_BUFFER,
                           m_position_ssb.current_binding_index,
                           buffer_offset.buffer.getName(),
                           buffer_offset.offset,
                           computePositionBufferSize(count));
    }

    void ReductionKernel::bindIndexBuffer(glutils::BufferOffset buffer_offset, std::size_t count) const
    {
        gl.BindBufferRange(GL_SHADER_STORAGE_BUFFER,
                           m_index_ssb.current_binding_index,
                           buffer_offset.buffer.getName(),
                           buffer_offset.offset,
                           computeIndexBufferSize(count));
    }


} // placement