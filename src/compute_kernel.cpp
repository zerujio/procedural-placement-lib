#include "placement/compute_kernel.hpp"

#include "glutils/error.hpp"
#include "gl_context.hpp"

namespace placement {

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

        // binding index
        const GLenum prop = GL_BUFFER_BINDING;
        int binding = -1;
        kernel.m_program->getResource(static_cast<GLenum>(type), m_resource_index, 1, &prop, 1, nullptr, &binding);
        if (binding < 0)
            throw glutils::GLError("Invalid interface block buffer binding index");
        m_binding_index = binding;
    }

    ComputeKernel::UniformBlock::UniformBlock(const ComputeKernel &kernel, const std::string &name) :
            InterfaceBlock(kernel, InterfaceBlockType::uniform, name)
    {}

    void ComputeKernel::UniformBlock::setBinding(const ComputeKernel &kernel, glutils::GLuint index)
    {
        gl.UniformBlockBinding(kernel.m_program->getName(), m_getResourceIndex(), index);
        m_binding_index = index;
    }

    ComputeKernel::ShaderStorageBlock::ShaderStorageBlock(const ComputeKernel &kernel, const std::string &name) :
            InterfaceBlock(kernel, InterfaceBlockType::shader_storage, name)
    {}
} // placement