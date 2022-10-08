#include "placement/ssb_proxy.hpp"

#include "glutils/gl.hpp"

namespace placement {


    ShaderStorageBlockProxy::ShaderStorageBlockProxy(glutils::Program program, const char *name) :
        m_program(program),
        m_resource_index(GL_INVALID_INDEX),
        m_binding_index()
    {

    }
} // placement