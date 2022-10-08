#ifndef PROCEDURALPLACEMENTLIB_SSB_PROXY_HPP
#define PROCEDURALPLACEMENTLIB_SSB_PROXY_HPP

#include "glutils/gl_types.hpp"
#include "glutils/program.hpp"

namespace placement {

    using glutils::GLuint;

    class InterfaceBlockInfo
    {
    public:
        /**
         * @brief Query info for an interface block
         * @param program The OpenGL program that includes the interface block.
         * @param name The name of the interface block.
         */
        InterfaceBlockInfo(glutils::Program program, const char* name);
    private:
    };

    /// Allows the configuration of a shader program's shader storage buffer interface block.
    class ShaderStorageBlockProxy
    {
    public:
        /**
         * @brief Create a proxy to modify the shader storage buffer block named @name found in @p program.
         * @param program An OpenGL program.
         * @param name The name of the shader storage buffer block.
         */
        ShaderStorageBlockProxy(glutils::Program program, const char* name);

        /// Set the binding index for the interface block. Calls glShaderStorageBlockBinding.
        void setBinding(GLuint index);

        /// Get the current binding for the interface block.
        [[nodiscard]] GLuint getBinding() const;

    private:
        glutils::Program m_program;
        GLuint m_resource_index;
        GLuint m_binding_index;
    };

} // placement

#endif //PROCEDURALPLACEMENTLIB_SSB_PROXY_HPP
