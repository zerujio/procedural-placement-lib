#include "gl_context.hpp"
#include "placement/placement.hpp"

namespace placement {

    bool loadGLContext(GLloader loader) noexcept
    {
        return glutils::loadGLContext(loader);
    }

} // placement
