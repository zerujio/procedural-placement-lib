#ifndef PROCEDURALPLACEMENTLIB_PLACEMENT_HPP
#define PROCEDURALPLACEMENTLIB_PLACEMENT_HPP

#include "placement_pipeline.hpp"

namespace placement {
    using GLproc = void (*)();
    using GLloader = GLproc (*)(const char*);

    /**
     * @brief Load the current OpenGL context for the calling thread.
     * @return a boolean indicating if the operation was successful.
     */
    bool loadGLContext(GLloader) noexcept;
}

#endif //PROCEDURALPLACEMENTLIB_PLACEMENT_HPP
