#ifndef PROCEDURALPLACEMENTLIB_DENSITY_MAP_HPP
#define PROCEDURALPLACEMENTLIB_DENSITY_MAP_HPP

#include "glutils/gl_types.hpp"

namespace placement {

/// A density map specifies the probability distribution of a single class of object over the landscape.
struct DensityMap
{
    /// name of an OpenGL texture object.
    GLuint texture{0};

    /// Values in texture will be multiplied by this factor.
    float scale{1};

    /// Values in texture will be offset by this amount, after scaling.
    float offset{0};

    /// Values in texture will be clamped to the range [min_value, max_value], after scaling and offset.
    float min_value{0};
    float max_value{1};
};

} // placement

#endif //PROCEDURALPLACEMENTLIB_DENSITY_MAP_HPP
