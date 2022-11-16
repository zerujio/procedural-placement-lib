#ifndef PROCEDURALPLACEMENTLIB_OSTREAM_OPERATORS_H
#define PROCEDURALPLACEMENTLIB_OSTREAM_OPERATORS_H

#include "placement/placement_pipeline_kernel.hpp"
#include "glm/glm.hpp"
#include <ostream>
#include <vector>

template<auto L, typename T, auto Q>
auto operator<< (std::ostream& out, glm::vec<L, T, Q> v) -> std::ostream&
{
    constexpr auto sep = ", ";
    out << "(" << v.x << sep << v.y;
    if constexpr (L > 2)
    {
        out << sep << v.z;
        if constexpr (L > 3)
            out << sep << v.w;
    }
    return out << ")";
}

template<class T>
auto operator<< (std::ostream& out, std::vector<T> vector) ->std::ostream&
{
    if (vector.empty())
        return out << "[]";

    out << "[";
    auto it = vector.cbegin();
    out << *it++;
    while (it != vector.cend())
        out << ", " << *it++;
    return out << "]";
}

#endif //PROCEDURALPLACEMENTLIB_OSTREAM_OPERATORS_H