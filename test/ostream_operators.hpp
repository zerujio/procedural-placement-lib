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
    out << "{" << v.x << sep << v.y;
    if constexpr (L > 2)
    {
        out << sep << v.z;
        if constexpr (L > 3)
            out << sep << v.w;
    }
    return out << "}";
}

template<class T>
auto operator<< (std::ostream& out, std::vector<T> vector) ->std::ostream&
{
    if (vector.empty())
        return out << "[]";

    out << "[";
    for (const T& x : vector)
        out << x << ", ";
    return out << "]";
}

auto operator<< (std::ostream& out, placement::PlacementPipelineKernel::Candidate candidate) -> std::ostream &
{
    return out << "{" << candidate.position << ", " << candidate.index << "}";
}

#endif //PROCEDURALPLACEMENTLIB_OSTREAM_OPERATORS_H
