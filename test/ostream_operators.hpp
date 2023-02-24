#ifndef PROCEDURALPLACEMENTLIB_OSTREAM_OPERATORS_H
#define PROCEDURALPLACEMENTLIB_OSTREAM_OPERATORS_H

#include "placement/placement_result.hpp"
#include "glm/glm.hpp"
#include <ostream>
#include <vector>
#include <set>

namespace glm {

template<auto L, typename T, auto Q>
std::ostream &operator<<(std::ostream &out, const vec<L, T, Q> &v)
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

}

template<typename T, typename Y>
std::ostream& operator<< (std::ostream& out, const std::pair<T, Y>& pair)
{
    return out << "{first: " << pair.first << ", second: " << pair.second << "}";
}

template<class T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& vector)
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

template<typename T, typename Compare, typename Alloc>
std::ostream& operator<< (std::ostream& out, const std::set<T, Compare, Alloc>& set)
{
    if (set.empty())
        return out << "{}";
    auto it = set.begin();
    out << "{" << *it++;
    while (it != set.end())
        out << ", " << *it++;
    return out << "}";
}

namespace placement {
std::ostream &operator<<(std::ostream &out, const Result::Element &element)
{
    return out << "{position=" << element.position << ", class_index=" << element.class_index << "}";
}
} // placement

#endif //PROCEDURALPLACEMENTLIB_OSTREAM_OPERATORS_H
