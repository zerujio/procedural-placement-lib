#ifndef PROCEDURALPLACEMENTLIB_LOAD_OBJ_HPP
#define PROCEDURALPLACEMENTLIB_LOAD_OBJ_HPP

#include <vector>
#include <string>
#include <tuple>

#include "glm/vec3.hpp"
#include "glm/vec2.hpp"

struct MeshData
{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> tex_coords;
    std::vector<uint> indices;
};

/// loads a mesh from an .obj file.
[[nodiscard]] MeshData loadOBJ(const std::string& filename);

#endif //PROCEDURALPLACEMENTLIB_LOAD_OBJ_HPP
