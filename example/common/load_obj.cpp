#include "load_obj.hpp"

#define FAST_OBJ_IMPLEMENTATION
#include "fast_obj.h"

#include "glm/gtc/type_ptr.hpp"

#include <memory>
#include <stdexcept>
#include <map>


bool operator< (const fastObjIndex& l, const fastObjIndex& r)
{
    return std::make_tuple(l.p, l.n, l.t) < std::make_tuple(r.p, r.n, r.t);
}

MeshData loadOBJ(const std::string& filename)
{
    std::unique_ptr<fastObjMesh, void(*)(fastObjMesh*)> mesh {fast_obj_read(filename.c_str()), fast_obj_destroy};

    if (!mesh)
        throw std::runtime_error("couldn't load mesh from file: " + filename);

    MeshData mesh_data;
    mesh_data.positions.reserve(mesh->position_count / 3);
    mesh_data.positions.reserve(mesh->normal_count / 3);
    mesh_data.positions.reserve(mesh->texcoord_count / 2);
    mesh_data.indices.reserve(mesh->index_count);

    std::map<fastObjIndex, uint> obj_to_gl;

    for (const fastObjIndex* obj_index = mesh->indices; obj_index != mesh->indices + mesh->index_count; obj_index++)
    {
        // check if gl vertex already exists
        const auto [iter, inserted] = obj_to_gl.try_emplace(*obj_index, obj_to_gl.size());

        if (inserted)
        {
            mesh_data.positions.emplace_back(glm::make_vec3(mesh->positions + obj_index->p * 3));
            if (mesh->normal_count > 0)
                mesh_data.normals.emplace_back(glm::make_vec3(mesh->normals + obj_index->n * 3));
            if (mesh->texcoord_count > 0)
                mesh_data.tex_coords.emplace_back(glm::make_vec3(mesh->texcoords + obj_index->t * 2));
        }

        mesh_data.indices.emplace_back(iter->second);
    }

    return mesh_data;
}
