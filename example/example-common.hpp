/// This header declares various utilities used in the examples.

#ifndef PROCEDURALPLACEMENTLIB_EXAMPLE_COMMON_HPP
#define PROCEDURALPLACEMENTLIB_EXAMPLE_COMMON_HPP

#include "common/glfw_wrapper.hpp"
#include "common/camera_controller.hpp"
#include "common/imgui_wrapper.hpp"

#include "simple-renderer/mesh.hpp"
#include "simple-renderer/shader_program.hpp"

#include <utility>

/// loads a texture from a file and create an OpenGL texture object from it.
GLuint loadTexture(const char* filename);

std::pair<simple::Mesh, simple::ShaderProgram> makeAxes();

const std::vector<glm::vec3>& getCubePositions();
const std::vector<glm::vec3>& getCubeNormals();
const std::vector<glm::vec2>& getCubeUVs();
const std::vector<unsigned int>& getCubeIndices();
simple::Mesh makeCubeMesh();

std::vector<glm::vec3> generateCirclePositions(unsigned int num_vertices);

#endif //PROCEDURALPLACEMENTLIB_EXAMPLE_COMMON_HPP
