layout(location = 4) in vec3 position_offset;

out vec3 f_normal;
out vec3 f_position;
out vec2 f_texcoord;

void main()
{
    const vec4 world_position = model_matrix * vec4(vertex_position, 1.0f) + vec4(position_offset, 0.0f);

    gl_Position = proj_matrix * view_matrix * world_position;
    f_position = vec3(world_position);

    f_normal = mat3(transpose(inverse(model_matrix))) * vertex_normal;

    f_texcoord = vertex_uv;
}