#define INVALID_INDEX 0xFFffFFff
#define MAX_LAYER_COUNT 64

layout(location = 4) in vec3 position_offset;
layout(location = 5) in uint layer_index;

uniform vec3 u_layer_colors[MAX_LAYER_COUNT];

out vec3 f_normal;
out vec3 f_position;
out vec3 f_base_color;

void main()
{
    const vec4 world_position = model_matrix * vec4(vertex_position, 1.0f) + vec4(position_offset, 0.0f);

    gl_Position = proj_matrix * view_matrix * world_position;
    f_position = vec3(world_position);

    f_normal = mat3(transpose(inverse(model_matrix))) * vertex_normal;

    f_base_color = layer_index == INVALID_INDEX ? vec3(1.0f) : u_layer_colors[layer_index];
}