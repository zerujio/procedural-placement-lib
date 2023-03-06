in vec3 f_normal;
in vec3 f_position;
in vec2 f_texcoord;

uniform vec3 u_view_position;
uniform vec3 u_light_color      = { 1., 1., 1. };
uniform vec3 u_light_position   = { 0., 0., 1000. };
uniform float u_ambient_light_intensity     = 0.1f;
uniform float u_specular_light_intensity    = 0.5f;
uniform float u_specular_highlight_factor   = 1.0f;
uniform sampler2D u_heightmap;
uniform sampler2D u_color_palette;
uniform uint u_color_palette_low;
uniform uint u_color_palette_high;

void main()
{
    const vec3 normal = normalize(f_normal);
    const vec3 light_direction = normalize(u_light_position - f_position);

    const float diffuse_light_intensity = max(dot(normal, light_direction), 0.f);

    const vec3 view_direction = normalize(u_view_position - f_position);
    const vec3 reflect_direction = reflect(-light_direction, normal);
    const float spec = pow(max(dot(view_direction, reflect_direction), 0.f),
                        u_specular_highlight_factor);

    const float normalized_height = texture(u_heightmap, f_texcoord).r;
    const float color_palette_size = textureSize(u_color_palette, 0).x;
    const float low_u = float(u_color_palette_low) / color_palette_size;
    const float high_u = float(u_color_palette_high) / color_palette_size;
    const vec3 material_color = texture(u_color_palette, vec2(mix(low_u, high_u, normalized_height), .5)).rgb;

    const vec3 color_sum = (u_ambient_light_intensity + diffuse_light_intensity + u_specular_light_intensity * spec)
                            * u_light_color * material_color.rgb;

    frag_color = vec4(color_sum, 1.0f);
}