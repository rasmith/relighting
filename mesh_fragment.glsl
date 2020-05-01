#version 330

uniform vec3 color;

in vec4 normal;
in vec4 light_direction;
in vec4 reflect_direction;
in vec4 eye_direction;
out vec4 fragment_color;

void main(void)
{
    float dot_nl = dot(normalize(light_direction), normalize(normal));
    dot_nl = clamp(dot_nl, 0.0, 1.0);
    float dot_rv = dot(normalize(eye_direction), normalize(reflect_direction));
    dot_rv = clamp(dot_rv, 0.0, 1.0);
    float pow_rv = ceil(dot_nl) * clamp(pow(dot_rv, 1.0), 0.0, 1.0);
    fragment_color = clamp((dot_nl + dot_rv) * vec4(color, 1.0), 0.0, 1.0);
}
