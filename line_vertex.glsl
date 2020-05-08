#version 330

in vec3 vertex_position;
in vec3 vertex_color;
flat out vec3 vs_color;
uniform mat4 model;

void main(void)
{
    vs_color = vertex_color;
    gl_Position = model * vec4(vertex_position, 1.0);
}
