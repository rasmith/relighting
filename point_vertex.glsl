#version 330

in vec3 vertex_position;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;

void main(void)
{
    mat4 model_view  = view * model;
    mat4 mvp = projection*model_view;
    gl_Position = mvp * vec4(vertex_position, 1.0);
}
