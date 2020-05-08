#version 330

uniform vec3 color;
uniform float opacity;
out vec4 fragment_color;

void main(void)
{
    fragment_color = vec4(color, opacity);
}
