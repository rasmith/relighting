#version 330
uniform vec3 color;
uniform float opacity;
out vec4 fragment_color;
flat in vec3 gs_color;

void main() {
  vec3 dummy = color;
	//fragment_color = vec4(color, 1.0);
  fragment_color = vec4(gs_color, opacity);
}

