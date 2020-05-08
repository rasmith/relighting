#version 330 

in vec3 vertex_normal;
in vec3 vertex_position;
uniform vec3 light_position;
uniform mat4 model;
void main() {
	vec3 dummy = vertex_normal;
	vec3 dummy1 = light_position;
	gl_Position = model * vec4(vertex_position, 1.0);
}
