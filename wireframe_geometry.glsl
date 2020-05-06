#version 330

layout (triangles) in;
layout (line_strip, max_vertices = 3) out;

uniform mat4 projection;
uniform mat4 view;

in vec4 vs_position[];

void main() {
	for (int n = 0; n < gl_in.length(); n++) {
		gl_Position = projection * view * gl_in[n].gl_Position;
		EmitVertex();
	}
	EndPrimitive();
}
