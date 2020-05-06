#version 330

layout (lines) in;
layout (triangle_strip, max_vertices = 40) out;

uniform mat4 projection;
uniform mat4 view;

flat in vec3 vs_color[];
flat out vec3 gs_color;

void main() {
  mat4 pv = projection * view;
  int num_steps = 10;
  float pi = 3.14159265359;
  vec3 color;
  vec3 from;
  vec3 to;
  vec3 w;
  int i;
  vec3 u;
  vec3 v;
  float theta;
  float delta;
  vec3 left;
  vec3 right;
  vec3 q0;
  vec3 q1;
  vec3 q2;
  vec3 q3;
  float r = 0.0075;
	for (int n = 0; n < gl_in.length(); n+=2) {
		gl_Position = projection * view * gl_in[n].gl_Position;
    color = vs_color[n];
    from = gl_in[n].gl_Position.xyz;
    to = gl_in[n+1].gl_Position.xyz;
    w = normalize((to - from).xyz);
    i = (w.x <= w.y && w.x <= w.z ? 0 : (w.y <= w.x && w.y <= w.z ? 1 : 2));
    u = vec3(0.0f);
    u[i] = 1.0;
    v = normalize(cross(u, w));
    u = normalize(cross(w, v));
    theta = 0;
    delta = 2.0 * pi / num_steps;
    for (int j = 0; j < num_steps; ++j) {
      left = u * cos(theta) + v * sin(theta);
      right =u * cos(theta + delta) + v * sin(theta + delta);
      q0 = r*left + from;
      q1 = r*right + from;
      q2 = r*left + to;
      q3 = r*right + to;
      theta += delta;
      gl_Position = pv * vec4(q0, 1.0);
      gs_color = color;
      EmitVertex();
      gl_Position = pv * vec4(q1, 1.0);
      gs_color = color;
      EmitVertex();
      gl_Position = pv * vec4(q2, 1.0);
      gs_color = vs_color[n+1];
      EmitVertex();
      EndPrimitive();

      gl_Position = pv * vec4(q1, 1.0);
      gs_color = color;
      EmitVertex();
      gl_Position = pv * vec4(q3, 1.0);
      gs_color = vs_color[n+1];
      EmitVertex();
      gl_Position = pv * vec4(q2, 1.0);
      gs_color = vs_color[n+1];
      EmitVertex();
      EndPrimitive();
    }
	}
}
