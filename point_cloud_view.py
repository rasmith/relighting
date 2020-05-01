import glfw
import sys
import graphics_math as gm
import numpy as np
import igl
import traceback
from OpenGL.GL import *
from OpenGL.arrays import ArrayDatatype
from iglhelpers import *
from shader_program import *
from glfw_controller import *


class PointModel(GlfwModel):
    def __init__(self, points):
        self.vertices = points.astype(dtype=np.float32, order="C")

    def initialize(self):
        self.num_vertices = self.vertices.shape[0]
        self.center = np.mean(self.vertices, axis=0)
        self.max_vals = np.max(self.vertices, axis=0)
        self.min_vals = np.min(self.vertices, axis=0)
        self.extents = self.max_vals - self.min_vals
        print(
            "min = %s, max = %s, extents = %s, center = %s"
            % (self.min_vals, self.max_vals, self.extents, self.center)
        )

        self.vertices = (self.vertices - self.center) / self.extents
        self.vertex_byte_count = ArrayDatatype.arrayByteCount(self.vertices)
        self.faces = []
        for i in range(self.vertices.shape[0]):
            self.faces.append(i)
        self.faces = np.array(self.faces).astype(dtype=np.uint32, order="C")
        self.face_byte_count = ArrayDatatype.arrayByteCount(self.faces)
        self.num_faces = self.vertices.shape[0]


class PointCloudView(GlfwView):
    def __init__(self, fragment_shader_path, vertex_shader_path):
        self.fragment_shader_path = fragment_shader_path
        self.vertex_shader_path = vertex_shader_path

    def set_camera(self, eye, at, up, fov, near, far):
        self.eye = np.transpose([eye])
        self.at = np.transpose([at])
        self.up = np.transpose([up])
        self.fov = fov
        self.near = near
        self.far = far

    def update_vbos(self):
        glBindVertexArray(self.vao_id)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[0])
        glBufferData(
            GL_ARRAY_BUFFER,
            self.model.vertex_byte_count,
            self.model.vertices,
            GL_STATIC_DRAW,
        )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_id[1])
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            self.model.face_byte_count,
            self.model.faces,
            GL_STATIC_DRAW,
        )

    def set_hints(self):
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    def initialize(self):
        self.set_hints()

        # Load shaders.
        self.fragment = open(self.fragment_shader_path, "r").read()
        self.vertex = open(self.vertex_shader_path, "r").read()

        # Compile shaders and link program.
        self.program = ShaderProgram(fragment=self.fragment, vertex=self.vertex)

        glUseProgram(self.program.program_id)

        fragment_color_location = glGetFragDataLocation(
            self.program.program_id, "fragment_color"
        )

        # Generate VAOs.
        self.vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.vao_id)

        # Generate VBOs.
        self.vbo_id = glGenBuffers(2)

        # Setup the vertex data in VBO.
        self.vertex_location = self.program.attribute_location(
            "vertex_position"
        )
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[0])
        glVertexAttribPointer(
            self.vertex_location, 3, GL_FLOAT, GL_FALSE, 0, None
        )
        glEnableVertexAttribArray(self.vertex_location)

        # Setup the indices data VBO.
        self.model_location = self.program.uniform_location("model")
        self.view_location = self.program.uniform_location("view")
        self.projection_location = self.program.uniform_location("projection")

        self.update_vbos()

    def render(self, width, height):
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LESS)

        aspect = float(width) / float(height)

        projection_matrix = gm.perspective(
            self.fov, aspect, self.near, self.far
        )
        model_matrix = np.eye(4)
        view_matrix = gm.lookat(self.eye, self.at, self.up)

        # Specify program to be used
        glUseProgram(self.program.program_id)
        model_matrix_py = model_matrix.transpose().flatten().tolist()
        glUniformMatrix4fv(
            self.model_location, 1, GL_FALSE, (GLfloat * 16)(*model_matrix_py)
        )
        view_matrix_py = view_matrix.transpose().flatten().tolist()
        glUniformMatrix4fv(
            self.view_location, 1, GL_FALSE, (GLfloat * 16)(*view_matrix_py)
        )
        projection_matrix_py = projection_matrix.transpose().flatten().tolist()
        glUniformMatrix4fv(
            self.projection_location,
            1,
            GL_FALSE,
            (GLfloat * 16)(*projection_matrix_py),
        )

        # Bind to VAO.
        glBindVertexArray(self.vao_id)

        # Draw the points.
        glDrawElements(
            GL_POINTS, self.model.num_faces * 3, GL_UNSIGNED_INT, None
        )
