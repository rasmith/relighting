import glfw
import sys
import graphics_math as gm
import numpy as np
#import pyigl as igl
import igl
import traceback
from OpenGL.GL import *
from OpenGL.arrays import ArrayDatatype
from shader_program import *
from glfw_controller import *
import pdb

class MeshModel(GlfwModel):
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path

    def initialize(self):
    
        try:
            self.vertices, self.faces = igl.read_triangle_mesh(self.mesh_path)
            self.face_normals = igl.per_face_normals(self.vertices,self.faces, np.zeros((1,3)))
            self.vertex_normals = igl.per_vertex_normals(self.vertices, self.faces)
            self.vertices = self.vertices.astype(dtype = np.float32, order = 'C')
            self.faces = self.faces.astype(dtype = np.uint32, order = 'C')
            self.face_normals = self.face_normals.astype(dtype = np.float32, order = 'C')
            self.vertex_normals = self.vertex_normals.astype(dtype = np.float32, order = 'C')
        except:
            traceback.print_exc(file=sys.stdout)
            sys.exit(-1)

        self.num_faces = self.faces.shape[0]
        self.num_vertices = self.vertices.shape[0]
        self.center = np.mean(self.vertices, axis = 0)
        self.max_vals = np.max(self.vertices, axis = 0)
        self.min_vals = np.min(self.vertices, axis = 0)
        self.extents = self.max_vals - self.min_vals
        print("min = %s, max = %s, extents = %s, center = %s" % (self.min_vals, self.max_vals,
                                                    self.extents, self.center))

        self.vertices = (self.vertices - self.center) / self.extents
        self.vertex_byte_count = ArrayDatatype.arrayByteCount(self.vertices)
        self.vertex_normal_byte_count = ArrayDatatype.arrayByteCount(
            self.vertex_normals)
        self.face_byte_count = ArrayDatatype.arrayByteCount(self.faces)



class MeshView(GlfwView):
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

    def set_light_position(self, position):
        self.light_position = np.array(position)

    def update_vbos(self):
        glBindVertexArray(self.vao_id)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[0])
        glBufferData(GL_ARRAY_BUFFER, self.model.vertex_byte_count,
                     self.model.vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[1])
        glBufferData(GL_ARRAY_BUFFER, self.model.vertex_normal_byte_count,
                     self.model.vertex_normals, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_id[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,  self.model.face_byte_count,
                     self.model.faces, GL_STATIC_DRAW)

    def set_hints(self):
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    def initialize(self):
        self.set_hints()

        # Load shaders.
        self.fragment = open(self.fragment_shader_path, 'r').read()
        self.vertex = open(self.vertex_shader_path, 'r').read()

        # Compile shaders and link program.
        self.program = ShaderProgram(
            fragment=self.fragment, vertex=self.vertex)

        glUseProgram(self.program.program_id)

        fragment_color_location = glGetFragDataLocation(
            self.program.program_id, "fragment_color")

        # Generate VAOs.
        self.vao_id = glGenVertexArrays(1)
        glBindVertexArray(self.vao_id)

        # Generate VBOs.
        self.vbo_id = glGenBuffers(3)

        # Setup the vertex data in VBO.
        self.vertex_location = self.program.attribute_location(
            'vertex_position')
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[0])
        glVertexAttribPointer(self.vertex_location, 3,
                              GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(self.vertex_location)

        # Setup the normal data in VBO.
        self.vertex_normal_location = self.program.attribute_location(
            'vertex_normal')
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[1])
        glVertexAttribPointer(self.vertex_normal_location, 3,
                              GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(self.vertex_normal_location)

        # Setup the indices data VBO.
        self.model_location = self.program.uniform_location('model')
        self.view_location = self.program.uniform_location('view')
        self.projection_location = self.program.uniform_location('projection')
        self.light_position_location = self.program.uniform_location(
            'light_position')

        self.update_vbos()


    def render(self, width, height):
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LESS)

        aspect = float(width) / float(height)
        
        projection_matrix = gm.perspective(
            self.fov, aspect, self.near, self.far)
        model_matrix = np.eye(4)
        view_matrix = gm.lookat(self.eye, self.at, self.up)

        # Specify program to be used
        glUseProgram(self.program.program_id)
        model_matrix_py = model_matrix.transpose().flatten().tolist()
        glUniformMatrix4fv(self.model_location, 1, GL_FALSE,
                           (GLfloat * 16)(*model_matrix_py))
        view_matrix_py = view_matrix.transpose().flatten().tolist()
        glUniformMatrix4fv(self.view_location, 1, GL_FALSE,
                           (GLfloat * 16)(*view_matrix_py))
        projection_matrix_py = projection_matrix.transpose().flatten().tolist()
        glUniformMatrix4fv(self.projection_location, 1, GL_FALSE,
                           (GLfloat * 16)(*projection_matrix_py))
        light_position_py = self.light_position.tolist()
        glUniform3fv(self.light_position_location, 1,
                     (GLfloat * 3)(*light_position_py))

        # Bind to VAO.
        glBindVertexArray(self.vao_id)

        # Draw the triangles.
        glDrawElements(GL_TRIANGLES, self.mesh.num_faces *
                       3, GL_UNSIGNED_INT, None)


class MultiMeshModel(GlfwModel):
    def __init__(self, infos):
        self.mesh_infos = [MeshInfo(info) for info in infos]

    def initialize(self):
        for m in self.mesh_infos:
            m.model.initialize()


class MeshInfo:
    def __init__(self, info):
        self.program_info = info
        self.model = MeshModel(info['mesh'])

    def update_vbos(self):
        glBindVertexArray(self.vao_id)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[0])
        glBufferData(GL_ARRAY_BUFFER, self.model.vertex_byte_count,
                     self.model.vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id[1])
        glBufferData(GL_ARRAY_BUFFER, self.model.vertex_normal_byte_count,
                     self.model.vertex_normals, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_id[2])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,  self.model.face_byte_count,
                     self.model.faces, GL_STATIC_DRAW)

        
class MultiMeshView(GlfwView):
    def __init__(self):
        pass

    def set_camera(self, eye, at, up, fov, near, far):
        self.eye = np.transpose([eye])
        self.at = np.transpose([at])
        self.up = np.transpose([up])
        self.fov = fov
        self.near = near
        self.far = far

    def set_light_position(self, position):
        self.light_position = np.array(position)

    def set_hints(self):
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    def initialize(self):
        self.set_hints()

        for mesh_info in self.model.mesh_infos:
            # Load shaders.
            mesh_info.fragment = open(mesh_info.program_info['fragment'], 'r').read()
            mesh_info.vertex = open(mesh_info.program_info['vertex'], 'r').read()
            mesh_info.geometry = open(mesh_info.program_info['geometry'], 'r').read() if mesh_info.program_info['geometry'] is not None else None

            # Compile shaders and link program.
            mesh_info.program = ShaderProgram(mesh_info.vertex, mesh_info.fragment, mesh_info.geometry)

            glUseProgram(mesh_info.program.program_id)

            fragment_color_location = glGetFragDataLocation(
                mesh_info.program.program_id, "fragment_color")

            # Generate VAOs.
            mesh_info.vao_id = glGenVertexArrays(1)
            glBindVertexArray(mesh_info.vao_id)

            # Generate VBOs.
            mesh_info.vbo_id = glGenBuffers(3)

            # Setup the vertex data in VBO.
            mesh_info.vertex_location = mesh_info.program.attribute_location(
                'vertex_position')
            glBindBuffer(GL_ARRAY_BUFFER, mesh_info.vbo_id[0])
            glVertexAttribPointer(mesh_info.vertex_location, 3,
                                  GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(mesh_info.vertex_location)

            # Setup the normal data in VBO.
            mesh_info.vertex_normal_location = mesh_info.program.attribute_location(
                'vertex_normal')
            glBindBuffer(GL_ARRAY_BUFFER, mesh_info.vbo_id[1])
            glVertexAttribPointer(mesh_info.vertex_normal_location, 3,
                                  GL_FLOAT, GL_FALSE, 0, None)
            glEnableVertexAttribArray(mesh_info.vertex_normal_location)

            # Setup the indices data VBO.
            mesh_info.model_location = mesh_info.program.uniform_location('model')
            mesh_info.view_location = mesh_info.program.uniform_location('view')
            mesh_info.projection_location = mesh_info.program.uniform_location('projection')
            mesh_info.light_position_location = mesh_info.program.uniform_location(
                'light_position')

            mesh_info.update_vbos()

    def render_meshes(self, width, height):
        aspect = float(width) / float(height)

        projection_matrix = gm.perspective(
            self.fov, aspect, self.near, self.far)
        model_matrix = np.eye(4)
        view_matrix = gm.lookat(self.eye, self.at, self.up)

        for mesh_info in self.model.mesh_infos: 
            # Specify program to be used
            glUseProgram(mesh_info.program.program_id)
            model_matrix_py = model_matrix.transpose().flatten().tolist()
            glUniformMatrix4fv(mesh_info.model_location, 1, GL_FALSE,
                               (GLfloat * 16)(*model_matrix_py))
            view_matrix_py = view_matrix.transpose().flatten().tolist()
            glUniformMatrix4fv(mesh_info.view_location, 1, GL_FALSE,
                               (GLfloat * 16)(*view_matrix_py))
            projection_matrix_py = projection_matrix.transpose().flatten().tolist()
            glUniformMatrix4fv(mesh_info.projection_location, 1, GL_FALSE,
                               (GLfloat * 16)(*projection_matrix_py))
            light_position_py = self.light_position.tolist()
            glUniform3fv(mesh_info.light_position_location, 1,
                         (GLfloat * 3)(*light_position_py))

            # Bind to VAO.
            glBindVertexArray(mesh_info.vao_id)

            # Draw the triangles.
            glDrawElements(GL_TRIANGLES, mesh_info.model.num_faces *
                           3, GL_UNSIGNED_INT, None)


    def render(self, width, height):
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthFunc(GL_LESS)
        self.render_meshes(width, height)

