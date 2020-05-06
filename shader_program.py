from OpenGL.GL import *


class ShaderProgram(object):
    def __init__(self, vertex, fragment, geometry=None):
        self.program_id = glCreateProgram()
        vs_id = self.add_shader(vertex, GL_VERTEX_SHADER)
        frag_id = self.add_shader(fragment, GL_FRAGMENT_SHADER)
        if geometry is not None:
            geom_id = self.add_shader(geometry, GL_GEOMETRY_SHADER)

        glAttachShader(self.program_id, vs_id)
        glAttachShader(self.program_id, frag_id)
        if geometry is not None:
            glAttachShader(self.program_id, geom_id)
        glLinkProgram(self.program_id)
        info = glGetProgramInfoLog(self.program_id)

        if glGetProgramiv(self.program_id, GL_LINK_STATUS) != GL_TRUE:
            info = glGetProgramInfoLog(self.program_id)
            glDeleteShader(vs_id)
            glDeleteShader(frag_id)
            if geometry is not None:
                glDeleteShader(geom_id)
            raise RuntimeError("Error linking program: %s" % (info))

    def add_shader(self, source, shader_type):
        try:
            shader_id = glCreateShader(shader_type)
            glShaderSource(shader_id, source)
            glCompileShader(shader_id)
            if glGetShaderiv(shader_id, GL_COMPILE_STATUS) != GL_TRUE:
                info = glGetShaderInfoLog(shader_id)
                raise RuntimeError(
                    "Shader compilation failed: %s\nsource=%s" % (info, source)
                )
            return shader_id
        except:
            glDeleteShader(shader_id)
            raise

    def uniform_location(self, name):
        location = glGetUniformLocation(self.program_id, name)
        if location < 0:
            raise RuntimeError(
                f"Bad uniform location name = '{name}' location = {location}\n"
            )
        return location

    def attribute_location(self, name):
        location = glGetAttribLocation(self.program_id, name)
        if location < 0:
            raise RuntimeError(
                f"Bad attribute location name = '{name}' location = {location}\n"
            )
        return location
