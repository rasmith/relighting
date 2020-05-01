#!/usr/bin/env python3
from glfw_controller import *
from mesh_view import *
from image_view import *

app = GlfwApp()
app.init()

multi_controller = GlfwMultiController()

mesh_path = "bunny.off"
mesh_fragment_shader = "mesh_fragment.glsl"
mesh_vertex_shader = "mesh_vertex.glsl"

model = MultiMeshModel(
    [
        {
            "mesh": "bunny.off",
            "fragment": "mesh_fragment.glsl",
            "vertex": "mesh_vertex.glsl",
            "geometry": None,
        }
    ]
)

view = MultiMeshView()
eye = [0.0, 0.0, 2.0, 1.0]
at = [0.0, 0.0, 0.0, 1.0]
up = [0.0, 1.0, 0.0, 1.0]
fov = 45.0
near = 0.0001
far = 100
light_position = [0.0, 5.0, 1.0]
view.set_camera(eye, at, up, fov, near, far)
view.set_light_position(light_position)

mesh_controller = GlfwController(
    400, 300, 100, 100, "MultiMeshView", view, model
)
multi_controller.add(mesh_controller)

multi_controller.run()
