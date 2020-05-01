#!/usr/bin/env python3
from glfw_controller import *
from mesh_view import *
from image_view import *

import numpy as np

def read_poses(pose_file):
    lines = open(pose_file).read().splitlines()
    poses = [[float(z) for z in x.split()[1:]] for x in lines]
    poses = np.array(poses)
    translations = np.zeros((poses.shape[0], 4))
    translations[:, 0:3] = poses[:, 0:3]
    translations[:, 3] = 1
    vertices = translations[:, 0:3]
    vertices = np.concatenate((vertices, np.array([vertices[0, :]])))
    return vertices 

poses_file = 'targets/camera_to_image/poses.txt'

app = GlfwApp()
app.init()

multi_controller = GlfwMultiController()


model = MultiMeshModel(
    [
        {
            "type": "mesh",
            "mesh": "sphere.obj",
            "fragment": "wireframe_fragment.glsl",
            "vertex": "wireframe_vertex.glsl",
            "geometry": "wireframe_geometry.glsl"
        },
        {
            "type":"points",
            "mesh": read_poses(f'./{poses_file}'),
            "fragment": "point_fragment.glsl",
            "vertex" : "point_vertex.glsl",
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
