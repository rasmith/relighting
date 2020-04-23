import cv2
import importlib
import numpy as np
import pkgutil
import tasks
import time
import torch
import igl

from evaluator import Evaluator
from glfw_controller import *
from image_view import *
from mesh_view import MeshModel
from mesh_view import MeshView
from multiprocessing import Process
from point_cloud_view import PointCloudView
from point_cloud_view import PointModel
from video_capture import VideoCapture

from tasks.camera_to_image import CfgLoader

def read_poses(pose_file):
    lines = open(pose_file).read().splitlines()
    poses = [[float(z) for z in x.split()[1:]] for x in lines]

    return poses

weights_file = 'weights/camera_to_image.pth'
cfg = CfgLoader().get_cfg("cpu")
model = cfg['model']
model.load_state_dict(torch.load(f'./{weights_file}'))

value = 
value = Variable(value).cpu() 

frames = VideoCapture('targets/camera_to_image/movie.mov')
poses = read_poses('targets/camera_to_image/poses.txt')
poses = np.array(poses)

translations = np.zeros((poses.shape[0], 4))
translations[:, 0:3] = poses[:, 0:3]
translations[:, 3] = 1
quaternions = poses[:, 3:7]
# lights = poses[:, 7:]
vertices = translations[:, 0:3]

points = translations.T
num_points = translations.shape[0]
A = 2 * points 
A[3, :] = 1
f = np.zeros((1, num_points))

for i in range(num_points):
    f[0, i] = np.linalg.norm(points[0:3, i])**2

C, res, rank, svals = np.linalg.lstsq(A.T, f.T, rcond = None)
print(f"C = {C}, R = {((np.linalg.norm(C[0:3]) ** 2) + C[3]) ** (1/2)}")

app = GlfwApp()
app.init()

multi_controller = GlfwMultiController()
width, height = 640, 480
xpos, ypos, title = 0, 0, "Camera"
# mesh_fragment_shader = "mesh_fragment.glsl"
# mesh_vertex_shader = "mesh_vertex.glsl"
point_fragment_shader = "point_fragment.glsl"
point_vertex_shader = "point_vertex.glsl"
# model = MeshModel("bunny.obj")
vertices = np.concatenate((vertices, np.array([vertices[0, :]])))
model = PointModel(vertices)
view = PointCloudView(point_fragment_shader, point_vertex_shader)
eye = [0.0, 0.0, 2.0, 1.0]
at = [0.0, 0.0, 0.0, 1.0]
up = [0.0, 1.0, 0.0, 1.0]
fov = 45.0
near = 0.0001
far = 100
view.set_camera(eye, at, up, fov, near, far)

mesh_controller = GlfwController(width, height, xpos, ypos, title, view, model)
multi_controller.add(mesh_controller)

image_fragment_shader = "image_fragment.glsl"
image_vertex_shader = "image_vertex.glsl"
image_path = "checkerboard.jpg"
image_model = ImageModel(image_path)
image_view = ImageView(image_fragment_shader, image_vertex_shader)

image_controller = GlfwController(400, 300, 500, 100, "Image View", image_view, image_model)
multi_controller.add(image_controller)

multi_controller.run()
