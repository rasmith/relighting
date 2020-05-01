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
from torch.autograd import Variable

from tasks.camera_to_image import CfgLoader

def load_mesh(mesh_file):
    pass

def render_image(model, pose):
    pose = torch.from_numpy(np.reshape(poses[0], (1, 1, 1, 7)).astype(np.float32))
    pose = Variable(pose).cpu()
    img = model(pose)
    return img

def checker_board():
    return cv2.cvtColor(cv2.imread('checkerboard.jpg'), cv2.COLOR_BGR2RGB)

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 128, 128)
    return x

def to_numpy_img(x):
    x = to_img(x)
    x = x.detach().numpy().squeeze() if len(x.shape) == 4 else x
    x = np.transpose(x, (1, 2, 0))
    x *= 255.0
    x = np.clip(x, 0.0, 255.0)
    x = x.astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x

def to_torch_pose(x):
    x = torch.from_numpy(np.reshape(x, (1, 1, 1, 7)).astype(np.float32))
    x = Variable(x).cpu()
    return x

def read_poses(pose_file):
    lines = open(pose_file).read().splitlines()
    poses = [[float(z) for z in x.split()[1:]] for x in lines]
    return poses

def sphere():
    pass

load_configuration = False
load_model = False
load_weights = False
load_frames = False
load_poses = True
if load_configuration:
    print(f'Loading configuration ...')
    cfg = CfgLoader().get_cfg("cpu")
    if load_model:
        print('Loading model ...')
        model = cfg['model']
        if load_weights:
            print('Loading model weights ...')
            weights_file = cfg['weights_file']
            model.load_state_dict(torch.load(f'./{weights_file}'))

poses_file = f'{cfg["target_dir"]}/poses.txt' if load_configuration else 'targets/camera_to_image/poses.txt'
movie_file = f'{cfg["target_dir"]}/movie.mov' if load_configuration else 'targets/camera_to_image/movie.mov'

if load_frames:
    print(f'Loading frames: ./{movie_file} ...')
    frames = VideoCapture(f'./{movie_file}')

if load_poses:
    print(f'Loading poses: ./{poses_file} ...')
    poses = read_poses(f'./{poses_file}')
    poses = [[float(x) for x in l.split()[1:]] for l in  open(poses_file).read().splitlines()]
    poses = np.array(poses)

print('Finding bounding sphere ...')
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

point_fragment_shader = "point_fragment.glsl"
point_vertex_shader = "point_vertex.glsl"

vertices = np.concatenate((vertices, np.array([vertices[0, :]])))
point_model = PointModel(vertices)
point_view = PointCloudView(point_fragment_shader, point_vertex_shader)
eye = [0.0, 0.0, 2.0, 1.0]
at = [0.0, 0.0, 0.0, 1.0]
up = [0.0, 1.0, 0.0, 1.0]
fov = 45.0
near = 0.0001
far = 100
point_view.set_camera(eye, at, up, fov, near, far)

mesh_controller = GlfwController(width, height, xpos, ypos, title, point_view, point_model)
multi_controller.add(mesh_controller)

image_fragment_shader = "image_fragment.glsl"
image_vertex_shader = "image_vertex.glsl"

output_image = to_numpy_img(model(to_torch_pose(poses[0]))) if load_model else checker_board()

image_model = ImageModel(output_image)
image_view = ImageView(image_fragment_shader, image_vertex_shader)

image_controller = GlfwController(400, 300, 500, 100, "Image View", image_view, image_model)
multi_controller.add(image_controller)

multi_controller.run()
