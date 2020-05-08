import cv2
import glfw
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
from mesh_view import MultiMeshModel
from mesh_view import MultiMeshView
from multiprocessing import Process
from video_capture import VideoCapture
from torch.autograd import Variable

from tasks.camera_to_image import CfgLoader

import graphics_math as gm


def render_image(model, pose):
    pose = torch.from_numpy(
        np.reshape(poses[0], (1, 1, 1, 7)).astype(np.float32)
    )
    pose = Variable(pose).cpu()
    img = model(pose)
    return img


def checker_board():
    return cv2.cvtColor(cv2.imread("checkerboard.jpg"), cv2.COLOR_BGR2RGB)


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


def axes():
    return (
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
        np.array([0, 1, 0, 2, 0, 3]),
        np.array(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        ),
    )


def update_axes(which, angle, orientation):
    axis = orientation[:, which]
    transformation = gm.rotate(angle, axis)
    orientation = transformation.dot(orientation)
    return orientation


load_configuration = True
load_model = True 
load_weights = True 
load_frames = True 
load_poses = True 
if load_configuration:
    print(f"Loading configuration ...")
    cfg = CfgLoader().get_cfg("cpu")
    if load_model:
        print("Loading model ...")
        model = cfg["model"]
        if load_weights:
            print("Loading model weights ...")
            weights_file = cfg["weights_file"]
            model.load_state_dict(torch.load(f"./{weights_file}"))

poses_file = (
    f'{cfg["target_dir"]}/poses.txt'
    if load_configuration
    else "targets/camera_to_image/poses.txt"
)
movie_file = (
    f'{cfg["target_dir"]}/movie.mov'
    if load_configuration
    else "targets/camera_to_image/movie.mov"
)

if load_frames:
    print(f"Loading frames: ./{movie_file} ...")
    frames = VideoCapture(f"./{movie_file}")

if load_poses:
    print(f"Loading poses: ./{poses_file} ...")
    poses = read_poses(f"./{poses_file}")
    poses = [
        [float(x) for x in l.split()[1:]]
        for l in open(poses_file).read().splitlines()
    ]
    poses = np.array(poses)

print("Finding bounding sphere ...")
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
    f[0, i] = np.linalg.norm(points[0:3, i]) ** 2

C, res, rank, svals = np.linalg.lstsq(A.T, f.T, rcond=None)
radius = (np.linalg.norm(C[0:3]) ** 2 + C[3]) ** (1 / 2)
print(f"C = {C}, R = {((np.linalg.norm(C[0:3]) ** 2) + C[3]) ** (1/2)}")
print(f"C[0] = {C[0]}, C[1] = {C[1]}, C[2] = {C[2]}")

app = GlfwApp()
app.init()

multi_controller = GlfwMultiController()
width, height = 640, 480
xpos, ypos, title = 0, 0, "Camera"

point_fragment_shader = "point_fragment.glsl"
point_vertex_shader = "point_vertex.glsl"

vertices = np.concatenate((vertices, np.array([vertices[0, :]])))
big_point = np.array([[0, 0, 0]])
multi_mesh_model = MultiMeshModel(
    [
        {
            "name": "bounding_sphere",
            "type": "mesh",
            "mesh": "sphere.obj",
            "M": gm.uniform_scale(5.0).dot(gm.translate(0, -0.1, 0)),
            "fragment": "wireframe_fragment.glsl",
            "vertex": "wireframe_vertex.glsl",
            "geometry": "wireframe_geometry.glsl",
            "color": np.array([1.0, 1.0, 1.0]),
            "opacity": 0.5,
        },
        {
            "name": "points",
            "type": "points",
            "mesh": vertices,
            "M": gm.uniform_scale(0.5 / radius).dot(
                gm.translate(-C[0], -C[1], -C[2])
            ),
            "fragment": "point_fragment.glsl",
            "vertex": "point_vertex.glsl",
            "geometry": None,
            "color": np.array([1.0, 0.0, 0.0]),
        },
        {
            "name": "axes",
            "type": "lines",
            "mesh": axes(),
            "R": np.eye(4),
            "T": gm.translate(0, 0, 0),
            "scale": gm.uniform_scale(0.20),
            "M": gm.uniform_scale(0.20),
            "fragment": "line_fragment.glsl",
            "vertex": "line_vertex.glsl",
            "geometry": "line_geometry.glsl",
            "color": np.array([0.0, 1.0, 0.0]),
        },
    ]
)


class KeyCallbackHandler:
    def __init__(self, data):
        self.data = data

    def update_orientation(self, name, which, angle):
        obj = self.data.name_to_mesh_info[name]
        R = obj["R"]
        axis = np.expand_dims(R[:, which][0:3], 0).T
        # import pdb
        # pdb.set_trace()
        M = gm.rotate(angle, axis)
        obj["R"] = gm.rotate(angle, axis).dot(R)

    def update_translation(self, name, tx, ty, tz):
        obj = self.data.name_to_mesh_info[name]
        obj["T"] = gm.translate(tx, ty, tz).dot(obj["T"])

    def update_model_matrix(self, name):
        obj = self.data.name_to_mesh_info[name]
        obj["M"] = np.linalg.multi_dot([obj["T"], obj["R"], obj["scale"]])

    def key_handler(self, key, scancode, action, mods):
        if key == glfw.KEY_W and action == glfw.PRESS:
            self.update_translation("axes", 0, 0, 0.025)
        elif key == glfw.KEY_A and action == glfw.PRESS:
            self.update_translation("axes", -0.025, 0, 0)
        elif key == glfw.KEY_S and action == glfw.PRESS:
            self.update_translation("axes", 0, 0, -0.025)
        elif key == glfw.KEY_D and action == glfw.PRESS:
            self.update_translation("axes", 0.025, 0, 0.0)
        elif key == glfw.KEY_R and action == glfw.PRESS:
            self.update_translation("axes", 0, 0.025, 0.0)
        elif key == glfw.KEY_F and action == glfw.PRESS:
            self.update_translation("axes", 0, -0.025, 0.0)
        elif key == glfw.KEY_U and action == glfw.PRESS:
            self.update_orientation("axes", 0, 0.25)
        elif key == glfw.KEY_J and action == glfw.PRESS:
            self.update_orientation("axes", 0, -0.25)
        elif key == glfw.KEY_H and action == glfw.PRESS:
            self.update_orientation("axes", 1, 0.25)
        elif key == glfw.KEY_K and action == glfw.PRESS:
            self.update_orientation("axes", 1, -0.25)
        elif key == glfw.KEY_O and action == glfw.PRESS:
            self.update_orientation("axes", 2, 0.25)
        elif key == glfw.KEY_L and action == glfw.PRESS:
            self.update_orientation("axes", 2, -0.25)
        elif key == glfw.KEY_SPACE and action == glfw.PRESS:
            print("Render!")
        obj = self.data.name_to_mesh_info["axes"]
        self.update_model_matrix("axes")


multi_mesh_view = MultiMeshView()
eye = [0.0, 0.0, 2.0, 1.0]
at = [0.0, 0.0, 0.0, 1.0]
up = [0.0, 1.0, 0.0, 1.0]
fov = 45.0
near = 0.0001
far = 100
light_position = [0.0, 0.0, 4.0]
multi_mesh_view.set_camera(eye, at, up, fov, near, far)
multi_mesh_view.set_light_position(light_position)

mesh_controller = GlfwController(
    width, height, xpos, ypos, title, multi_mesh_view, multi_mesh_model
)
mesh_controller.register_user_key_callback(KeyCallbackHandler(multi_mesh_model))
multi_controller.add(mesh_controller)

image_fragment_shader = "image_fragment.glsl"
image_vertex_shader = "image_vertex.glsl"

output_image = (
    to_numpy_img(model(to_torch_pose(poses[0])))
    if load_model
    else checker_board()
)

image_model = ImageModel(output_image)
image_view = ImageView(image_fragment_shader, image_vertex_shader)

image_controller = GlfwController(
    400, 300, 500, 100, "Image View", image_view, image_model
)
multi_controller.add(image_controller)

multi_controller.run()
