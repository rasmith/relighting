import numpy as np
import cv2
from video_capture import VideoCapture
from glfw_controller import *
from image_view import *

from mesh_view import MeshModel
from mesh_view import MeshView

def random_normal(d):
    u = np.random.rand(1, d)
    return u / np.linalg.norm(u)
def random_sphere():
    c = np.random.rand(1, 3)    
    u = random_normal(3)
    v = random_normal(3)
    v = np.cross(u, v)  
    v = v / np.linalg.norm(v)
    return u, v, c

u, v, c = random_sphere()
n = np.cross(u, v)
n = n / np.linalg.norm(n)

r = 3
num_points = 10
points = np.zeros((4, num_points))
for i in range(num_points):
    theta, phi = np.pi * (np.random.ranf(2) - 0.5)
    p = c + r * n * np.sin(phi)
    p += r * u * np.cos(phi) * np.cos(theta)
    p += r * v * np.cos(phi) * np.sin(theta)
    points[0:3, i] = p

A = 2 * points
A[3, :] = 1
C=np.append(c, r * r - np.linalg.norm(c))
f = np.zeros((1, num_points))

for i in range(num_points):
    f[0, i] = np.linalg.norm(points[0:3, i])**2

C, res, rank, svals = np.linalg.lstsq(A.T, f.T, rcond = None)
print(f"C = {C}, R = {((np.linalg.norm(C[0:3]) ** 2) + C[3]) ** (1/2)}")
print(f"c = {c.T}, r = {r}")

app = GlfwApp()
app.init()

multi_controller = GlfwMultiController()
width, height = 640, 480
xpos, ypos, title = 0, 0, "Camera"
msh_fragment_shader = "mesh_fragment.glsl"
mesh_vertex_shader = "mesh_vertex.glsl"
model = MeshModel("cow.off")
view = MeshView(mesh_fragment_shader, mesh_vertex_shader)
eye = [0.0, 0.0, 2.0, 1.0]
at = [0.0, 0.0, 0.0, 1.0]
up = [0.0, 1.0, 0.0, 1.0]
fov = 45.0
near = 0.0001
far = 100
light_position = [0.0, 5.0, 1.0];
view.set_camera(eye, at, up, fov, near, far)
view.set_light_position(light_position)


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


