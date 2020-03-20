# importing libraries
import cv2 as cv
import numpy as np
import time
from math import log2
from matplotlib import pyplot as plt
from glfw_controller import *
from image_view import *

app = GlfwApp()
app.init()

multi_controller = GlfwMultiController()


class VideoCapture:
    def __init__(self, filename, down_sample=1):
        self.filename = filename
        self.cap = cv.VideoCapture(self.filename)
        self.down_sample = down_sample

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if ret != True:
            self.cap.release()
            raise StopIteration()
        if self.down_sample != 1:
            output = cv.resize(
                frame,
                (
                    int(
                        self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
                        // self.down_sample
                    ),
                    int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                    // self.down_sample,
                ),
            )
        else:
            output = frame
        return output

    def extents(self):
        return (
            int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH)) // self.down_sample,
            int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) // self.down_sample),
            int(self.cap.get(cv.CAP_PROP_FRAME_COUNT)),
        )


# Create a VideoCapture object and read from input file.
v = VideoCapture("videos/IMG_0174.m4v", down_sample=4)
start = time.time()
print("Loading frames...")
width, height, num_frames = v.extents()
print(f"width = {width}, height = {height}, num_frames = {num_frames}")
frames = [f for f in v]
end = time.time()
duration = end - start
print(f"{num_frames} frames loaded in {duration:.3f} seconds.")

print("Computing optical flow...")
start = time.time()
flow = [None] * len(frames)
for i, f in enumerate(frames):
    last = cv.cvtColor(f, cv.COLOR_BGR2GRAY) if i == 0 else current
    current = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    flow[i] = cv.calcOpticalFlowFarneback(
        last, current, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

end = time.time()
duration = end - start
print(
    f"Optical flow for {num_frames} frames computedin {duration:.3f} seconds."
)


print("Annotating frames...")
hsv = np.zeros_like(frames[0])
kp_frames = [None] * len(frames)
start = time.time()
for i, f in enumerate(flow):
    mag, ang = cv.cartToPolar(f[..., 0], f[..., 1])
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    kp_frames[i] = cv.cvtColor(hsv, cv.COLOR_HSV2BGR, cv.CV_8U)
    kp_frames[i] = cv.cvtColor(kp_frames[i], cv.COLOR_BGR2RGBA, cv.CV_8U)
duration = end - start
print(f"{num_frames} frames annotated in {duration:.3f} seconds.")


def get_frame(i):
    return kp_frames[i]


xpos, ypos, title = 0, 0, "Camera"
image_fragment_shader = "image_fragment.glsl"
image_vertex_shader = "image_vertex.glsl"
model = ImageModel((get_frame(i) for i in range(num_frames)))
view = ImageView(image_fragment_shader, image_vertex_shader)

image_controller = GlfwController(width, height, xpos, ypos, title, view, model)
multi_controller.add(image_controller)
multi_controller.run()
