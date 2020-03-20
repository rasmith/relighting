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
v = VideoCapture("calib/IMG_0182.MOV", down_sample=4)
start = time.time()
print("Loading frames...")
width, height, num_frames = v.extents()
print(f"width = {width}, height = {height}, num_frames = {num_frames}")
frames = [f for f in v]
end = time.time()
duration = end - start
print(f"{num_frames} frames loaded in {duration:.3f} seconds.")

print("Computing calibration data...")
start = time.time()

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

for f in frames:
    gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

end = time.time()
duration = end - start
print(
    f"Calibration data for {num_frames} frames computed in {duration:.3f} seconds."
)

print("Calibrating ...")

start = time.time()

print(f"type(objpoints)={objpoints[0].shape}")
print(f"type(imgpoints)={imgpoints[0].shape}")
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

end = time.time()
duration = end - start
print(f"Calibration for {num_frames} frames computed in {duration:.3f} seconds.")


# def get_frame(i):
    # return kp_frames[i]


# xpos, ypos, title = 0, 0, "Camera"
# image_fragment_shader = "image_fragment.glsl"
# image_vertex_shader = "image_vertex.glsl"
# model = ImageModel((get_frame(i) for i in range(num_frames)))
# view = ImageView(image_fragment_shader, image_vertex_shader)

# image_controller = GlfwController(width, height, xpos, ypos, title, view, model)
# multi_controller.add(image_controller)
# multi_controller.run()
