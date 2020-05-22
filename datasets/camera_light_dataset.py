import cfg
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from graphics_math import lookat
import re
from video_capture import VideoCapture
import cv2 as cv
from scipy.spatial.transform import Rotation as R

def normalize_numpy_array(values):
    max_vals = values[0]
    min_vals = values[0]
    for m in values:
        max_vals = np.maximum(max_vals, m)
        min_vals = np.minimum(min_vals, m)
    divisor = np.maximum(max_vals - min_vals, np.ones(max_vals.shape))
    normalized_values = (values - min_vals) / divisor - 0.5
    return normalized_values


class CameraLightDataset(data.Dataset):
    def __init__(self, target_dir, task_name, transform = None, target_transform = None):
        self.target_dir = target_dir
        self.task_name = task_name
        self.transform = transform
        self.target_transform = target_transform
        self.load()
    
    def text_to_frame_and_pose(self, text):
      tokens = text.split()
      return int(tokens[0]), np.array([float(x) for x in tokens[1:]])

    def load_video(self, video_file):
      v  = VideoCapture(video_file)
      self.frames = [f for f in v]
      

    def load_poses(self, pose_file):
      lines = open(pose_file, 'r').read().splitlines() 
      self.poses = [self.text_to_frame_and_pose(l) for l in lines]

    def load(self):
        video_file = f'{self.target_dir}/movie.mov'
        pose_file = f'{self.target_dir}/poses.txt'
        self.load_video(video_file)
        self.load_poses(pose_file)
        self.temp = []
        for i, p in self.poses:
          resized = cv.resize(self.frames[i-1], dsize=(128,128))
          self.temp.append(resized)
        self.frames = self.temp
        del self.temp

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx][1]
        t, q, l = pose[0:3], pose[3:7], pose[7:]
        r = R.from_quat(q)
        a = r.as_rotvec()
        pose = np.concatenate((t, a, l), axis = 0)
        pose = np.reshape(pose, (1, len(pose))).astype(np.float32) 
        pose = torch.from_numpy(pose).view(1, 1, 9)
        
        # pose = np.reshape(self.poses[idx][1], (1, len(self.poses[idx][1]))).astype(np.float32) 
        # pose = torch.from_numpy(pose).view(1, 1, 10)
        image = self.frames[idx]
        if self.transform is not None:
            pose = self.transform(pose)
        if self.target_transform is not None:
            image = self.target_transform(image)
        return pose, image
