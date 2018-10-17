#!/usr/bin/env python3
import numpy as np
import graphics_math as gm
import os
import sys
import subprocess
from multiprocessing import Pool
import json
import codecs, json
import time

  # V = up_rotation.dot(V)
  # forward = np.array([V[2, 0:3]])
  # eye  = at - camera_distance * forward

class CameraSampler(object):
  def __init__(self, eye, at, up, num_up_samples, num_right_samples,\
               up_speed, right_speed):
    self.radius = np.linalg.norm(eye - at)
    self.eye = eye
    self.up =  up
    self.at = at
    self.num_up_samples = num_up_samples
    self.num_right_samples = num_right_samples
    self.up_speed = up_speed
    self.right_speed = right_speed
    self.lookat = gm.lookat(self.eye.transpose(), self.at.transpose(),\
                                 self.up.transpose())
    self.view_matrix = np.linalg.inv(self.lookat)
    self.total_samples = num_right_samples * num_up_samples
    self.count = 0
    print("self.view_matrix = \n%s" % (str(self.view_matrix)))
    print("self.eye = %s" % (str(self.eye)))

  def __iter__(self):
    return self

  def __next__(self):
    if self.count < self.total_samples:
      # compute indices
      i  = self.count // self.num_up_samples
      j = self.count % self.num_up_samples
      print("i = %d, j = %d" % (i, j))
      current_view = self.view_matrix
      old_up = np.array([current_view[0:3, 1]])
      right = np.array([current_view[0:3, 0]])
      # rotate about right
      rotation = gm.rotate(i * self.right_speed, right.transpose())
      current_view = rotation.dot(current_view)
      # rotate about up
      rotation = gm.rotate(j * self.up_speed, old_up.transpose())
      current_view = rotation.dot(current_view)
      forward = np.array([current_view[0:3, 2]])
      new_eye  = self.at + self.radius * forward
      new_up = np.array([current_view[0:3, 1]])
      self.count = self.count + 1
      print("eye = %s" % str(new_eye))
      print("up= %s" % str(new_up))
      print("at= %s" % str(self.at))
      return new_eye, self.at, new_up
    else:
      raise StopIteration()
    

def call_embree(data):
  i, j, samples, infile, outfile, isa, num_threads, verbose = data
  size, eye, at, up, fov, light, color, spp = samples[i][j].values()
  start_time = time.time()
  subprocess.call([bin_path,
                       "-i", "crown/crown.xml",
                       "-o", "out/out-%04d-%04d.ppm" % (i, j),
                       "--size", str(size[0]), str(size[1]),
                       "--vp", str(eye[0]), str(eye[1]), str(eye[2]),
                       "--vi", str(at[0]), str(at[1]), str(at[2]),
                       "--vu", str(up[0]), str(up[1]), str(up[2]),
                       "--fov", str(fov),
                       "--pointlight", 
                       str(light[0]), str(light[1]), str(light[2]),
                       str(color[0]), str(color[1]), str(color[2]),
                       "--threads", str(num_threads),
                       "--isa", isa,
                       "--spp", str(spp),
                      "--shader", "default",
                       "--verbose", str(verbose)])
  end_time = time.time()
  print("(%d, %d) render time = %f" % (i, j, end_time - start_time))


file_name = 'crown.xml' # -i
size = np.array([[256, 256]], dtype = np.int) # -size
eye =  np.array([[61.4156, 102.11, -1325.25]], dtype = np.float) # = -vp
at = np.array([[3.72811, 87.9981, -82.1874]], dtype = np.float) # = -vi
up = np.array([[-0.000525674, 0.999936, 0.0113274]], dtype = np.float) # -vu
fov = 10.1592 # - fov
light = eye # - pointlight
color = np.array([[10000000, 10000000, 10000000]])
num_threads = 8# --threads 
isa = 'sse4.2' # --isa
max_path_length = 4 # --max-path-length 
samples_per_pixel = 128 # --spp
verbose = 0 # --verbose 

num_up_samples = 100# rotations about up
num_right_samples = 100# rotations about forward
num_light_samples = 1

M = gm.lookat(eye.transpose(), at.transpose(), up.transpose())
V = np.linalg.inv(M)
right = V[0:3, 2]
print("M = %s" % (str(M)))
print("V = %s" % (str(V)))
e = V.dot(np.array([0, 0, 0, 1], dtype = np.float))
print("V * [0 0 0 1] = %s" % str(e))
up_rotation_speed = 2.0 * np.pi / num_up_samples
up_rotation = gm.rotate(up_rotation_speed, up.transpose())
right_rotation_speed = 2.0 * np.pi / num_right_samples
right = np.array([V[0:3, 0]])
right_rotation = gm.rotate(right_rotation_speed, right.transpose())
light_rotation = gm.rotate(up_rotation_speed, right.transpose())

bin_path = "/usr/bin/embree3/pathtracer"
samples = {}
camera_sampler = CameraSampler(eye, at, up, num_up_samples,\
                               num_right_samples, up_rotation_speed,\
                               right_rotation_speed)
i = 0
for eye, at, up in camera_sampler:
  light  =  np.array([[61.4156, 102.11, -1325.25]], dtype = np.float) 
  samples[i] = {}
  for j in range(num_light_samples):
    # print("render # %d, %d" % (i, j))
    # print("V = %s" % str(V))
    # print("eye = %s" % str(eye))
    # print("light = %s" % str(light))
    samples[i][j] = {'size':size.flatten().tolist(),
                     'eye': eye.flatten().tolist(), 
                     'at': at.flatten().tolist(), 
                     'up': up.flatten().tolist(), 
                     'fov': fov,
                     'light':light.flatten().tolist(),
                     'color':color.flatten().tolist(),
                     'spp':samples_per_pixel}   
    light = np.array([light_rotation[0:3, 0:3].dot(light[0, :])])
  i = i + 1

num_camera_samples = camera_sampler.total_samples
start_rendering = time.time()
pool = Pool(processes = 4)
pool.map(call_embree, [(i, j, samples, \
                       "crown/crown.xml", "out/out-%04d-%04d.ppm" % (i, j), \
                       isa, num_threads, verbose) \
                       for i in range(num_camera_samples) \
                       for j in range(num_light_samples)])

stop_rendering = time.time()
total_render_time =  stop_rendering - start_rendering
num_images = num_light_samples * num_camera_samples
print("total time = %f, num_images = %d, time per image = %f" % \
      (total_render_time, num_images, total_render_time / num_images))

max_path_length = 1 # --max-path-length 
samples_per_pixel = 1 # --spp
num_threads = 24
eye =  np.array([[61.4156, 102.11, -1325.25]], dtype = np.float) # = -vp
at = np.array([[3.72811, 87.9981, -82.1874]], dtype = np.float) # = -vi
up = np.array([[-0.000525674, 0.999936, 0.0113274]], dtype = np.float) # -vu
camera_sampler = CameraSampler(eye, at, up,  num_up_samples,\
                               num_right_samples, up_rotation_speed,\
                               right_rotation_speed)
i = 0
for eye, at, up, in camera_sampler:
  print("normal# %d" % (i))
  print("eye = %s" % str(eye))
  subprocess.call([bin_path,
                       "-i", "crown/crown.xml",
                       "-o", "out/normal_%04d.ppm" % (i),
                       "--size", str(size[0, 0]), str(size[0, 1]),
                       "--vp", str(eye[0, 0]), str(eye[0, 1]), str(eye[0, 2]),
                       "--vi", str(at[0, 0]), str(at[0, 1]), str(at[0, 2]),
                       "--vu", str(up[0, 0]), str(up[0, 1]), str(up[0, 2]),
                       "--fov", str(fov),
                       "--threads", str(num_threads),
                       "--isa", isa,
                       "--spp", str(samples_per_pixel),
                      "--shader", "Ng",
                       "--verbose", str(verbose)])
  i = i + 1

json.dump(samples, codecs.open("config.cfg", 'w', encoding='utf-8'),\
    separators=(',', ':'), sort_keys=True, indent=4)


