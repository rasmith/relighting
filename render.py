#!/usr/bin/env python3
from multiprocessing import Pool, Lock, Manager
from threading import Thread
from wand.image import Image
import codecs
import curses
import datetime
import graphics_math as gm
import json
import json
import numpy as np
import os
import subprocess
import sys
import time


shader_name_to_shader_arg = {
    'ambient_occlusion':'ao',
    'curvature':'curvature',
    'default':'default',
    'depth':'depth',
    'diffuse_albedo':'diffuse_albedo',
    'normals':'Ng',
    'specular_albedo':'specular_albedo'
}
  # V = up_rotation.dot(V)
  # forward = np.array([V[2, 0:3]])
  # eye  = at - camera_distance * forward


class Sampler(object):
  def __init__(self, eye, at, up, num_up_samples, num_right_samples,\
               up_speed, right_speed, sampler_type):
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
    self.sampler_type = sampler_type
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
      # print("i = %d, j = %d" % (i, j))
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
      # print("eye = %s" % str(new_eye))
      # print("up= %s" % str(new_up))
      # print("at= %s" % str(self.at))
      if self.sampler_type == 'camera':
        return new_eye, self.at, new_up
      else:
        return new_eye

    else:
      raise StopIteration()
    
def call_embree2(data):
  shader, lock, d, i, j, samples, infile, outfile, isa, num_threads, verbose = data
  size, eye, at, up, fov, light, color, spp = samples[i][j].values()
  start_time = time.time()
  args = [bin_path,
          "-i", infile,
          "-o", outfile,
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
          "--shader", shader_name_to_shader_arg[shader],
          "--verbose", str(verbose)]
  # subprocess.call(args)
  process = subprocess.Popen(args)
  out, err = process.communicate()
  d['out']=out
  d['err']=err
  d['errcode']= process.returncode
  d['cmd']=' '.join(args)
  end_time = time.time()
  # f = open(outfile, 'wb')
  # with Image(filename = outfile) as img:
    # with img.convert('png') as converted:
      # converted.save(filename = outfile.replace('ppm', 'png')) 
  t = end_time - start_time
  convert_bin_path = '/usr/bin/convert'
  outfile_png = outfile.replace('ppm','png')
  args=[convert_bin_path, outfile, outfile_png]
  subprocess.call(args)
  # print(f'({i}, {j}) render time = {t}')
  lock.acquire()
  d['render_time'] = t
  d['count'] = d['count'] + 1
  d['i'] = i
  d['j'] = j
  d['changed']=True
  d['outfile']=outfile_png
  lock.release()



def get_output_file(shader, i, j):
  if shader == 'default':
    return f'out/out-{i:04d}-{j:04d}.ppm'
  return f'targets/{shader}/{shader}-{i:04d}-0000.ppm'

# class RenderMonitor(object):
  # def __init__(self, total):
    # self.count = 0
    # self.total = total
    # # self.stdscr = curses.initscr()
    # self.changed = False
    # self.message = None
    # # curses.noecho()
    # # self.stdscr.keypad(1)
    # # curses.nocbreak()
    # # self.stdscr.keypad(0)
    # # curses.echo()

  # def __del__(self):
    # pass
    # # curses.endwin()

  # def display(self):
    # # print(f'new message = {self.message} changed = {self.changed}')
    # # with self.lock:
      # # print(self.message)
      # # if self.changed:
        # # stdscr.addstr(0, 0, self.message)
        # # stdscr.refresh()
        # # print(self.message)
        # # self.changed = False
    # # print(f'current message = {self.message} changed = {self.changed}')

def render_update_thread(l, d, start, stdscr):
  done = False
  while not done:
    l.acquire()
    if d['changed'] == True:
      i = d['i']
      j = d['j']
      t = d['render_time']
      p = 100.0 * d['count'] / d['total']
      elapsed = time.time() - start
      sec_per_image = elapsed / d['count']
      outfile = d['outfile']
      count = d['count']
      total = d['total']
      err=d['err']
      out=d['out']
      cmd=d['cmd']
      errcode=d['errcode']
      elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
      # message = f'{outfile} rendered in {t:1.2f} seconds {p:3.2f}% {sec_per_image:2.2f} s/img'
      stdscr.addstr(0, 0, f'progress = {p:3.2f}%        {count}/{total}')
      stdscr.addstr(1, 0, f'outfile = {outfile}')
      stdscr.addstr(2, 0, f'time = {t:1.2f} seconds')
      stdscr.addstr(3, 0, f'{sec_per_image:2.2f} sec/img')
      stdscr.addstr(5, 0, f'out = {out}')
      stdscr.addstr(6, 0, f'err = {err}')
      stdscr.addstr(7, 0, f'errcode = {errcode}')
      stdscr.addstr(9, 0, f'elapsed = {elapsed_str}')
      stdscr.addstr(11, 0, f'cmd={cmd}')
      if d['count'] == d['total']:
        done = True

      stdscr.refresh()
      # print(f'{message}')
      d['changed'] = False
    l.release()
    time.sleep(0.1)

def render(shader, lock, d, samples):
  input_file = 'scenes/crown/crown.xml'
  start_rendering = time.time()
  pool.map(call_embree2, [(shader, lock, d, i, j, samples, \
                         input_file, get_output_file(shader, i, j), \
                         isa, num_threads, verbose) \
                         for i in range(num_camera_samples) \
                         for j in range(num_light_samples)])
  stop_rendering = time.time()
  total_render_time =  stop_rendering - start_rendering
  num_images =  camera_sampler.total_samples

if __name__ == '__main__':
  file_name = 'crown.xml' # -i
  size = np.array([[128, 128]], dtype = np.int) # -size
  eye =  np.array([[61.4156, 102.11, -1325.25]], dtype = np.float) # = -vp
  at = np.array([[3.72811, 87.9981, -82.1874]], dtype = np.float) # = -vi
  up = np.array([[-0.000525674, 0.999936, 0.0113274]], dtype = np.float) # -vu
  fov = 10.1592 # - fov
  # light = eye # - pointlight
  light  =  np.array([[61.4156, 102.11, -1325.25]], dtype = np.float) 
  color = np.array([[10000000, 10000000, 10000000]])
  num_threads = 8# --threads 
  isa = 'sse4.2' # --isa
  max_path_length = 4 # --max-path-length 
  samples_per_pixel = 128 # --spp
  verbose = 0 # --verbose 

  # num_up_samples = 100# rotations about up
  # num_right_samples = 100# rotations about forward
  # num_light_samples = 1

  num_up_samples = int(sys.argv[2])# rotations about up
  num_right_samples = int(sys.argv[3])# rotations about forward
  num_light_samples = int(sys.argv[4])

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

  # bin_path = "/usr/bin/embree3/pathtracer"
  bin_path = "/home/agrippa/git/embree/build/pathtracer"
  samples = {}
  camera_sampler = Sampler(eye, at, up, num_up_samples,\
                                 num_right_samples, up_rotation_speed,\
                                 right_rotation_speed, 'camera')
  i = 0
  for eye, at, up in camera_sampler:
    light_sampler = Sampler(light, at, up, num_up_samples,\
                                 num_right_samples, up_rotation_speed,\
                                 right_rotation_speed, 'light')
    # light  =  np.array([[61.4156, 102.11, -1325.25]], dtype = np.float) 
    samples[i] = {}
    for current_light in light_sampler:
    # for j in range(num_light_samples):
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

  #setup ncurses
  stdscr = curses.initscr()
  curses.noecho()
  curses.cbreak()
  stdscr.keypad(True)
  stdscr.clear()


  manager = Manager()
  start = time.time()
  d = manager.dict()
  d['render_time'] = 0.0
  d['count'] = 0
  d['i'] = 0
  d['j'] = 0
  d['changed']=False
  d['total'] = float(camera_sampler.total_samples)
  d['outfile']=''
  lock = manager.Lock()
  pool = Pool(processes = 8)
  message = None
  t = Thread(target = render_update_thread, args=(lock, d, start, stdscr))
  t.start()
  shader = sys.argv[1]
  render(shader, lock, d, samples)
  t.join()
  end = time.time()
  total_render_time = end - start
  num_images = camera_sampler.total_samples

  #unsetup ncurses
  curses.nocbreak()
  stdscr.keypad(False)
  curses.echo()
  curses.endwin()

  print("total time = %f, num_images = %d, time per image = %f" % \
              (total_render_time, num_images, total_render_time / num_images))

  if shader == 'default':
    json.dump(samples, codecs.open("config.cfg", 'w', encoding='utf-8'),\
            separators=(',', ':'), sort_keys=True, indent=4)

  exit()
