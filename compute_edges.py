#!/usr/bin/env python3
import numpy as np
import os
import glob 
from skimage import feature
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.io import imsave
from skimage import img_as_uint
import re
input_directory = "out"
output_directory = "edges"
images = glob.glob("%s/out*.png" % (input_directory))
if not os.path.exists('./edges'):
    os.mkdir('./edges')
print('%s' % (images[0]))
for image_path in images:
    img = imread(image_path)
    img = rgb2gray(img)
    edges = feature.canny(img)
    m = re.search('out-(\d+)-(\d+).png', image_path)
    print('m[0]=%s, m[1]=%s, m[2]=%s' % (m[0], m[1], m[2]))
    imsave('edges/edge-%s-%s.png' % (m[1], m[2]), img_as_uint(edges))
    
