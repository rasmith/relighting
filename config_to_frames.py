import numpy as np
import json
import argparse
from graphics_math import *
from pyquaternion import Quaternion

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--destination", dest="destination", help="destination file"
)
parser.add_argument("--source", dest="source", help="input file to convert")


args = parser.parse_args()

input_file = args.source
output_file = args.destination

d = json.loads(open(input_file).read())

f = lambda k: int(k)
frame_number = 1
for i in sorted(d.keys(), key=f):
    for j in sorted(d[i].keys(), key=f):
        # print(f'i = {i} j = {j} {d[i][j].keys()}')
        info = d[i][j]
        m = lookat(
            np.array([info["eye"]]).T,
            np.array([info["at"]]).T,
            np.array([info["up"]]).T,
        )
        l = info["light"]
        r = m[0:3, 0:3]
        q = quaternion(r)
        t = m[0:3, 3]
        print(
            f"{frame_number} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]} {l[0]} {l[1]} {l[2]}"
        )
        frame_number += 1
