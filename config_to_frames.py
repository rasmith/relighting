import numpy as np
import json
import argparse
from graphics_math import *

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--destination",
                    dest="destination",
                    help="destination file")
parser.add_argument("--source", dest="source", help="input file to convert")

args = parser.parse_args()

input_file = args.source
output_file = args.destination

d = json.loads(open(input_file).read())

dest_file = open(output_file, 'w')

sort_key = lambda k: int(k)
frame_number = 1
for i in sorted(d.keys(), key=sort_key):
    for j in sorted(d[i].keys(), key=sort_key):
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
        dest_file.write(
            f"{frame_number} {t[0]} {t[1]} {t[2]} {q[0]} {q[1]} {q[2]} {q[3]} "
            + f"{l[0]} {l[1]} {l[2]}\n")
        frame_number += 1
dest_file.close()
