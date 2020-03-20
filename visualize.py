import numpy as np

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

A = 2 * points
f = np.zeros((1, num_points))

for i in range(num_points):
    f[0, i] = np.linalg.norm(points[0:3, i])**2

C, res, rank, svals = np.linalg.lstsq(A.T, f.T)
