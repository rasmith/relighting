import numpy as np
from scipy.spatial.transform import Rotation as R


def perspective(fovy, aspect, near, far):
    f = 1.0 / np.tan(fovy * np.pi / 360.0)
    d = near - far
    matrix = np.zeros((4, 4))
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (near + far) / d
    matrix[2, 3] = 2.0 * near * far / d
    matrix[3, 2] = -1.0
    return matrix


def lookat(eye, at, up):
    z_direction = eye[0:3, 0] - at[0:3, 0]
    z_direction = z_direction / np.linalg.norm(z_direction)
    x_direction = np.cross(up[0:3, 0] / np.linalg.norm(up[0:3, 0]), z_direction)
    x_direction = x_direction / np.linalg.norm(x_direction)
    y_direction = np.cross(z_direction, x_direction)
    y_direction = y_direction / np.linalg.norm(y_direction)
    matrix = np.eye(4)
    matrix[0, 0:3] = x_direction
    matrix[1, 0:3] = y_direction
    matrix[2, 0:3] = z_direction
    matrix[0:3, 3] = -matrix[0:3, 0:3].dot(eye[0:3, 0])
    return matrix


def rotate(angle, axis):
    # if isinstance(axis, np.ndarray):
        # axis = axis.flatten().tolist()
    # if isinstance(axis, list) and len(axis) >= 3:
        # x, y, z = axis[:3]
    # r = R.from_rotvec(angle * np.array(axis))
    # print(r)
    # matrix = np.eye(4,4, dtype = np.float)
    # matrix[0:3, 0:3] = r[0]
    # return matrix
    if isinstance(axis, np.ndarray):
        axis = axis.flatten().tolist()
    if isinstance(axis, list) and len(axis) >= 3:
        x, y, z = axis[:3]
    else:
        print(f'axis = {axis}')
        raise f"Need type to be list or ndarray and have at least 3 values."
    mag = np.sqrt(x * x + y * y + z * z)
    np.seterr(all='raise')
    x /= mag
    y /= mag
    z /= mag
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float)
    I = np.eye(3, 3, dtype=np.float)
    R = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * K.dot(K)
    matrix = np.eye(4, 4, dtype=np.float)
    matrix[0:3, 0:3] = R
    return matrix

def quaternion(m):
    if isinstance(m, list):
        m = np.reshape(np.array(m[:9], dtype=np.float32, order = 'C'), (3, 3))
    r = R.from_matrix(m[:3, :3])
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])
    return q

def quaternion_to_matrix(q):
    mag = np.linalg.norm(q[1:4])
    if mag == 0.0:
        angle = 0
        axis = [1, 0, 0]
    else:
        angle = 2.0 * np.arctan2(mag, q[0])
        axis = q[1:4] / mag
    # print(f'q2m: q={q},  angle = {angle}, axis = {axis}')
    return rotate(angle, axis)

def uniform_scale(value):
    scale_matrix = np.eye(4)
    scale_matrix[0, 0] = value
    scale_matrix[1, 1] = value
    scale_matrix[2, 2] = value
    return scale_matrix


def translate(tx, ty, tz):
    translate_matrix = np.eye(4)
    translate_matrix[0, 3] = tx
    translate_matrix[1, 3] = ty
    translate_matrix[2, 3] = tz
    return translate_matrix
