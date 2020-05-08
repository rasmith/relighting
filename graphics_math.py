import numpy as np

def perspective(fovy, aspect, near, far):
  f = 1.0 / np.tan(fovy * np.pi / 360.0);
  d = near - far;
  matrix = np.zeros((4, 4))
  matrix[0,0]= f / aspect
  matrix[1,1] = f
  matrix[2,2] = (near + far) / d
  matrix[2,3] = 2.0 * near * far / d
  matrix[3,2] = -1.0
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
  matrix[0:3, 3] = -matrix[0:3,0:3].dot(eye[0:3,0])
  return matrix

def rotate(angle, axis):
  x, y, z = axis.transpose()[0, :]
  K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype = np.float)
  I = np.eye(3, 3, dtype = np.float)
  R = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * K.dot(K)
  matrix = np.eye(4, 4, dtype = np.float)
  matrix[0:3, 0:3] = R
  return matrix

def quaternion(m):
  cos_theta = 0.5 * (np.trace(m) - 1)
  sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
  theta = np.arccos(cos_theta)
  x = 0.5 * m[2, 1] - m[1, 2] / sin_theta
  y = 0.5 * m[0, 2] - m[2, 0] / sin_theta
  z = 0.5 * m[1, 0] - m[0, 1] / sin_theta
  v = np.array([x, y, z])
  v = v / np.linalg.norm(v)
  c = np.cos(0.5 * theta)
  s = np.sin(0.5 * theta)
  q = np.array([c, v[0] * s, v[1] * s, v[2] * s])
  return q
  
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
