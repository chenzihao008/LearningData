#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import math
#旋转矩阵转四元数需要pyquaternion包
from pyquaternion import Quaternion
#四元数转旋转矩阵需要scipy
from scipy.spatial.transform import Rotation as R
 
def isRotationMatrix(R) :
  Rt = np.transpose(R)
  shouldBeIdentity = np.dot(Rt, R)
  I = np.identity(3, dtype = R.dtype)
  n = np.linalg.norm(I - shouldBeIdentity)
  return n < 1e-6
#rotationMatrixToEulerAngles 用于旋转矩阵转欧拉角
def rotationMatrixToEulerAngles(R) :
 
  assert(isRotationMatrix(R))
   
  sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
   
  singular = sy < 1e-6
 
  if not singular :
    x = math.atan2(R[2,1] , R[2,2])
    y = math.atan2(-R[2,0], sy)
    z = math.atan2(R[1,0], R[0,0])
  else :
    x = math.atan2(-R[1,2], R[1,1])
    y = math.atan2(-R[2,0], sy)
    z = 0
 
  return np.array([x, y, z])
#eulerAnglesToRotationMatrix欧拉角转旋转矩阵
def eulerAnglesToRotationMatrix(theta) :
  R_x = np.array([[1,     0,         0          ],
          [0,     math.cos(theta[0]), -math.sin(theta[0]) ],
          [0,     math.sin(theta[0]), math.cos(theta[0]) ]
          ])
  R_y = np.array([[math.cos(theta[1]),  0,   math.sin(theta[1]) ],
          [0,           1,   0          ],
          [-math.sin(theta[1]),  0,   math.cos(theta[1]) ]
          ])
         
  R_z = np.array([[math.cos(theta[2]),  -math.sin(theta[2]),  0],
          [math.sin(theta[2]),  math.cos(theta[2]),   0],
          [0,           0,           1]
          ])
  R = np.dot(R_z, np.dot( R_y, R_x ))
  return R
#旋转矩阵转四元数
def rotateToQuaternion(rotateMatrix):
    q = Quaternion(matrix=rotateMatrix)
    return q
 
if __name__ == '__main__' :
  #初始化旋转矩阵
  rotationMat = np.array([[ -0.90748313, -0.30075654, -0.29329146],
                 [-0.05041803, -0.61514386, 0.78680115],
                 [ -0.41705203, 0.72879595, 0.54306912]])
 
 
  #旋转矩阵转欧拉角
  EulerAngles = rotationMatrixToEulerAngles(rotationMat)
  print("\nOutput Euler angles :\n{0}".format(EulerAngles))
 
 
  #欧拉角转旋转矩阵
  rotationMat_1 = eulerAnglesToRotationMatrix(EulerAngles)
  print ("\nR1 :\n{0}".format(rotationMat_1))
 
  #旋转矩阵转四元数
  Quaternion = rotateToQuaternion(rotationMat)
  print("四元数x为: ", Quaternion.x, "\n四元数y为: ", Quaternion.y, "\n四元数z为: ", Quaternion.z, "\n四元数w为: ", Quaternion.w)
 
 
  #四元数转旋转矩阵
  Rq = [Quaternion.x.astype(float), Quaternion.y.astype(float), Quaternion.z.astype(float), Quaternion.w.astype(float)]
  Rm = R.from_quat(Rq)
  rotation_matrix = Rm.as_matrix()
  print('rotation:\n', rotation_matrix)