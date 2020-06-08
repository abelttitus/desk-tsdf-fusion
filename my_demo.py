import time


import numpy as np
import cv2

_EPS = np.finfo(float).eps * 4.0
def transform44(l):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.
    
    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.
         
    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.array((
        (                1.0,                 0.0,                 0.0, t[0])
        (                0.0,                 1.0,                 0.0, t[1])
        (                0.0,                 0.0,                 1.0, t[2])
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], t[0]),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], t[1]),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], t[2]),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 573
  cam_intr = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  print "Camera Intrinsics",cam_intr
  cam_poses=np.loadtxt("data/pose_associateq.txt")
  print "Cam poses shape",cam_poses.shape
  vol_bnds = np.zeros((3,2))
	
  base_dir="/home/ashfaquekp/rgbd_dataset_freiburg1_desk/"
  file = open("associate.txt")
  data = file.read()
  lines = data.split("\n") 
  
  for line in lines:     #This is used to loop all images
    contents=line.split(" ")
    rgb_file=base_dir+contents[1]
    depth_file=base_dir+contents[3]
   
    print rgb_file
    print depth_file

    depth_im = cv2.imread(depth_file,-1)
    depth_im /= 5000. 
    print "Depth shape",depth_im.shape
    print "Depth max",np.max(depth_im)
    print "Depth min",np.min(depth_im)
    cv2.imshow("Depth",depth_im)

    cv2.waitKey()
    cv2.destroyAllWindows()
    
    rgb_im = cv2.imread(rgb_file)
    print "rgb max",np.max(rgb_im)
    print "rgb min",np.min(rgb_im)
    cv2.imshow("RGB",rgb_im)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    cam_pose=transform44(cam_poses[0,:])
    print cam_pose
     # depth is saved in 16-bit PNG in millimeters
    break

  
