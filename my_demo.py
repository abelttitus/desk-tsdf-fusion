import time


import numpy as np
import cv2
import fusion

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
  
  index=0
  for line in lines:     #This is used to loop all images
    contents=line.split(" ")
    try:
      rgb_file=base_dir+contents[1]
      depth_file=base_dir+contents[3]
    except:
      continue

    depth_im = cv2.imread(depth_file,-1)
    depth_im=depth_im.astype("float")
    depth_im /= 5000. 
    

    cam_pose=transform44(cam_poses[index,:])
    index+=1
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  print "Volume Bounds:",vol_bnds
  file.close()

  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()

  file = open("associate.txt")
  data = file.read()
  lines = data.split("\n") 
  
  i=0
  for line in lines:     #This is used to loop all images
    contents=line.split(" ")
    print("Fusing frame %d/%d"%(i+1, n_imgs))
    try:
      rgb_file=base_dir+contents[1]
      depth_file=base_dir+contents[3]
    except:
      continue

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread(depth_file,-1).astype(float)
    depth_im /= 5000.
    #depth_im[depth_im == 65.535] = 0
    cam_pose=transform44(cam_poses[i,:])
    

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh-desk1.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc-desk1.ply", point_cloud)

    

  
