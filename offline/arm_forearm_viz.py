import open3d as o3d
import numpy as np


#load the fused point cloud of the upper arm and the forearm
arm = o3d.io.read_point_cloud("/Users/computer/Desktop/final project HSI/ICP/arm_forearm/arm_fused.ply")
forearm = o3d.io.read_point_cloud("/Users/computer/Desktop/final project HSI/ICP/arm_forearm/forearm_fused.ply")

#paint the arm point cloud in red
arm.paint_uniform_color([1, 0, 0])

#paint the forearm point cloud in blue
forearm.paint_uniform_color([0, 0, 1])

#apply a manual translation to the forearm
#shift it by 2 cm along the negative x-axis
forearm.translate([-0.02, 0.0, 0.0])

#compute and print the centroid of the arm and forearm point clouds
#the centroid is obtained by averaging all point coordinates
print("Arm center:", np.mean(np.asarray(arm.points), axis=0))
print("Forearm center:", np.mean(np.asarray(forearm.points), axis=0))

#visualize both point clouds together
o3d.visualization.draw_geometries([arm, forearm], window_name="Before alignment")
