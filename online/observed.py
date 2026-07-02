import numpy as np
import open3d as o3d
import os

#path of the newobserved image
OBSERVED_PATH = "/Users/computer/Desktop/final project HSI/observed/observed3.npy"
SAVE_PATH = "/Users/computer/Desktop/observed_cleaned.ply"

#camera intrinsic parameters
FX_DEPTH = 504.5604248046875
FY_DEPTH = 504.5923767089844
CX_DEPTH = 321.67755126953125
CY_DEPTH = 327.68536376953125

#depth parameters
MAX_DEPTH = 900
MIN_DEPTH = 200

VOXEL_SIZE = 3          
DBSCAN_EPS = 40
DBSCAN_MIN_POINTS = 30

#load depth
depth = np.load(OBSERVED_PATH)
print("Shape:", depth.shape)

#transform the depth into the point cloud
points = []

height, width = depth.shape

for y in range(height):
    for x in range(width):
        z = depth[y, x]

        if z == 0:
            continue

        if z < MIN_DEPTH or z > MAX_DEPTH:
            continue

        X = (x - CX_DEPTH) * z / FX_DEPTH
        Y = (y - CY_DEPTH) * z / FY_DEPTH
        Z = z

        points.append([X, Y, Z])

points = np.array(points)

print("Initial points:", len(points))

observed = o3d.geometry.PointCloud()
observed.points = o3d.utility.Vector3dVector(points)

#downsample
observed = observed.voxel_down_sample(voxel_size=VOXEL_SIZE)

print("After downsample:", len(observed.points))

#remove outlier
observed, ind = observed.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)

observed = observed.select_by_index(ind)

print("After outlier removal:", len(observed.points))

#DBSCAN clustering
labels = np.array(
    observed.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=True
    )
)

max_label = labels.max()
print("Clusters found:", max_label + 1)

if max_label < 0:
    raise ValueError("NO bigger cluster found.")

#we keep the biggest cluster
largest_cluster = None
largest_size = 0

for i in range(max_label + 1):
    size = np.sum(labels == i)

    print(f"Cluster {i}: {size} points")

    if size > largest_size:
        largest_size = size
        largest_cluster = i

indices = np.where(labels == largest_cluster)[0]
observed = observed.select_by_index(indices)

print("Selected cluster:", largest_cluster)
print("Final points:", len(observed.points))

#final cleaning
observed, ind = observed.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=1.5
)

observed = observed.select_by_index(ind)

#color, save and visualize
observed.paint_uniform_color([0, 0, 1])
o3d.io.write_point_cloud(SAVE_PATH, observed)
print("Saved:", SAVE_PATH)
print("Cleaning completed.")
