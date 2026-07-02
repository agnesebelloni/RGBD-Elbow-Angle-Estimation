import open3d as o3d
import numpy as np

files = [
    "/Users/computer/Desktop/final project HSI/segments/clean/arm1_clean.ply",
    "/Users/computer/Desktop/final project HSI/segments/clean/arm2_clean.ply",    
    "/Users/computer/Desktop/final project HSI/segments/clean/arm3_clean.ply",
    "/Users/computer/Desktop/final project HSI/segments/clean/arm4_clean.ply",
    "/Users/computer/Desktop/final project HSI/segments/clean/arm5_clean.ply"
]

#first cloude as base
pcd_base = o3d.io.read_point_cloud(files[0])
pcd_base_down = pcd_base.voxel_down_sample(voxel_size=0.005)
pcd_base_down.estimate_normals()

#doing the registration of all the point cloud on the first one
for path in files[1:]:
    print(f"Registrazione ICP con: {path}")
    pcd_next = o3d.io.read_point_cloud(path)
    pcd_next_down = pcd_next.voxel_down_sample(voxel_size=0.005)
    pcd_next_down.estimate_normals()

    threshold = 0.05
    reg = o3d.pipelines.registration.registration_icp(
        pcd_next_down, pcd_base_down, threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print("Trasformazione trovata:\n", reg.transformation)

    #we apply the trasformation of the original cloud
    pcd_next.transform(reg.transformation)

    #we add the new cloud to the base
    pcd_base += pcd_next

    #updated downasmple for the newt interaction
    pcd_base_down = pcd_base.voxel_down_sample(voxel_size=0.005)
    pcd_base_down.estimate_normals()

#save final fusion
output_path = "/Users/computer/Desktop/final project HSI/ICP/arm_forearm/arm_fused.ply"  # cambia in forearm_fused.ply se serve
o3d.io.write_point_cloud(output_path, pcd_base)
print(f"Fusione salvata in: {output_path}")

#visualize
o3d.visualization.draw_geometries([pcd_base], window_name="Fusione ICP completata")
