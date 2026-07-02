import open3d as o3d
import numpy as np
import copy
import os

desktop = os.path.expanduser("~/Desktop")

#input: observed point cloud
observed_files = {
    "observed1": desktop + "/observed1_cleaned.ply",
    "observed2": desktop + "/observed2_cleaned.ply",
    "observed3": desktop + "/observed3_cleaned.ply"
}

#offline articulated model
#offline models correspond to albow angle
model_paths = {
    0: desktop + "/elbow_results/combined_elbow_0.ply",
    30: desktop + "/elbow_results/combined_elbow_30.ply",
    60: desktop + "/elbow_results/combined_elbow_60.ply",
    90: desktop + "/elbow_results/combined_elbow_90.ply"
}

save_root = desktop + "/final_online_results"
os.makedirs(save_root, exist_ok=True)

VOXEL_SIZE = 0.005      #downsampling (meters)
ICP_THRESHOLD = 0.05    #maximum correspondence distance used by ICP (meters)

#loads an observed point cloud, converts it from millimeters
#to meters, downsamples it, and removes remaining outliers
def load_observed(path):
    pcd = o3d.io.read_point_cloud(path)

    #observed mm -> meters
    pcd.scale(0.001, center=(0, 0, 0))

    pcd = pcd.voxel_down_sample(VOXEL_SIZE)

    if len(pcd.points) > 30:
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )
        pcd = pcd.select_by_index(ind)

    return pcd

#loads an articulated model point cloud and applies voxel downsampling
def load_model(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    return pcd


#online angle estimation
summary_results = {}

#loop for all observed point clouds
for observed_name, observed_path in observed_files.items():

    print("\n===================================")
    print("PROCESSING:", observed_name)
    print("===================================")

    observed = load_observed(observed_path) #load and preprocess

    print("Observed points:", len(observed.points))

    if len(observed.points) == 0:
        print("Observed empty, skipping.")
        continue #skip the current obs if point cloud is empty

    observed.paint_uniform_color([1, 0, 0]) #observed clud red
    #variables to store best results
    best_rmse = np.inf
    best_angle = None
    results = {}

    #test all candidate articulated models
    for angle, model_path in model_paths.items():

        print("\nTesting angle:", angle)

        model = load_model(model_path)

        print("Model points:", len(model.points))

        if len(model.points) == 0:
            print("Model empty, skipping angle.")
            continue

        model.paint_uniform_color([0, 0, 1])

        observed_temp = copy.deepcopy(observed)
        model_temp = copy.deepcopy(model)

        #initial center alignment
        model_temp.translate(
            observed_temp.get_center() - model_temp.get_center()
        )

        reg = o3d.pipelines.registration.registration_icp(
            model_temp,
            observed_temp,
            ICP_THRESHOLD,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        rmse = reg.inlier_rmse
        fitness = reg.fitness

        print("RMSE:", rmse)
        print("Fitness:", fitness)

        results[angle] = rmse

        model_temp.transform(reg.transformation)

        combined = model_temp + observed_temp

        cloud_path = os.path.join(
            save_root,
            f"{observed_name}_angle_{angle}.ply"
        )

        o3d.io.write_point_cloud(cloud_path, combined)
        print("Saved cloud:", cloud_path)

        if rmse < best_rmse:
            best_rmse = rmse
            best_angle = angle
            best_combined = combined

    #results for the current observed point cloud
    summary_results[observed_name] = {
        "best_angle": best_angle,
        "best_rmse": best_rmse,
        "all_rmse": results
    }

    #best match configuration
    if best_angle is not None:
        best_path = os.path.join(
            save_root,
            f"{observed_name}_BEST_{best_angle}.ply"
        )
        o3d.io.write_point_cloud(best_path, best_combined)
        print("Saved BEST:", best_path)

    print("\nRESULTS:", results)
    print("BEST ANGLE:", best_angle)
    print("BEST RMSE:", best_rmse)


print("\n===================================")
print("FINAL RESULTS")
print("===================================")

for name, result in summary_results.items():
    print(
        f"{name} -> BEST ANGLE = {result['best_angle']} | "
        f"RMSE = {result['best_rmse']} | "
        f"ALL RMSE = {result['all_rmse']}"
    )

print("\nSaved in:", save_root)
