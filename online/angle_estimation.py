import open3d as o3d
import numpy as np
import copy
import os

#observed files
observed_files = {
    "observed1": "/Users/computer/Desktop/observed1_cleaned.ply",
    "observed2": "/Users/computer/Desktop/observed2_cleaned.ply",
    "observed3": "/Users/computer/Desktop/observed3_cleaned.ply"
}


#models
model_paths = {
    0: "/Users/computer/Desktop/elbow_results/combined_elbow_0.ply",
    30: "/Users/computer/Desktop/elbow_results/combined_elbow_30.ply",
    60: "/Users/computer/Desktop/elbow_results/combined_elbow_60.ply",
    90: "/Users/computer/Desktop/elbow_results/combined_elbow_90.ply"
}


save_root = "/Users/computer/Desktop/angle_estimation_results"
os.makedirs(save_root, exist_ok=True)

#icp function
def evaluate_model(observed, model_path, angle):

    model = o3d.io.read_point_cloud(model_path)

    observed_temp = copy.deepcopy(observed)
    model_temp = copy.deepcopy(model)

    observed_temp = observed_temp.voxel_down_sample(5)
    model_temp = model_temp.voxel_down_sample(5)

    observed_temp.paint_uniform_color([1, 0, 0])
    model_temp.paint_uniform_color([0, 0, 1])

    #center clouds
    observed_temp.translate(-observed_temp.get_center())
    model_temp.translate(-model_temp.get_center())

    #icp
    threshold = 40

    icp_result = o3d.pipelines.registration.registration_icp(
        observed_temp,
        model_temp,
        threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    observed_temp.transform(icp_result.transformation)

    return icp_result.inlier_rmse, observed_temp, model_temp


#loop over observed files
for observed_name, observed_path in observed_files.items():

    print("\n===================================")
    print("PROCESSING:", observed_name)
    print("===================================")

    observed = o3d.io.read_point_cloud(observed_path)

    results = {}

    best_rmse = 999999
    best_angle = None
    best_observed = None
    best_model = None

   #test all angles

    for angle, model_path in model_paths.items():

        print("\nTesting angle:", angle)

        rmse, observed_aligned, model_aligned = evaluate_model(
            observed,
            model_path,
            angle
        )

        print("RMSE:", rmse)

        results[angle] = rmse

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)

        vis.add_geometry(observed_aligned)
        vis.add_geometry(model_aligned)

        vis.get_render_option().background_color = np.asarray([1, 1, 1])

        vis.poll_events()
        vis.update_renderer()

        save_img = os.path.join(
            save_root,
            f"{observed_name}_angle_{angle}.png"
        )

        vis.capture_screen_image(save_img)
        vis.destroy_window()

        print("Saved image:", save_img)

        #cloud
        save_cloud = os.path.join(
            save_root,
            f"{observed_name}_angle_{angle}.ply"
        )

        combined = observed_aligned + model_aligned

        o3d.io.write_point_cloud(save_cloud, combined)

        print("Saved cloud:", save_cloud)

        #best results
        if rmse < best_rmse:

            best_rmse = rmse
            best_angle = angle

            best_observed = observed_aligned
            best_model = model_aligned

    #final results
    print("\nRESULTS:", results)
    print("BEST ANGLE:", best_angle)
    print("BEST RMSE:", best_rmse)

    #best match
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)

    vis.add_geometry(best_observed)
    vis.add_geometry(best_model)

    vis.get_render_option().background_color = np.asarray([1, 1, 1])

    vis.poll_events()
    vis.update_renderer()

    best_img = os.path.join(
        save_root,
        f"{observed_name}_BEST_{best_angle}.png"
    )

    vis.capture_screen_image(best_img)
    vis.destroy_window()

    print("Saved BEST image:", best_img)

print("\n===================================")
print("ALL PROCESSING COMPLETED")
print("Results saved in:")
print(save_root)
print("===================================")
