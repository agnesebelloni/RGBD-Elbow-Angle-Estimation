import open3d as o3d
import numpy as np
import copy
import os

#load fused arm and forearm with ICP
ARM_PATH = "/Users/computer/Desktop/final project HSI/ICP/arm_forearm/arm_fused.ply"
FOREARM_PATH = "/Users/computer/Desktop/final project HSI/ICP/arm_forearm/forearm_fused.ply"

OUT_DIR = "/Users/computer/Desktop/elbow_results"
os.makedirs(OUT_DIR, exist_ok=True)

#parameters
VOXEL_SIZE = 0.005 #to reduce n of points maintaining the point cloud
ANGLES = [0, 30, 60, 90] #angles we woint to simulate

FOREARM_LENGTH = 0.35 #approx forearm lenght
FOREARM_RADIUS = 0.18 #approx forearm radius
ELBOW_MARGIN = 0.03 #added margin

#preprocessing point cloud
def load_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Cloud empty: {path}")

    pcd = pcd.voxel_down_sample(VOXEL_SIZE)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd = pcd.select_by_index(ind)
    return pcd #we mantain only the valid points


#elbow center estimation
def get_true_elbow(arm, forearm):
    arm_pts = np.asarray(arm.points)
    forearm_pts = np.asarray(forearm.points)

    best_i, best_j = 0, 0
    best_d = np.inf

    for i in range(len(arm_pts)):
        diff = forearm_pts - arm_pts[i]
        dists = np.sum(diff * diff, axis=1)
        j = np.argmin(dists)

        if dists[j] < best_d:
            best_d = dists[j]
            best_i = i
            best_j = j

    arm_joint = arm_pts[best_i]
    forearm_joint = forearm_pts[best_j]
    elbow_center = (arm_joint + forearm_joint) / 2

    print("Elbow center:", elbow_center)
    print("Joint distance:", np.sqrt(best_d))

    return elbow_center #estimates the elbow joint center by finding the closest pair of points
#between the arm model and the forearm model


#computes the principal axis of a point cloud using Principal Component Analysis (PCA)
def main_axis(points):
    center = points.mean(axis=0)
    centered = points - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    axis = axis / np.linalg.norm(axis)
    return center, axis

#computes the rotation matrix that aligns one vector with another
def rotation_between_vectors(v_from, v_to):
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    cross = np.cross(v_from, v_to)
    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)

    if np.linalg.norm(cross) < 1e-8:
        return np.eye(3)

    axis = cross / np.linalg.norm(cross)
    angle = np.arccos(dot)

    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

#allineation between of arm and forearm
def pre_align_forearm(arm, forearm, elbow_center):
    arm_pts = np.asarray(arm.points)
    forearm_pts = np.asarray(forearm.points)

    _, arm_axis = main_axis(arm_pts)
    forearm_center, forearm_axis = main_axis(forearm_pts)

    R = rotation_between_vectors(forearm_axis, arm_axis)

    aligned = copy.deepcopy(forearm)
    aligned.translate(-elbow_center)
    aligned.rotate(R, center=(0, 0, 0))
    aligned.translate(elbow_center)

    return aligned

#we keep only a part ofthe forearm that starts from the elbow and delete teh points that are too far
def clean_forearm_region(arm, forearm, elbow_center):
    
    pts = np.asarray(forearm.points)

    _, axis = main_axis(pts)

    # scegli direzione che va lontano dall'arm
    arm_center = np.asarray(arm.points).mean(axis=0)
    forearm_center = pts.mean(axis=0)

    if np.dot(forearm_center - elbow_center, axis) < 0:
        axis = -axis

    rel = pts - elbow_center

    longitudinal = rel @ axis

    projected = np.outer(longitudinal, axis)
    lateral = rel - projected
    lateral_dist = np.linalg.norm(lateral, axis=1)

    mask = (
        (longitudinal > -ELBOW_MARGIN) &
        (longitudinal < FOREARM_LENGTH) &
        (lateral_dist < FOREARM_RADIUS)
    )

    idx = np.where(mask)[0]

    cleaned = forearm.select_by_index(idx)

    if len(cleaned.points) > 20:
        cleaned, ind = cleaned.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=1.5
        )
        cleaned = cleaned.select_by_index(ind)

    cleaned = cleaned.voxel_down_sample(VOXEL_SIZE)

    return cleaned

#rotates the forearm point cloud around the elbow center by a given angle expressed in degrees
def rotate_forearm_at_elbow(forearm, angle_deg, elbow_center):
    rotated = copy.deepcopy(forearm)

    theta = np.deg2rad(angle_deg)

    #foration in the plane XY around Z
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, theta))

    rotated.translate(-elbow_center)
    rotated.rotate(R, center=(0, 0, 0))
    rotated.translate(elbow_center)

    return rotated

#creates a small green sphere used to visualize the elbow center
def make_sphere(center):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    sphere.translate(center)
    sphere.paint_uniform_color([0, 1, 0])
    return sphere

#saves a screenshot of the given Open3D geometries
def save_view(geometries, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)

    for g in geometries:
        vis.add_geometry(g)

    vis.get_render_option().background_color = np.asarray([1, 1, 1])
    vis.poll_events()
    vis.update_renderer()

    path = os.path.join(OUT_DIR, filename)
    vis.capture_screen_image(path)
    vis.destroy_window()

    print("Saved image:", path)


àloading
arm = load_pcd(ARM_PATH)
forearm = load_pcd(FOREARM_PATH)

arm.paint_uniform_color([1, 0, 0])
forearm.paint_uniform_color([0, 0, 1])

print("Arm points:", len(arm.points))
print("Forearm points before clean:", len(forearm.points))

#elbow
elbow_center = get_true_elbow(arm, forearm)

#pre-align
forearm = pre_align_forearm(arm, forearm, elbow_center)

#clean forearm region
forearm = clean_forearm_region(arm, forearm, elbow_center)
forearm.paint_uniform_color([0, 0, 1])

print("Forearm points after clean:", len(forearm.points))

sphere = make_sphere(elbow_center)

#visualize and save elbow
o3d.visualization.draw_geometries(
    [arm, forearm, sphere],
    window_name="Elbow center"
)

save_view([arm, forearm, sphere], "elbow_center.png")

#rotations
#generate articulated models for each elbow angle
for angle in ANGLES:
    print("Angle:", angle)

    #rotate the forearm around the elbow center according to the current flexion angle
    forearm_rot = rotate_forearm_at_elbow(
        forearm,
        angle_deg=angle,
        elbow_center=elbow_center
    )

    forearm_rot.paint_uniform_color([0, 0, 1])

    combined = arm + forearm_rot

    img_path = f"elbow_{angle}.png"
    ply_path = os.path.join(OUT_DIR, f"combined_elbow_{angle}.ply")

    o3d.io.write_point_cloud(ply_path, combined)
    print("Saved cloud:", ply_path)

    o3d.visualization.draw_geometries(
        [combined],
        window_name=f"Elbow angle {angle}"
    )

    save_view([combined], img_path)

