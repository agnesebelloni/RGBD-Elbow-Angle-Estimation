import cv2
import numpy as np
import open3d as o3d

DEPTH_PATH = '/Users/computer/Desktop/final project HSI/images/Azure2_depth_wide.png'

#Azure Kinect intrinsics
fx = 505.38
fy = 505.55
cx = 334.50
cy = 333.27

#load depht image
depth_image = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)
if depth_image is None:
    raise ValueError("Could not load depth image!")

#preparation for drawing
depth_clipped = np.clip(depth_image, 300, 1500).astype(np.uint16)
depth_vis = ((depth_clipped - 300) / (1500 - 300) * 255).astype(np.uint8)
depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)  #for coloring
depth_vis_colored = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)  #for drawing

drawing = False
current_poly = []
all_polygons = []

def draw_polygon(event, x, y, flags, param):
    global drawing, current_poly, all_polygons
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_poly = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_poly.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(current_poly) > 2:
            all_polygons.append(current_poly.copy())
            print(f"Polygon {len(all_polygons)} completed.")
            current_poly.clear()

cv2.namedWindow('Depth Image')
cv2.setMouseCallback('Depth Image', draw_polygon)

print("Draw TWO regions (e.g., arm and forearm). Press 'q' after finishing the second.")

while True:
    temp = depth_vis_colored.copy()
    for poly in all_polygons:
        cv2.polylines(temp, [np.array(poly)], isClosed=True, color=(0, 255, 0), thickness=2)
    if current_poly:
        cv2.polylines(temp, [np.array(current_poly)], isClosed=False, color=(255, 0, 0), thickness=1)
    cv2.imshow('Depth Image', temp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or len(all_polygons) == 2:
        break

cv2.destroyAllWindows()

if len(all_polygons) != 2:
    print("You must select exactly TWO regions.")
    exit(0)

#process each polygon region
for i, poly in enumerate(all_polygons):
    part_name = "arm" if i == 0 else "forearm"

    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(poly)], 255)

    coords = np.column_stack(np.where(mask > 0))  
    print(f"Points in mask '{part_name}': {len(coords)}")

    points_3d = []
    colors_3d = []

    for (y, x) in coords:
        z_mm = float(depth_image[y, x])
        if z_mm == 0 or z_mm < 300 or z_mm > 2000:
            continue
        z = z_mm / 1000.0
        X = (x - cx) * z / fx
        Y = (y - cy) * z / fy
        points_3d.append([X, Y, z])

        bgr = depth_colormap[y, x]
        rgb = bgr[::-1] / 255.0
        colors_3d.append(rgb)

    if len(points_3d) == 0:
        print(f"No valid 3D points found for '{part_name}'. Skipping...")
        continue

    #save point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_3d)

    output_path = f"/Users/computer/Desktop/{part_name}1_depth_segment_colored.ply"
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved {part_name} to: {output_path}")

    #visualization
    o3d.visualization.draw_geometries([pcd])
