import numpy as np
import open3d as o3d


def pick_points(pcd):
    """ Open3D interactive tool to pick points from a point cloud. """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=1000, height=1000)
    vis.add_geometry(pcd)
    vis.run()  # User picks points
    vis.destroy_window()
    return vis.get_picked_points()


def visualize_landmark_errors(selected_points, reference_points):
    """ Visualize selected landmarks vs. reference points with error-based color coding. """
    selected_cloud = o3d.geometry.PointCloud()
    selected_cloud.points = o3d.utility.Vector3dVector(selected_points)
    
    reference_cloud = o3d.geometry.PointCloud()
    reference_cloud.points = o3d.utility.Vector3dVector(reference_points)
    reference_cloud.paint_uniform_color([0, 0, 1])  # Blue for reference points
    
    # Compute error distances
    errors = np.linalg.norm(np.array(selected_points) - np.array(reference_points), axis=1)
    colors = [[0, 1, 0] if err < 5 else [1, 0, 0] for err in errors]  # Green if <5mm, else Red
    selected_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    print("\nLandmark Co-Registration Errors (mm):")
    for i, err in enumerate(errors):
        print(f"  Point {i+1}: {err:.3f} mm")
    
    # Show only the selected and reference points
    o3d.visualization.draw_geometries([selected_cloud, reference_cloud], width=1000, height=1000)

def main():
    # Generate a synthetic point cloud for testing
    num_points = 5000
    synthetic_cloud = np.random.uniform(-50, 50, (num_points, 3))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(synthetic_cloud)

    print("\nPick at least 3 points from the visualization.")
    picked_indices = pick_points(pcd)
    selected_points = [synthetic_cloud[i] for i in picked_indices]

    # Generate reference points slightly different from selected ones
    reference_points = np.array(selected_points) + np.random.uniform(-2, 2, (len(selected_points), 3))

    # Visualize and compare
    visualize_landmark_errors(selected_points, reference_points)


if __name__ == "__main__":
    main()