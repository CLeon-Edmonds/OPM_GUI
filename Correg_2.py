#!/usr/bin/env python3
"""
Point Cloud Registration Script using Open3D and MNE.

This script:
-Loads and preprocesses point clouds.
-Performs global and ICP registration.
-Saves transformation matrices.
-Opens an interactive window for landmark selection (freeview style).
"""

import os
import numpy as np
import copy
import mne
import open3d as o3d


# Step 1: Define Paths and Parameters
# --------------------------
BASE_DIR = os.getcwd()  # Current working directory
DATA_DIR = os.path.join(BASE_DIR, "/Volumes/kowalcau-opm-recordings/ChrisMilan/")
os.makedirs(DATA_DIR, exist_ok=True)  # Create data directory if it doesn't exist

# File paths (change these paths as needed)
head_mesh = '/Volumes/CHRISDRIVE1/Birmingham/Code/Milan_Outside.ply'
helmet_mesh = '/Volumes/CHRISDRIVE1/Birmingham/Code/Milan_Inside.ply'
mri_cloud = '/Volumes/CHRISDRIVE1/Birmingham/Code/Milan_Test_3_mesh.ply'
meg_data_files = [os.path.join(DATA_DIR, "20241211_163259_sub-Pilot2_file-Auditory1_raw.fif")]
landmarks = [1, 2, 3, 4, 5, 6, 7] #Do we actually need 7 or can we do 6? 

# Step 2: Define Utility Functions
# --------------------------
def check_file_exists(filepath, file_description=""):
    """Check if a file exists; raise an error if not found."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f" ERROR: {file_description} file not found at {filepath}")

# Validate input files
check_file_exists(head_mesh, "LIDAR Scan Outside Mesh")
check_file_exists(helmet_mesh, "LIDAR Scan Inside Mesh")
check_file_exists(mri_cloud, "MRI Scalp Surface")
for i, meg_file in enumerate(meg_data_files):
    check_file_exists(meg_file, f"MEG Data File {i+1}")

def preprocess_point_cloud(pcd, voxel_size=2, normal_nn=30, feature_nn=100):
    """
    Downsample the point cloud and compute FPFH features.
    
    Returns:
        tuple: Downsampled point cloud and its FPFH features.
    """
    # Downsample the point cloud using a voxel grid.
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # Estimate normals for the downsampled cloud 
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=normal_nn)
    )
    
    # Compute the Fast Point Feature Histograms for the downsampled cloud.
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=feature_nn)
    )
    
    # Return both the downsampled point cloud and its FPFH features.
    return pcd_down, pcd_fpfh

def perform_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size=2, ransac_iter=100000):
    """
    Perform global feature-based registration using RANSAC.
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_iter, 0.999)
    )
    return result

def perform_icp_registration(source_cloud, target_cloud, initial_transformation=None, max_iterations=8000, threshold=2.0):
    """
    Refine registration using ICP.
    """
    if initial_transformation is None:
        initial_transformation = np.identity(4)
    result_icp = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    return result_icp.transformation

def save_transformation(datafile, transformation_matrix, transform_type, suffix="_trans.fif"):
    """
    Save a transformation matrix to a file using MNE.
    """
    src, dest = transform_type.split("->")
    trans = mne.transforms.Transform(src, dest, trans=None)
    trans['trans'] = transformation_matrix.copy()
    trans['trans'][0:3, 3] /= 1000 
    output_filename = os.path.join(DATA_DIR, os.path.basename(datafile).replace(".fif", suffix))
    mne.write_trans(output_filename, trans, overwrite=True)
    return output_filename

def print_final_outputs(meg_file, output_file_meg, output_file_mri):
    """Print a summary of the generated transformation files."""
    print("\n✅ Registration Completed Successfully!")
    print("🔹 Input MEG Data File:")
    print(f"   📄 {meg_file}")
    print("\n🔹 Generated Transformation Files:")
    print(f"   📂 MEG-to-Head Transform: {output_file_meg}")
    print(f"   📂 MRI-to-Head Transform: {output_file_mri}")
    print("\n📌 All files are stored in the 'data' directory.\n")

# --- Interactive Landmark Selection ---
def pick_points_standard(pcd, window_name, num_points, instructions):
    """
    Opens Open3D's visualiser for manual point selection.
    
    Parameters:
      pcd: Point cloud to select points from.
      window_name: Title for the window.
      num_points: Expected number of points.
      instructions: Prompt message.
      
    Returns:
      List of picked point indices.
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=window_name, width=800, height=800)
    vis.add_geometry(pcd)
    print(instructions)
    vis.run()  # User selects points then presses 'Q' to exit.
    vis.destroy_window()
    picked_ids = vis.get_picked_points()
    if len(picked_ids) != num_points:
        print(f" Warning: You picked {len(picked_ids)} points, expected {num_points}.")
    return picked_ids

def get_user_landmarks(mesh_path):
    """
    Loads a triangle mesh, samples it to a point cloud and then allows manual landmark selection.
    
    Returns:
      Tuple of (initial_points, additional_points) as coordinates.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_poisson_disk(100000)
    
    initial_ids = pick_points_standard(
        pcd, 
        window_name="Select 7 Points", 
        num_points=7, 
        instructions="🔹 Instructions: Shift + Left Click to select 7 initial points, then press 'Q' to exit."
    )
    print("You picked the following 7 initial point indices:")
    print(initial_ids)
    
    additional_ids = pick_points_standard(
        pcd, 
        window_name="Select 3 Points", 
        num_points=3, 
        instructions="🔹 Instructions: Shift + Left Click to select the naison and ear points (3 points), then press 'Q' to exit."
    )
    print("You picked the following 3 additional point indices:")
    print(additional_ids)
    
    points = np.asarray(pcd.points)
    initial_points = points[initial_ids, :]
    additional_points = points[additional_ids, :]
    return initial_points, additional_points

# Step 3: Main Execution
# --------------------------
def main():
    # Load point clouds from files
    print("Loading point clouds...")
    pcd_head = o3d.io.read_point_cloud(head_mesh)
    pcd_helmet = o3d.io.read_point_cloud(helmet_mesh)
    pcd_mri = o3d.io.read_point_cloud(mri_cloud)
    
    # Preprocess point clouds for registration
    print("Preprocessing point clouds...")
    pcd_head_down, pcd_head_fpfh = preprocess_point_cloud(pcd_head)
    pcd_helmet_down, pcd_helmet_fpfh = preprocess_point_cloud(pcd_helmet)
    
    # Global registration 
    print("Performing global registration...")
    global_reg = perform_global_registration(pcd_helmet_down, pcd_head_down, pcd_helmet_fpfh, pcd_head_fpfh)
    print("Global registration transformation:")
    print(global_reg.transformation)
    
    # ICP registration 
    print("Performing ICP registration...")
    icp_transformation = perform_icp_registration(pcd_helmet, pcd_head, global_reg.transformation)
    print("ICP refined transformation:")
    print(icp_transformation)
    
    # Save transformation matrices for MEG and MRI data
    meg_file = meg_data_files[0]
    output_file_meg = save_transformation(meg_file, icp_transformation, "meg->head")
    output_file_mri = save_transformation(meg_file, icp_transformation, "mri->head")
    
    # Print summary of outputs
    print_final_outputs(meg_file, output_file_meg, output_file_mri)
    
    # Launch interactive landmark selection (freeview thing)
    print("Launching interactive landmark selection...")
    initial_landmarks, additional_landmarks = get_user_landmarks(head_mesh)
    print("Landmark selection completed.")
    print("Initial Landmarks (7 points):")
    print(initial_landmarks)
    print("Additional Landmarks (nasion and ear points):")
    print(additional_landmarks)

if __name__ == '__main__':
    main()