import os
import argparse
import numpy as np
import copy
import mne
import open3d as o3d
from sys import exit

# Set up directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Utility functions
def check_file_exists(filepath):
    """Check if a file exists and print an error message if not."""
    if not os.path.isfile(filepath):
        print(f"Error: File not found - {filepath}")
        exit(1)

def pick_points_from_cloud(point_cloud, window_size=(1000, 1000)):
    """Open an interactive Open3D visualization window for selecting points."""
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=window_size[0], height=window_size[1])
    vis.add_geometry(point_cloud)
    vis.run()  
    vis.destroy_window()
    return vis.get_picked_points()

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample a point cloud and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Perform global feature-based registration."""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def perform_icp_registration(source_cloud, target_cloud, initial_transformation=None, voxel_size=2, max_iterations=8000):
    """Perform ICP registration between two point clouds."""
    source_down, source_fpfh = preprocess_point_cloud(source_cloud, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)

    result_global = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    trans_init = result_global.transformation if initial_transformation is None else initial_transformation

    threshold = 2.00
    result_icp = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
    
    return result_icp.transformation

def save_transformation(datafile, transformation_matrix, transform_type, suffix="_trans.fif"):
    """Save a transformation matrix to a file in the data directory."""
    trans = mne.transforms.Transform(transform_type.split("->")[0], transform_type.split("->")[1], trans=None)
    trans['trans'] = transformation_matrix.copy()
    trans['trans'][0:3, 3] /= 1000  # Convert mm to meters

    output_filename = os.path.join(DATA_DIR, os.path.basename(datafile).replace(".fif", suffix))
    mne.write_trans(output_filename, trans, overwrite=True)
    
    return output_filename

# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("-om", "--outside_mesh", help="LIDAR scan of head outside the MEG Helmet", required=True)
parser.add_argument("-im", "--inside_mesh", help="LIDAR scan of head inside the MEG Helmet", required=True)
parser.add_argument("-s", "--mri_scalp", help="MRI scalp surface from Freesurfer", required=True)
parser.add_argument("-m", "--megdata", help="MEG data to generate the transform into", nargs='+', required=True)
parser.add_argument("-lm", "--landmarks", help="Landmarks to use for helmet registration", type=int, nargs='+')

args = parser.parse_args()

# Convert input paths to use `DATA_DIR`
helmet_mesh = os.path.join(DATA_DIR, args.inside_mesh)
head_mesh = os.path.join(DATA_DIR, args.outside_mesh)
mri_cloud = os.path.join(DATA_DIR, args.mri_scalp)
meg_data_files = [os.path.join(DATA_DIR, f) for f in args.megdata]

# Check all input files
for f in [helmet_mesh, head_mesh, mri_cloud] + meg_data_files:
    check_file_exists(f)

# Load and process transformations
X1 = perform_icp_registration(o3d.io.read_triangle_mesh(helmet_mesh).sample_points_poisson_disk(100000), 
                              o3d.io.read_triangle_mesh(head_mesh).sample_points_poisson_disk(100000))

[standard_trans, standard_head] = perform_icp_registration(o3d.io.read_triangle_mesh(head_mesh).sample_points_poisson_disk(100000),
                                                           o3d.io.read_triangle_mesh(mri_cloud).sample_points_poisson_disk(100000))

X2 = perform_icp_registration(standard_head, o3d.io.read_triangle_mesh(helmet_mesh).sample_points_poisson_disk(100000))
X21 = np.dot(X2, X1)

# Process each MEG data file
for MEG_data in meg_data_files:
    output_file_meg = save_transformation(MEG_data, X21, "meg->head")
    output_file_mri = save_transformation(MEG_data, standard_trans, "mri->head")

    print("\nTransforms written to:")
    print(f"  MEG data: {MEG_data}")
    print(f"  MEG->Head transformation: {output_file_meg}")
    print(f"  MRI->Head transformation: {output_file_mri}")

if __name__ == '__main__':
    print("\nProcessing complete! 🚀")