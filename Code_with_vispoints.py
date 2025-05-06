import numpy as np
import copy
from sys import exit
import argparse
import mne
import os

# Approximate distance from Rubidium chamber to external tip of
# sensor housing that makes contact with head

sensor_length = 6e-3


def usage():
    print(f"""Usage: {os.path.basename(__file__)} --outsidemesh --insidemesh --mriscalp --megdata

    -om, --outsidemesh, LIDAR scan of head outside the MEG Helmet)
    -im, --insidemesh,  LIDAR scan of head inside the MEG Helmet)
    -s,  --mriscalp,    MRI scalp surface from Freesurfer)
    -m,  --megdata,     MEG data file(s) to generate the transform into
    (multiple files space-seperated))
    """)

    exit(0)


def pick_points(pcd):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(width=1000, height=1000)
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()


def preprocess_point_cloud(pcd, voxel_size):
    import open3d as o3d
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    import open3d as o3d
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def vis_controls():
    print("\nControls:")
    print("  Shift + Left-Click:  Add a point.")
    print("  Shift + Right-Click: Remove a point.")
    print("  Hit q when all points are made.\n")


def head_to_helmet(source_in, landmarks):
    print("\nSetting up...")
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # Align manually selected points from a LIDAR scan with
    # known landmark coordinates

    # Open stl files and convert to 100k point clouds so that we can use point-wise registration
    print("Converting STL to PointCloud...")
    mesh = o3d.io.read_triangle_mesh(source_in)
    source_cloud = mesh.sample_points_poisson_disk(100000)

    # Known positions of stickers in helmet reference frame
    rsticker_pillars = np.zeros([7, 3])
    rsticker_pillars[0] = [102.325, 0.221, 16.345]
    rsticker_pillars[1] = [92.079, 66.226, -27.207]
    rsticker_pillars[2] = [67.431, 113.778, -7.799]
    rsticker_pillars[3] = [-0.117, 138.956, -5.576]
    rsticker_pillars[4] = [-67.431, 113.778, -7.799]
    rsticker_pillars[5] = [-92.079, 66.226, -27.207]
    rsticker_pillars[6] = [-102.325, 0.221, 16.345]

    landmarks = np.asarray(landmarks)-1

    # Make into cloud
    rst_cloud = o3d.geometry.PointCloud()
    rst_cloud.points = o3d.utility.Vector3dVector(rsticker_pillars[landmarks[:]])

    # Make the true pillar landmarks blue so we can see them relative to red cloud
    rst_cloud.paint_uniform_color([0, 0, 1])

    # Get anchor points
    vis_controls()
    print("Select the Helmet Labels from left to right.")

    # Display visualiser
    red_points = pick_points(source_cloud)

    # Define which sticker-pillar points correspond to which selected points
    corr = np.zeros((len(rsticker_pillars), 2))
    corr = np.zeros((len(landmarks), 2))
    for ii, lm in enumerate(landmarks):
        corr[ii, 0] = lm  # So users don't have to deal with zero-indexing
    corr[:, 1] = red_points

    # Calculate transform based on anchor points alone
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(rst_cloud, source_cloud,
                                            o3d.utility.Vector2iVector(corr))
    X1 = trans_init

    # Have a look at anchor-based registration
    test = copy.deepcopy(rst_cloud)
    test.transform(trans_init)

    # Print errors on anchor-point registration
    print("\nLandmark co-registration Errors:")
    for ii, coords in enumerate(test.points):
        print("%.3f mm " % np.linalg.norm(coords - source_cloud.points[int(corr[ii][1])]))
    return X1


def head_to_standard(target_in):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    # Transform head-outside into CTF-like coordinate system
    # We don't really need to have this intermediate coordinate space.
    # In principle, we could just use MEG device and MRI spaces but
    # this might prove useful during analysis as it ape the way that
    # legacy systems dealt with transforms.
    mesh = o3d.io.read_triangle_mesh(target_in)
    print("\nSetting up next step...")
    target_cloud = mesh.sample_points_poisson_disk(100000)

    # Get anchor points
    print("\nIn order, please select: ")
    print("  - right pre-auricular")
    print("  - left pre-auricular")
    print("  - nasion\n")

    anat_points = pick_points(target_cloud)
    R_aur = target_cloud.points[anat_points[0]]
    nas = target_cloud.points[anat_points[1]]
    L_aur = target_cloud.points[anat_points[2]]

    # Get position of CTF-style origin in original LIDAR data
    origin = R_aur + (R_aur - L_aur) / 2.

    # Define anatomical points in 'standard' space to align with
    standard = np.zeros([3, 3])

    # right pre-auricular on -ve y-axis
    standard[0] = [0, -np.linalg.norm(R_aur - L_aur)/2., 0]

    # left pre-auricular on +ve y-axis
    standard[1] = [0, np.linalg.norm(R_aur - L_aur)/2., 0]

    # Nasion on x-axis
    standard[2] = [np.linalg.norm(origin-nas), 0, 0]

    # Make into cloud
    standard_cloud = o3d.geometry.PointCloud()
    standard_cloud.points = o3d.utility.Vector3dVector(standard)

    # Define which LIDAR points correspond to which standard points
    corr = np.zeros((3, 2))
    corr[:, 0] = anat_points
    corr[:, 1] = [0, 1, 2]

    # Calculate transform and apply
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(target_cloud, standard_cloud,
                                            o3d.utility.Vector2iVector(corr))
    target_cloud.transform(trans_init)

    return(trans_init, target_cloud)


def head_to_head(standard_trans, target_cloud, source_in):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    print("\nSetting up next step...")
    mesh = o3d.io.read_triangle_mesh(source_in)
    source_cloud = mesh.sample_points_poisson_disk(100000)

    # Global registration
    voxel_size = 2
    source_down, source_fpfh = preprocess_point_cloud(source_cloud, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)
    result_global = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    trans_init = result_global.transformation

    # Refine registration with ICP
    threshold = 2.00
    result_icp = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000))
    source_cop = copy.deepcopy(source_cloud)
    source_cop.transform(result_icp.transformation)
    print("\nPreview...\n - Press q to contiunue")
    o3d.visualization.draw_geometries([source_cop, target_cloud],
                                      zoom=0.8, front=[50.0, 0.0, 0.0],
                                      lookat=[-1.0, 01.0, 0.0], up=[0, 0, 1],
                                      width=1000, height=1000)
    X2 = result_icp.transformation
    return X2


def head_to_mri(target_cloud, source_in):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    print("\nSetting up MRI surface...")
    mesh = o3d.io.read_triangle_mesh(source_in)
    source_cloud = mesh.sample_points_poisson_disk(100000)

    # Crop MRI to avoid edge effects
    source_crop = copy.deepcopy(source_cloud)
    points = np.asarray(source_crop.points)
    y_threshold = 0.
    source_crop = source_crop.select_by_index(np.where(points[:, 1] > y_threshold)[0])
    points = np.asarray(source_crop.points)
    z_threshold = -100.
    source_crop = source_crop.select_by_index(np.where(points[:, 2] > z_threshold)[0])

    # Global registration
    voxel_size = 2
    source_down, source_fpfh = preprocess_point_cloud(source_crop, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_cloud, voxel_size)
    result_global = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    trans_init = result_global.transformation

    # Refine registration with ICP
    threshold = 2.00
    result_icp = o3d.pipelines.registration.registration_icp(
        source_crop, target_cloud, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000))
    source_cloud.transform(result_icp.transformation)
    print("\nPreview\nPress q to continue")
    o3d.visualization.draw_geometries([source_cloud, target_cloud],
                                      zoom=0.8, front=[50.0, 0.0, 0.0],
                                      lookat=[-1.0, 01.0, 0.0], up=[0, 0, 1],
                                      width=1000, height=1000)
    X3 = copy.deepcopy(result_icp.transformation)
    return(source_cloud, X3)


def check_headpoints(mri_cloud, datafile, X21):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    raw = mne.io.read_raw_fif(datafile, 'default', preload=False)

    # Check results with sensor position based 'digitization points'
    # Load headpoints
    head_points = []
    for chan in raw.info['chs']:
        head_points.append([chan['loc'][0]-chan['loc'][9]*sensor_length,
                            chan['loc'][1]-chan['loc'][10]*sensor_length,
                            chan['loc'][2]-chan['loc'][11]*sensor_length])
    head_points = np.array(head_points)
    head_points = head_points*1000

    # Make into cloud
    head_point_cloud = o3d.geometry.PointCloud()
    head_point_cloud.points = o3d.utility.Vector3dVector(head_points)

    # Make the points blue
    head_point_cloud.paint_uniform_color([0, 0, 1])
    head_point_cloud.transform(X21)

    # Visual check
    print("\nPreview of sensor locations over structural scan")
    o3d.visualization.draw_geometries([mri_cloud, head_point_cloud], width=1000, height=1000)

    # Calculate distances between sensor points and scalp and report median value
    mins = []
    for hp in head_point_cloud.points:
        distances = np.linalg.norm(hp-mri_cloud.points, axis=1)
        mins.append(np.min(distances))
    mins = np.asarray(mins)
    print("Median sensor-scalp distance %.3f mm " % np.median(mins))
    return 0


def write_ouput(datafile, X21, X3):
    # Write the results to file
    raw = mne.io.read_raw_fif(datafile, 'default', preload=True)
    dev_head_t = mne.transforms.Transform("meg", "head", trans=None)
    dev_head_t['trans'] = X21.copy()

    # Metres for mne, mm in open3D
    # so translation elements of transform have to be re-scaled
    # but rotation is fine
    dev_head_t['trans'][0:3, 3] = np.divide(dev_head_t['trans'][0:3, 3], 1000)
    raw.info.update(dev_head_t=dev_head_t)
    raw.save(datafile, overwrite=True)

    # Write seperate trans file for MRI->head transform
    mri_head_t = mne.transforms.Transform("mri", "head", trans=None)
    mri_head_t['trans'] = X3.copy()
    mri_head_t['trans'][0:3, 3] = np.divide(mri_head_t['trans'][0:3, 3], 1000)
    outfile = datafile.split('.fif')[0] + '_trans.fif'
    mne.write_trans(outfile, mri_head_t, overwrite=True)
    return outfile

    def is_file(file):
        if not os.path.isfile(file):
            print(f'Error with {file}.')
            return False
        else:
            return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-om", "--outside_mesh", help="LIDAR scan of head outside the MEG Helmet", required=True)
    parser.add_argument("-im", "--inside_mesh", help="LIDAR scan of head inside the MEG Helmet", required=True)
    parser.add_argument("-s", "--mri_scalp", help="MRI scalp surface from Freesurfer", required=True)
    parser.add_argument("-m", "--megdata", help="MEG data to generate the transform into", nargs='+', required=True)
    parser.add_argument("-lm", "--landmarks", help="Landmarks to use for helmet registration", type=int, nargs='+')

    args = parser.parse_args()

    helmet_mesh = args.inside_mesh
    head_mesh = args.outside_mesh
    mri_cloud = args.mri_scalp

    # Check these files exist.
    for f in [helmet_mesh, head_mesh, mri_cloud]:
        if not os.path.isfile(f):
            print(f'Error opening {f}')
            exit(1)

    if args.landmarks is not None:
        landmarks = args.landmarks
    else:
        landmarks = [1, 2, 3, 4, 5, 6, 7]

    X1 = head_to_helmet(helmet_mesh, landmarks)
    [standard_trans, standard_head] = head_to_standard(head_mesh)
    X2 = head_to_head(standard_trans, standard_head, helmet_mesh)
    X21 = np.dot(X2, X1)
    [mri_cloud, X3] = head_to_mri(standard_head, mri_cloud)

    for MEG_data in args.megdata:
        check_headpoints(mri_cloud, MEG_data, X21)
        outfile = write_ouput(MEG_data, X21, X3)

    print("\nTransforms written to:")
    print(MEG_data)
    print(outfile)


if __name__ == '__main__':
    main()
