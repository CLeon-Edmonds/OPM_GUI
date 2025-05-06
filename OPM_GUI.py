import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['QT_MAC_WANTS_LAYER'] = '1'

import numpy as np
import mne
import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QProgressBar, QFrame, QLineEdit, QGridLayout, QMainWindow, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys
import copy
import time
import open3d as o3d
from mne.coreg import read_fiducials
from mne.io.constants import FIFF
from mne.transforms import apply_trans, read_trans
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import pdist

vtk.vtkObject.GlobalWarningDisplayOff()

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

sensor_length = 6e-3

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down

def execute_global_registration(self, source_down, target_down, source_fpfh,
                              target_fpfh, voxel_size):
    #Global registration using RANSAC and feature matching
    try:
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
        
    except Exception as e:
        print(f"Error in execute_global_registration: {e}")
        import traceback
        traceback.print_exc()
        return None

def validate_fiducials(fiducial_points):
    #Validate fiducial points and return whether they're in a reasonable configuration
    if len(fiducial_points) < 3:
        print("ERROR: Not enough fiducial points")
        return False
        
    #Extract fiducials
    nas = fiducial_points[0]
    lpa = fiducial_points[1]
    rpa = fiducial_points[2]
    
    #Calculate distances between fiducials
    nas_lpa_dist = np.linalg.norm(nas - lpa)
    nas_rpa_dist = np.linalg.norm(nas - rpa)
    lpa_rpa_dist = np.linalg.norm(lpa - rpa)
    
    #Typical ranges based on head anatomy (in mm)
    typical_nas_ear_dist_range = (80, 250)  #Distance from nasion to ears
    typical_ear_to_ear_dist_range = (120, 200)  #Distance between ears
    
    #Check if distances are within reasonable ranges
    valid_nas_lpa = typical_nas_ear_dist_range[0] <= nas_lpa_dist <= typical_nas_ear_dist_range[1]
    valid_nas_rpa = typical_nas_ear_dist_range[0] <= nas_rpa_dist <= typical_nas_ear_dist_range[1]
    valid_lpa_rpa = typical_ear_to_ear_dist_range[0] <= lpa_rpa_dist <= typical_ear_to_ear_dist_range[1]
    
    print(f"Fiducial validation:")
    print(f"  NAS-LPA: {nas_lpa_dist:.1f} mm {'✓' if valid_nas_lpa else '✗'}")
    print(f"  NAS-RPA: {nas_rpa_dist:.1f} mm {'✓' if valid_nas_rpa else '✗'}")
    print(f"  LPA-RPA: {lpa_rpa_dist:.1f} mm {'✓' if valid_lpa_rpa else '✗'}")
    
    #Check if nasion is significantly higher than ears (anatomically correct)
    nas_height = nas[2]
    lpa_height = lpa[2]
    rpa_height = rpa[2]
    avg_ear_height = (lpa_height + rpa_height) / 2
    height_diff = nas_height - avg_ear_height
    
    #Typical nasion should be higher than ears in most coordinate systems
    valid_height = height_diff > 0
    print(f"  Nasion vs ears height diff: {height_diff:.1f} mm {'✓' if valid_height else '✗'}")
    
    #Print overall validation result
    all_valid = valid_nas_lpa and valid_nas_rpa and valid_lpa_rpa
    print(f"Overall fiducial validation: {'PASS' if all_valid else 'WARNING - unusual fiducial configuration'}")
    
    return all_valid

def improved_compute_transform_with_ransac(source_points, target_points, ransac_iterations=1000, inlier_threshold=5.0):
    #Compute transformation with RANSAC for robustness against outliers
    if len(source_points) < 3 or len(target_points) < 3:
        print("ERROR: Not enough points for RANSAC")
        return np.eye(4)
    
    best_inliers = 0
    best_transform = np.eye(4)
    
    print(f"Running RANSAC with {ransac_iterations} iterations, threshold={inlier_threshold}mm")
    
    for _ in range(ransac_iterations):
        #Randomly select 3 points if we have more than 3
        if len(source_points) > 3:
            indices = np.random.choice(len(source_points), 3, replace=False)
            sample_source = source_points[indices]
            sample_target = target_points[indices]
        else:
            sample_source = source_points
            sample_target = target_points
        
        #Compute transformation for this sample
        sample_transform = compute_rigid_transform(sample_source, sample_target)
        
        #Count inliers
        inliers = 0
        for i in range(len(source_points)):
            source_pt = source_points[i]
            target_pt = target_points[i]
            
            #Transform source point
            source_h = np.append(source_pt, 1.0)
            transformed = np.dot(sample_transform, source_h)[:3]
            
            #Check if its an inlier
            error = np.linalg.norm(transformed - target_pt)
            if error < inlier_threshold:
                inliers += 1
        
        #Update best result if needed
        if inliers > best_inliers:
            best_inliers = inliers
            best_transform = sample_transform
    
    print(f"RANSAC found {best_inliers}/{len(source_points)} inliers")
    
    #Refine with all inliers if we have a good initial estimate
    if best_inliers >= 3:
        #Find all inliers
        inlier_source = []
        inlier_target = []
        
        for i in range(len(source_points)):
            source_pt = source_points[i]
            target_pt = target_points[i]
            
            #Transform source point
            source_h = np.append(source_pt, 1.0)
            transformed = np.dot(best_transform, source_h)[:3]
            
            #Check if its an inlier
            error = np.linalg.norm(transformed - target_pt)
            if error < inlier_threshold:
                inlier_source.append(source_pt)
                inlier_target.append(target_pt)
        
        #Final refinement with all inliers
        refined_transform = compute_rigid_transform(np.array(inlier_source), np.array(inlier_target))
        print(f"Refined transform with {len(inlier_source)} inliers")
        return refined_transform
    
    return best_transform

def compute_rigid_transform(source, target):
    #Compute rigid transformation from source to target points
    #Center both point sets
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    
    #Center the points
    source_centered = source - source_center
    target_centered = target - target_center
    
    #Compute the covariance matrix
    H = np.dot(source_centered.T, target_centered)
    
    #SVD factorisation
    U, S, Vt = np.linalg.svd(H)
    
    #Rotation matrix
    R = np.dot(Vt.T, U.T)
    
    #Ensure proper rotation matrix (determinant = 1)
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    
    #Translation
    t = target_center - np.dot(R, source_center)
    
    #Create transformation matrix
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

def two_step_sensor_transformation(raw_sensors, X1, X2):
    # Transform sensors in two explicit steps for better control
    # raw_sensors: numpy array of shape (n, 3) with positions in meters
    # X1: 4x4 transformation matrix from device to helmet
    # X2: 4x4 transformation matrix from helmet to head/standard

    #Convert sensor positions from meters to millimeters
    sensors_mm = raw_sensors * 1000
    
    #Apply X1 to go from device to helmet space
    helmet_space_sensors = []
    for pos in sensors_mm:
        pos_h = np.append(pos, 1.0)
        transformed = np.dot(X1, pos_h)[:3]
        helmet_space_sensors.append(transformed)
    
    helmet_space_sensors = np.array(helmet_space_sensors)
    
    #Check helmet space sensor positions
    helmet_centroid = np.mean(helmet_space_sensors, axis=0)
    print(f"Helmet space sensors centroid: {helmet_centroid}")
    
    #Apply X2 to go from helmet to head/standard space
    head_space_sensors = []
    for pos in helmet_space_sensors:
        pos_h = np.append(pos, 1.0)
        transformed = np.dot(X2, pos_h)[:3]
        head_space_sensors.append(transformed)
    
    head_space_sensors = np.array(head_space_sensors)
    
    #Check head space sensor positions
    head_centroid = np.mean(head_space_sensors, axis=0)
    print(f"Head space sensors centroid: {head_centroid}")
    
    return head_space_sensors

def anatomically_aware_sensor_positioning(transformed_positions_mm, head_points_mm):
    #Adjust sensor positions based on anatomical alignment in millimetres

    #Get head metrics
    head_centroid = np.mean(head_points_mm, axis=0)
    head_max = np.max(head_points_mm, axis=0)
    head_min = np.min(head_points_mm, axis=0)
    head_size = head_max - head_min

    #Get sensor metrics
    sensor_centroid = np.mean(transformed_positions_mm, axis=0)

    #Horizontal alignment (X and Y)
    horizontal_adjustment = np.zeros(3)
    horizontal_adjustment[0] = head_centroid[0] - sensor_centroid[0]  # X
    horizontal_adjustment[1] = head_centroid[1] - sensor_centroid[1]  # Y

    #Vertical alignment (Z): add clearance above the head
    head_height = head_size[2]
    vertical_clearance = 0.15 * head_height  #15% of head height
    vertical_adjustment = np.zeros(3)
    if sensor_centroid[2] < head_max[2]:
        vertical_adjustment[2] = head_max[2] - sensor_centroid[2] + vertical_clearance

    #Forward offset (Y+): place helmet slightly forward relative to head
    head_depth = head_size[1]
    forward_offset = np.zeros(3)
    forward_offset[1] = 0.10 * head_depth  #10% of depth forward

    #Apply all adjustments
    total_adjustment = horizontal_adjustment + vertical_adjustment + forward_offset
    adjusted_positions = transformed_positions_mm + total_adjustment

    return adjusted_positions

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        vtk.vtkInteractorStyleTrackballCamera.__init__(self)
        self.parent = parent
        
    def OnLeftButtonDown(self):
        if not self.GetShiftKey():
            vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonDown(self)
            
    def OnLeftButtonUp(self):
        if not self.GetShiftKey():
            vtk.vtkInteractorStyleTrackballCamera.OnLeftButtonUp(self)

class WorkerThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    
    def __init__(self, task, args=None):
        super().__init__()
        self.task = task
        self.args = args if args is not None else []
        self.result = None
        
    def run(self):
        try:
            if self.task == "load_model":
                #Simulate loading process
                for i in range(101):
                    self.progress_signal.emit(i)
                    self.msleep(30)
                self.finished_signal.emit(None)
                
            elif self.task == "load_outside_msr":
                #Simulate loading process
                for i in range(101):
                    self.progress_signal.emit(i)
                    self.msleep(30)
                self.finished_signal.emit(None)
                
            elif self.task == "head_to_mri":
                #Actual MRI to head registration
                gui = self.args[0]  #Get the GUI instance
                
                #Compute the transformations from head to standard coordinate system
                self.progress_signal.emit(25)
                gui.X2 = gui.computeHeadToStandardTransform()
                
                #Compute the combined transformation from inside MSR to standard
                self.progress_signal.emit(50)
                gui.X21 = np.dot(gui.X2, gui.X1)
                
                #Compute the transformation from MRI to head
                self.progress_signal.emit(75)
                gui.X3 = None
                
                self.progress_signal.emit(100)
                self.finished_signal.emit(None)
                
            elif self.task == "final_registration":
                #This performs the actual head point check and file writing
                gui = self.args[0] if self.args else None
                
                if gui:
                    try:
                        #Compute head to standard transform
                        self.progress_signal.emit(10)
                        gui.X2 = gui.computeHeadToStandardTransform()
                        
                        #Compute combined transform
                        self.progress_signal.emit(30)
                        gui.X21 = np.dot(gui.X2, gui.X1)
                        
                        #Compute MRI to head transform
                        self.progress_signal.emit(50)
                        gui.X3 = None
                
                        self.progress_signal.emit(100)
                        self.finished_signal.emit(None)

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        self.finished_signal.emit(str(e))
                else:
                    for i in range(101):
                        self.progress_signal.emit(i)
                        self.msleep(50)
                    self.finished_signal.emit(None)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(str(e))

class OPMCoRegistrationGUI(QWidget):
        
    def __init__(self):
            super().__init__()

            self.selected_points = []
            self.point_actors = []
            self.file_paths = {}
            self.current_stage = "inside_msr"  

            # Known sensor positions (Change here for new sensor positions)
            self.known_sensor_positions = np.zeros([7, 3], dtype=float)
            self.known_sensor_positions[0] = [2.400627, 6.868828, 15.99383]
            self.known_sensor_positions[1] = [-3.36779, 3.436483, 10.86945]
            self.known_sensor_positions[2] = [-1.86113, -0.49753, 7.031777]
            self.known_sensor_positions[3] = [0.17963, 0.005712, 0.014363]
            self.known_sensor_positions[4] = [3.596149, 4.942685, -5.23564]
            self.known_sensor_positions[5] = [4.021887, 10.33369, -5.79216]
            self.known_sensor_positions[6] = [10.84183, 14.91172, -2.81954]

            # Updated fiducial labels to use "Auricular" instead of "Pre-auricular"
            self.fiducial_labels = ["Nasion", "Left Auricular", "Right Auricular"]
            self.point_labels = []
            self.actors = [] 

            # Storage for fiducial points
            self.fiducial_points = None
            self.fiducials_dict = {}

            # MRI and outside fiducials
            self.outside_fiducials = None
            self.mri_fiducials = None

            # Matrix transformations
            self.X1 = None  # Inside MSR to helmet transform
            self.X2 = None  # Standard transform
            self.X21 = None  # Combined transform
            self.X3 = None  # MRI to head transform

            # For thread safety
            self._source_cloud_points = None
            self._target_cloud_points = None
            self._sensor_points = None
            self._mri_cloud_points = None

            # File selection UI elements
            self.file_labels = {}  # Dictionary to store file status labels

            # Continue GUI setup
            self.initUI()

            # Show startup information
            QTimer.singleShot(500, self.showStartupInfo)

            # Delayed VTK initialization
            QTimer.singleShot(100, self.delayedVTKInitialization)

    def initUI(self):
        self.setWindowTitle("OPM-MEG Co-Registration")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #2e2e2e; color: white;")

        # Create main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left section
        left_frame = QFrame(self)
        left_frame.setFixedWidth(300)
        left_layout = QVBoxLayout()
        left_frame.setLayout(left_layout)
        main_layout.addWidget(left_frame)

        font = QFont("Arial", 12)

        # Add a Help button at the top of the left section
        help_button = QPushButton("?", self)
        help_button.setFont(QFont("Arial", 12))
        help_button.setFixedSize(30, 30)
        help_button.setStyleSheet("background-color: #555555; color: white; border-radius: 15px;")
        help_button.clicked.connect(self.showHelp)
        left_layout.addWidget(help_button, alignment=Qt.AlignRight)

        # File selection section - completely redesigned
        file_types = ["Inside MSR", "Outside MSR", "Scalp File", "OPM Data"]
        self.extensions = {"Inside MSR": "*.ply", "Outside MSR": "*.ply", "Scalp File": "*.stl", "OPM Data": "*.fif"}

        # Create a frame to contain all file selection controls
        files_frame = QFrame()
        files_frame.setStyleSheet("background-color: #333333; border-radius: 8px; padding: 10px;")
        files_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        files_layout = QVBoxLayout(files_frame)
        files_layout.setSpacing(5)  # Reduce spacing between elements
        files_layout.setContentsMargins(5, 5, 5, 5)  # Reduce margins

        title_label = QLabel("File Selection")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        files_layout.addWidget(title_label)

        for file_type in file_types:
            # Create a more compact frame for each file type
            file_frame = QFrame()
            file_frame.setStyleSheet("background-color: #3a3a3a; border-radius: 5px; margin: 2px; padding: 3px;")
            file_layout = QVBoxLayout(file_frame)
            file_layout.setSpacing(2)  # Reduce spacing
            file_layout.setContentsMargins(5, 3, 5, 3)  # Reduce margins

            # File type header - more compact
            type_label = QLabel(file_type)
            type_label.setFont(QFont("Arial", 11, QFont.Bold))
            file_layout.addWidget(type_label)

            # File status indicator
            status_label = QLabel("Not selected")
            status_label.setStyleSheet("color: #FF5555; font-weight: bold; padding: 2px;")
            status_label.setFont(QFont("Arial", 10))
            file_layout.addWidget(status_label)

            # Selection button
            select_button = QPushButton(f"Select {file_type}")
            select_button.setStyleSheet("background-color: #555555; color: white; padding: 3px;")
            select_button.setFixedHeight(25)  # Make buttons smaller
            select_button.clicked.connect(lambda checked, t=file_type: self.selectFile(t))
            file_layout.addWidget(select_button)

            # Store reference to status label
            self.file_labels[file_type] = status_label

            # Add to files section
            files_layout.addWidget(file_frame)
            
            # Add MRI Fiducials button after Scalp File
            if file_type == "Scalp File":
                # Add a "Select MRI Fiducials" button between Scalp File and OPM Data
                fiducials_frame = QFrame()
                fiducials_frame.setStyleSheet("background-color: #3a3a3a; border-radius: 5px; margin: 2px; padding: 3px;")
                fiducials_layout = QVBoxLayout(fiducials_frame)
                fiducials_layout.setSpacing(2)
                fiducials_layout.setContentsMargins(5, 3, 5, 3)
                
                fiducials_label = QLabel("MRI Fiducials")
                fiducials_label.setFont(QFont("Arial", 11, QFont.Bold))
                fiducials_layout.addWidget(fiducials_label)
                
                fiducials_status = QLabel("Not selected")
                fiducials_status.setStyleSheet("color: #FF5555; font-weight: bold; padding: 2px;")
                fiducials_status.setFont(QFont("Arial", 10))
                fiducials_layout.addWidget(fiducials_status)
                
                fiducials_button = QPushButton("Select MRI Fiducials")
                fiducials_button.setStyleSheet("background-color: #555555; color: white; padding: 3px;")
                fiducials_button.setFixedHeight(25)
                fiducials_button.clicked.connect(self.selectMRIFiducials)
                fiducials_layout.addWidget(fiducials_button)
                
                # Store reference to status label
                self.file_labels["MRI Fiducials"] = fiducials_status
                
                # Add to files section
                files_layout.addWidget(fiducials_frame)

        # Add files frame to left layout with stretch to take 70% of the available space
        left_layout.addWidget(files_frame, 7)  # Give it 70% of the available space

        # Add a spacer to push load and continue buttons to the bottom
        left_layout.addStretch(1)

        # Load button
        self.load_button = QPushButton("Load Files", self)
        self.load_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.load_button.setEnabled(False)
        self.load_button.setStyleSheet("background-color: #555555; color: white; padding: 8px; border-radius: 5px;")
        self.load_button.clicked.connect(self.loadModel)
        left_layout.addWidget(self.load_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #007BFF;
                width: 20px;
            }
        """)
        left_layout.addWidget(self.progress_bar)

        # Middle section
        middle_frame = QFrame(self)
        middle_frame.setFixedWidth(600)
        middle_layout = QVBoxLayout()
        middle_frame.setLayout(middle_layout)
        main_layout.addWidget(middle_frame)

        self.instructions_label = QLabel("", self)
        self.instructions_label.setFont(QFont("Arial", 14))
        self.instructions_label.setAlignment(Qt.AlignCenter)
        self.instructions_label.setStyleSheet("padding: 10px; background-color: #444444; border-radius: 5px;")
        middle_layout.addWidget(self.instructions_label)

        # Create VTK widget - Modified for Mac compatibility
        try:
            print("Creating VTK widget...")
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            middle_layout.addWidget(self.vtk_widget)

            # We'll initialize VTK components in delayedVTKInitialization
            self.ren = None
            self.iren = None

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error setting up VTK widget: {e}")
            # Fallback to a basic widget if VTK fails
            fallback_widget = QWidget()
            fallback_widget.setStyleSheet("background-color: #333333;")
            middle_layout.addWidget(fallback_widget)
            self.vtk_widget = None

        # Right section 
        right_frame = QFrame(self)
        right_frame.setFixedWidth(300)
        right_layout = QVBoxLayout()
        right_frame.setLayout(right_layout)
        main_layout.addWidget(right_frame)

        # Status section
        self.status_label = QLabel("Point Selection Status", self)
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.status_label)

        # Points status layout
        self.point_status_layout = QVBoxLayout()

        # For Inside MSR: 7 points
        self.distance_labels = []
        self.distance_boxes = []

        for i in range(7):
            label = QLabel(f"Point {i+1}: Not selected", self)
            label.setFont(font)
            box = QLineEdit(self)
            box.setFont(font)
            box.setReadOnly(True)
            box.setStyleSheet("background-color: #444444; color: white;")

            self.distance_labels.append(label)
            self.distance_boxes.append(box)

            self.point_status_layout.addWidget(label)
            self.point_status_layout.addWidget(box)

        right_layout.addLayout(self.point_status_layout)

        # Button controls with the new layout you requested
        button_grid_layout = QGridLayout()

        self.clear_button = QPushButton("Clear Points", self)
        self.clear_button.setFont(font)
        self.clear_button.setStyleSheet("background-color: #FF5733; color: white; font-weight: bold;")
        self.clear_button.clicked.connect(self.clearPoints)
        button_grid_layout.addWidget(self.clear_button, 0, 0)

        self.reverse_button = QPushButton("Reverse Last Point", self)
        self.reverse_button.setFont(font)
        self.reverse_button.setStyleSheet("background-color: #FFA500; color: white; font-weight: bold;")
        self.reverse_button.clicked.connect(self.reverseLastPoint)
        button_grid_layout.addWidget(self.reverse_button, 0, 1)

        self.confirm_button = QPushButton("Confirm Points", self)
        self.confirm_button.setFont(font)
        self.confirm_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self.confirmPoints)
        button_grid_layout.addWidget(self.confirm_button, 1, 0)

        # New Fit Points button
        self.fit_button = QPushButton("Fit Points", self)
        self.fit_button.setFont(font)
        self.fit_button.setStyleSheet("background-color: #9370DB; color: white; font-weight: bold;")
        self.fit_button.setEnabled(False)
        self.fit_button.clicked.connect(self.fitPoints)  # Connect to new method
        button_grid_layout.addWidget(self.fit_button, 1, 1)

        right_layout.addLayout(button_grid_layout)

        # Add continue button at the bottom right
        right_layout.addStretch()

        self.continue_button = QPushButton("Continue", self)
        self.continue_button.setFont(font)
        self.continue_button.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold;")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continueWorkflow)
        right_layout.addWidget(self.continue_button)

        # Save button for final step
        self.save_button = QPushButton("Save Results", self)
        self.save_button.setFont(font)
        self.save_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.saveResults)
        right_layout.addWidget(self.save_button)
        self.save_button.hide()  # Hide until needed

        # Initialize instructions
        self.updateInstructions()

        # Show the widget - must be called before rendering
        self.show()

        # Allow the widget to be properly realized
        QApplication.processEvents()
        
    def loadModel(self):
        """Handler for load button with improved coloring"""
        try:
            print("Load button clicked")
            
            # Check if Inside MSR file is selected
            if "Inside MSR" not in self.file_paths or not self.file_paths["Inside MSR"]:
                QMessageBox.warning(self, "Missing Files", "Please select the Inside MSR file.")
                return
            
            self.load_button.setEnabled(False)
            self.load_button.setText("Loading...")
            self.progress_bar.setValue(0)
            QApplication.processEvents()  # Force UI update
            
            file_path = self.file_paths["Inside MSR"]
            print(f"Loading PLY file: {file_path}")
            
            # Clear previous actors except point markers
            for actor in self.actors:
                if actor not in self.point_actors:
                    self.ren.RemoveActor(actor)
            self.actors = [actor for actor in self.actors if actor in self.point_actors]
            
            # Use a simpler approach that won't freeze - just points with better coloring
            try:
                # Read directly with VTK reader
                reader = vtk.vtkPLYReader()
                reader.SetFileName(file_path)
                reader.Update()
                
                if reader.GetOutput().GetNumberOfPoints() > 0:
                    print(f"VTK loaded {reader.GetOutput().GetNumberOfPoints()} points")
                    
                    # Create mapper and actor for direct point rendering
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(reader.GetOutputPort())
                    
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetPointSize(2)  # Moderate point size
                    # Use light gray instead of white for better detail visibility
                    actor.GetProperty().SetColor(0.8, 0.8, 0.8)  
                    
                    # Add actor
                    self.ren.AddActor(actor)
                    self.actors.append(actor)
                    
                    # Set black background for contrast
                    self.ren.SetBackground(0.0, 0.0, 0.0)
                    
                    # Set camera
                    self.ren.ResetCamera()
                    camera = self.ren.GetActiveCamera()
                    camera.Azimuth(30)
                    camera.Elevation(20)
                    
                    # Add ambient lighting to see details better
                    light = vtk.vtkLight()
                    light.SetLightTypeToHeadlight()
                    light.SetIntensity(0.7)  # Less intense light
                    self.ren.AddLight(light)
                    
                    # Add a second light from a different angle
                    side_light = vtk.vtkLight()
                    side_light.SetPosition(500, 0, 0)
                    side_light.SetFocalPoint(0, 0, 0)
                    side_light.SetIntensity(0.3)  # Soft side lighting
                    side_light.SetColor(0.8, 0.8, 1.0)  # Slightly bluish light
                    self.ren.AddLight(side_light)
                    
                    # Final render
                    self.vtk_widget.GetRenderWindow().Render()
                    
                    print("Model loaded successfully")
                else:
                    # Fall back to Open3D with downsampling
                    print("VTK reader failed, falling back to Open3D")
                    pcd = o3d.io.read_point_cloud(file_path)
                    points = np.asarray(pcd.points)
                    points /= 1000.0  # mm → m
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # Aggressively downsample for performance
                    max_points = 50000  # Limit to 50K points for performance
                    if len(points) > max_points:
                        print(f"Downsampling from {len(points)} to {max_points} points")
                        indices = np.random.choice(len(points), max_points, replace=False)
                        points = points[indices]
                    
                    # Create VTK points
                    vtk_points = vtk.vtkPoints()
                    for point in points:
                        vtk_points.InsertNextPoint(point)
                    
                    polydata = vtk.vtkPolyData()
                    polydata.SetPoints(vtk_points)
                    
                    # Create vertices
                    vertices = vtk.vtkCellArray()
                    for i in range(vtk_points.GetNumberOfPoints()):
                        vertex = vtk.vtkVertex()
                        vertex.GetPointIds().SetId(0, i)
                        vertices.InsertNextCell(vertex)
                    polydata.SetVerts(vertices)
                    
                    # Create mapper and actor
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputData(polydata)
                    
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetPointSize(2)
                    # Use light gray color for better detail visibility
                    actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                    
                    # Add actor
                    self.ren.AddActor(actor)
                    self.actors.append(actor)
                    
                    # Set camera
                    self.ren.ResetCamera()
                    camera = self.ren.GetActiveCamera()
                    camera.Azimuth(30)
                    camera.Elevation(20)
                    
                    # Final render
                    self.vtk_widget.GetRenderWindow().Render()
                    
                    print("Model loaded via Open3D fallback")
            except Exception as e:
                print(f"Error loading PLY: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Loading Error", f"Error loading model: {str(e)}")
            
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            # Update UI
            self.current_stage = "inside_msr"
            self.updateInstructions()
            self.clearPoints()
            
            # Re-enable load button
            self.load_button.setEnabled(True)
            self.load_button.setText("Load Files")
            
            print("Loading completed successfully")
            
        except Exception as e:
            print(f"Error in loadModel: {e}")
            import traceback
            traceback.print_exc()
            self.load_button.setEnabled(True)
            self.load_button.setText("Load Files")
            QMessageBox.critical(self, "Error", f"Error loading model: {str(e)}")
                
    def updateProgress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # Force UI update

    def loadModelFinished(self, error):
        """Handle completion of model loading"""
        try:
            if error:
                print(f"Worker reported error: {error}")
                QMessageBox.critical(self, "Loading Error", f"Error loading model: {error}")
                self.load_button.setEnabled(True)
                self.load_button.setText("Load Files")
                return
                
            file_path = self.file_paths["Inside MSR"]
            print(f"Loading file: {file_path}")
            
            # Clear previous actors except point markers
            for actor in self.actors:
                if actor not in self.point_actors:
                    self.ren.RemoveActor(actor)
            self.actors = [actor for actor in self.actors if actor in self.point_actors]
            
            # Create reader
            reader = vtk.vtkPLYReader()
            reader.SetFileName(file_path)
            reader.Update()
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetPointSize(2)
            actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # White color
            
            # Set black background
            self.ren.SetBackground(0.0, 0.0, 0.0)
            
            # Add to renderer
            self.ren.AddActor(actor)
            self.actors.append(actor)
            
            # Reset camera
            self.ren.ResetCamera()
            
            # Final render
            self.vtk_widget.GetRenderWindow().Render()
            
            # Reset selected points
            self.clearPoints()
            
            # Update GUI state
            self.current_stage = "inside_msr"
            self.updateInstructions()
            
            # Re-enable load button
            self.load_button.setEnabled(True)
            self.load_button.setText("Load Files")
            
            print("Loading completed successfully")
            
        except Exception as e:
            print(f"Error in loadModelFinished: {e}")
            import traceback
            traceback.print_exc()
            self.load_button.setEnabled(True)
            self.load_button.setText("Load Files")
            QMessageBox.critical(self, "Error", f"Error finishing model load: {str(e)}")
                        
    def forceExtraRender(self):
        """Force extra render to ensure visibility"""
        try:
            print("Forcing extra render...")
            
            # Make sure we can see the model by showing a simple test object
            # Add visible axes
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(100, 100, 100)
            axes.SetShaftTypeToCylinder()
            axes.SetCylinderRadius(0.02)
            
            self.ren.AddActor(axes)
            self.actors.append(axes)
            
            # Reset camera again
            self.ren.ResetCamera()
            
            # Force rendering
            self.vtk_widget.GetRenderWindow().Render()
            
            print("Extra render complete")
            
        except Exception as e:
            print(f"Error in forceExtraRender: {e}")
                        
    def delayedVTKInitialization(self):
        try:
            if not self.vtk_widget:
                print("VTK widget not available, skipping initialization")
                return
                
            print("Initializing VTK renderer with safe settings...")
            
            # Create a renderer - exactly as in your GitHub code
            self.ren = vtk.vtkRenderer()
            self.ren.SetBackground(0.1, 0.1, 0.1)  # Dark background
            
            # Add renderer to render window
            render_window = self.vtk_widget.GetRenderWindow()
            render_window.AddRenderer(self.ren)
            
            # Create interactor - exactly as in your GitHub code
            self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
            
            # Set interactor style
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.iren.SetInteractorStyle(style)
            
            # Initialize interactor
            print("Initializing VTK interactor...")
            self.iren.Initialize()
            
            # Add event handling
            self.iren.AddObserver("LeftButtonPressEvent", 
                                lambda obj, event: self.safeLeftButtonEvent(obj, event))
            
            # Test render with a simple cube to verify rendering works
            test_actor = self.createTestCube()
            self.ren.AddActor(test_actor)
            
            # Initial render
            self.vtk_widget.GetRenderWindow().Render()
            
            # Remove test actor after successful render
            self.ren.RemoveActor(test_actor)
            
            print("VTK initialization complete")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in VTK initialization: {e}")

    def debugVizRendering(self):
        """Comprehensive debug for visualization issues"""
        try:
            print("\n=== VISUALIZATION DEBUG ===")
            
            # Clear everything
            for actor in self.actors:
                self.ren.RemoveActor(actor)
            self.actors.clear()
            
            # 1. Create a very simple sphere (definitely should work)
            print("Creating test sphere...")
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(0, 0, 0)
            sphere.SetRadius(50.0)  # Large radius to be very visible
            sphere.SetPhiResolution(20)
            sphere.SetThetaResolution(20)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)  # Red
            
            # Add actor
            self.ren.AddActor(actor)
            self.actors.append(actor)
            
            # 2. Add axes for orientation
            print("Adding axes...")
            axes = vtk.vtkAxesActor()
            axes.SetTotalLength(100, 100, 100)
            axes.SetShaftTypeToLine()
            
            self.ren.AddActor(axes)
            self.actors.append(axes)
            
            # 3. Set black background for contrast
            self.ren.SetBackground(0, 0, 0)
            
            # 4. Reset camera with specific positioning
            print("Setting camera position...")
            self.ren.ResetCamera()
            camera = self.ren.GetActiveCamera()
            camera.SetPosition(0, 0, 500)  # Position camera far back on Z axis
            camera.SetFocalPoint(0, 0, 0)   # Look at origin
            camera.SetViewUp(0, 1, 0)       # Y axis is up
            
            # 5. Force multiple renders
            print("Forcing multiple renders...")
            self.vtk_widget.GetRenderWindow().Render()
            QApplication.processEvents()
            
            # Second render after a delay
            QTimer.singleShot(200, lambda: self.forceSecondRender())
            
            print("Debug visualization steps completed")
            return True
            
        except Exception as e:
            print(f"Visualization debug error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def forceSecondRender(self):
        """Force a second render with camera reset"""
        try:
            print("Forcing second render...")
            self.ren.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            print("Second render complete")
            
            # Try adding a text label in case visuals are working but not visible
            text = vtk.vtkTextActor()
            text.SetInput("DEBUG TEXT - IF YOU CAN SEE THIS, RENDERING WORKS")
            text.GetTextProperty().SetColor(1, 1, 1)  # White
            text.GetTextProperty().SetFontSize(24)
            text.SetPosition(10, 10)
            
            self.ren.AddActor(text)
            self.actors.append(text)
            
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"Second render error: {e}")
            
    def createTestCube(self):
        """Create a test cube for verifying rendering"""
        cube = vtk.vtkCubeSource()
        cube.SetXLength(10)
        cube.SetYLength(10)
        cube.SetZLength(10)
        cube.SetCenter(0, 0, 0)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cube.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # Red
        
        return actor
                      
    def handleMouseClick(self, obj, event):
        """Direct handler for mouse clicks"""
        try:
            if self.iren.GetShiftKey():
                # Get click position
                x, y = self.iren.GetEventPosition()
                
                # Use picker to determine 3D position
                if self.picker.Pick(x, y, 0, self.ren):
                    # Get the picked position
                    pos = self.picker.GetPickPosition()
                    print(f"Picked point at: {pos}")
                    
                    # Add the point
                    self.safeAddPoint(pos)
                else:
                    # If no point was picked, use a ray cast approach
                    print("No point picked directly, using ray cast")
                    
                    # Get camera parameters
                    camera = self.ren.GetActiveCamera()
                    focal_point = list(camera.GetFocalPoint())
                    position = list(camera.GetPosition())
                    
                    # Compute ray from camera through the clicked point
                    renderer = self.iren.GetRenderWindow().GetRenderers().GetFirstRenderer()
                    renderer.SetDisplayPoint(x, y, 0)
                    renderer.DisplayToWorld()
                    world_point = renderer.GetWorldPoint()
                    
                    # Create normalized ray
                    ray_point = world_point[:3]
                    ray_direction = [
                        ray_point[0] - position[0],
                        ray_point[1] - position[1],
                        ray_point[2] - position[2]
                    ]
                    
                    # Normalize the ray direction
                    magnitude = np.sqrt(sum([x*x for x in ray_direction]))
                    if magnitude > 0:
                        ray_direction = [x/magnitude for x in ray_direction]
                        
                        # Use a reasonable distance along the ray
                        distance = 200  # Adjust based on your model size
                        point_pos = [
                            position[0] + ray_direction[0] * distance,
                            position[1] + ray_direction[1] * distance,
                            position[2] + ray_direction[2] * distance
                        ]
                        
                        print(f"Created point via ray cast at: {point_pos}")
                        self.safeAddPoint(point_pos)
        except Exception as e:
            print(f"Error handling mouse click: {e}")
            import traceback
            traceback.print_exc()

    def safeAddPoint(self, position):
        """Safely add a selected point"""
        try:
            print(f"Adding point at: {position}")
            
            # Add to selected points
            self.selected_points.append(position)
            
            # Create sphere to visualize the point
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(position)
            sphere.SetRadius(3.0)
            sphere.SetPhiResolution(8)
            sphere.SetThetaResolution(8)
            
            # Create mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1, 0, 0)  # Red
            
            # Add to scene
            self.ren.AddActor(actor)
            self.point_actors.append(actor)
            self.actors.append(actor)
            
            # Update UI
            idx = len(self.selected_points) - 1
            if self.current_stage == "inside_msr":
                if idx < len(self.distance_labels):
                    self.distance_labels[idx].setText(f"Point {idx+1}: Selected")
                    # Update box to neutral color initially
                    self.distance_boxes[idx].setText("Pending confirmation")
                    self.distance_boxes[idx].setStyleSheet("background-color: #444444; color: white;")
            elif self.current_stage == "outside_msr":
                fiducial_idx = idx
                if fiducial_idx < len(self.fiducial_labels):
                    self.distance_labels[fiducial_idx].setText(f"{self.fiducial_labels[fiducial_idx]}: Selected")
                    self.distance_boxes[fiducial_idx].setText("Pending confirmation")
                    self.distance_boxes[fiducial_idx].setStyleSheet("background-color: #444444; color: white;")
                    label = self.fiducial_labels[fiducial_idx]
                    if label == "Nasion":
                        self.point_labels.append("NAS")
                    elif label == "Left Auricular":
                        self.point_labels.append("LPA")
                    elif label == "Right Auricular":
                        self.point_labels.append("RPA")
            
            # Enable buttons if enough points
            if ((self.current_stage == "inside_msr" and len(self.selected_points) >= 7) or
                (self.current_stage == "outside_msr" and len(self.selected_points) >= 3)):
                self.confirm_button.setEnabled(True)
                if self.current_stage == "inside_msr":
                    self.fit_button.setEnabled(True)
            
            # Render
            self.vtk_widget.GetRenderWindow().Render()
            
        except Exception as e:
            print(f"Error in safeAddPoint: {e}")
            import traceback
            traceback.print_exc()
            
    def closeEvent(self, event):
        """Properly clean up VTK resources when closing"""
        try:
            if hasattr(self, 'vtk_widget') and self.vtk_widget:
                self.vtk_widget.GetRenderWindow().Finalize()
                self.vtk_widget.close()
        except Exception as e:
            print(f"Error during closeEvent: {e}")
        super().closeEvent(event)

    def safeLeftButtonEvent(self, obj, event):
        """Safely handle left button click events"""
        try:
            if self.iren.GetShiftKey():
                # Get the click position
                clickPos = self.iren.GetEventPosition()
                
                # Create a picker to get the 3D position
                picker = vtk.vtkPropPicker()
                picker.Pick(clickPos[0], clickPos[1], 0, self.ren)
                pos = picker.GetPickPosition()
                
                print(f"Picked position: {pos}")
                
                # Add point at the clicked position
                self.safeAddPoint(pos)
                
        except Exception as e:
            print(f"Error in safeLeftButtonEvent: {e}")
            import traceback
            traceback.print_exc()

    def showStartupInfo(self):
        QMessageBox.information(self, "Prerequisites", 
                               "Welcome to OPM-MEG Co-Registration\n\n"
                               "This application helps you align OPM-MEG sensor positions with MRI data.\n\n"
                               "Before starting, make sure you have:\n"
                               "1. Inside MSR scan (PLY file)\n"
                               "2. Outside MSR scan (PLY file)\n"
                               "3. MRI scalp surface (STL file from FreeSurfer)\n"
                               "4. OPM-MEG data (.fif file)\n\n"
                               "Click the '?' button for detailed help.")

    def showHelp(self):
        help_text = """
        <h3>OPM-MEG Co-Registration Help</h3>
        
        <h4>Required Preprocessing Steps:</h4>
        <ol>
            <li>Process MRI data with FreeSurfer:
                <pre>recon-all -i your_t1.nii -s subject_name -all</pre>
            </li>
            <li>Generate scalp surface STL file:
                <pre>mris_convert $SUBJECTS_DIR/subject_name/surf/lh.seghead.surf scalp.stl</pre>
                If seghead surface is not available, use:
                <pre>mri_watershed -surf $SUBJECTS_DIR/subject_name/mri/T1.mgz $SUBJECTS_DIR/subject_name/surf/scalp
mris_convert $SUBJECTS_DIR/subject_name/surf/scalp scalp.stl</pre>
            </li>
            <li>Acquire OPM data (.fif file)</li>
            <li>Scan head inside and outside MSR (PLY files)</li>
        </ol>
        
        <h4>Workflow Steps:</h4>
        <ol>
            <li>Load all required files</li>
            <li>Mark seven helmet points in the inside MSR scan</li>
            <li>Mark fiducial points in the outside MSR scan</li>
            <li>Complete registration</li>
            <li>Save transformation files</li>
        </ol>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Help")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(help_text)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def updateInstructions(self):
        if self.current_stage == "inside_msr":
            self.instructions_label.setText(
                "Step 1: "
                "Select the 7 helmet label points from left to right\n"
                "Use Shift+Left Click to select points"
            )
            self.status_label.setText("Inside MSR Point Selection Status")
            
            #7 points should show up here.
            self.updatePointStatusPanel(7)
            
            #Show all buttons for inside MSR stage
            self.clear_button.show()
            self.reverse_button.show()
            self.confirm_button.show()
            self.fit_button.show()
            
        elif self.current_stage == "outside_msr":
            self.instructions_label.setText(
                "Step 2: Select the 3 fiducial points in this order:\n"
                "1. Naison\n"
                "2. Left pre-auricular\n"
                "3. Right pre-auricular\n"
                "Use Shift+Left Click to select points"
            )
            self.status_label.setText("Fiducial Point Selection Status")
            
            #Fiducial points to show up here
            self.updatePointStatusPanel(3, fiducials=True)
            
            #Only show clear and reverse buttons for outside MSR stage
            self.clear_button.show()
            self.reverse_button.show()
            self.confirm_button.show()
            self.fit_button.hide()
            
        elif self.current_stage == "mri_scalp":
            self.instructions_label.setText(
                "Step 3: Performing MRI scalp to head registration\n"
                "Preview will be shown when complete"
            )
            
            #Hide selection related buttons during processing
            self.clear_button.hide()
            self.reverse_button.hide()
            self.confirm_button.hide()
            self.fit_button.hide()
            
        elif self.current_stage == "finished":
            self.instructions_label.setText(
                "Co-registration complete!\n"
                "Click 'Save Results' to save the transformation files"
            )
            self.save_button.show()
            self.save_button.setEnabled(True)
            
            #Hide all buttons except save
            self.clear_button.hide()
            self.reverse_button.hide()
            self.confirm_button.hide()
            self.fit_button.hide()
            self.continue_button.hide()

    def updatePointStatusPanel(self, num_points, fiducials=False):
        for i in range(len(self.distance_labels)):
            self.distance_labels[i].setParent(None)
            self.distance_boxes[i].setParent(None)
        
        self.distance_labels.clear()
        self.distance_boxes.clear()
        
        font = QFont("Arial", 12)
        
        for i in range(num_points):
            if fiducials:
                label = QLabel(f"{self.fiducial_labels[i]}: Not selected", self)
            else:
                label = QLabel(f"Point {i+1}: Not selected", self)
                
            label.setFont(font)
            box = QLineEdit(self)
            box.setFont(font)
            box.setReadOnly(True)
            box.setStyleSheet("background-color: #444444; color: white;")
            
            self.distance_labels.append(label)
            self.distance_boxes.append(box)
            
            self.point_status_layout.addWidget(label)
            self.point_status_layout.addWidget(box)

    def selectFile(self, file_type):
        """Handler for file selection buttons"""
        options = QFileDialog.Options()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            f"Select {file_type} File", 
            "", 
            self.extensions[file_type], 
            options=options
        )
        
        if file_path:
            print(f"Selected {file_type} file: {file_path}")
            
            # For Scalp File, ensure it's an STL
            if file_type == "Scalp File" and not file_path.lower().endswith('.stl'):
                QMessageBox.warning(self, "Invalid File", 
                                "The scalp file must be an STL file generated from FreeSurfer.")
                return
            
            # Store the file path
            self.file_paths[file_type] = file_path
            
            # Update file status in UI using file_labels dictionary
            if file_type in self.file_labels:
                # Make the change very visible
                filename = os.path.basename(file_path)
                self.file_labels[file_type].setText(f"🟢 Selected: {filename}")
                self.file_labels[file_type].setStyleSheet("color: #55FF55; font-weight: bold; padding: 5px;")
                print(f"Updated status for {file_type}")
            else:
                print(f"Warning: Could not find label for {file_type} in self.file_labels")
            
            # Check if Inside MSR file is selected to enable the load button
            if "Inside MSR" in self.file_paths and self.file_paths["Inside MSR"]:
                self.load_button.setText("LOAD FILES")
                self.load_button.setEnabled(True)
                self.load_button.setStyleSheet("background-color: #00AA00; color: white; font-weight: bold; padding: 10px;")
                print("Load button enabled")
            
            # Force update of the UI
            QApplication.processEvents()

    def selectMRIFiducials(self):
        """Handler for MRI fiducials button with improved error handling"""
        options = QFileDialog.Options()
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select MRI Fiducials File", 
            "", 
            "FIF Files (*.fif)", 
            options=options
        )
        
        if file_path:
            print(f"Selected MRI fiducials file: {file_path}")
            
            # Store the file path
            self.file_paths["MRI Fiducials"] = file_path
            
            # Try to load to verify it's valid
            try:
                fiducials_dict = self.load_mri_fiducials(file_path)
                if fiducials_dict:
                    print("Successfully loaded MRI fiducials")
                    # Explicitly assign to both variables for safety
                    self.mri_fiducials_dict = fiducials_dict
                    self.mri_fiducials = fiducials_dict  # Create both for compatibility
                    
                    # Update status label
                    filename = os.path.basename(file_path)
                    self.file_labels["MRI Fiducials"].setText(f"🟢 Selected: {filename}")
                    self.file_labels["MRI Fiducials"].setStyleSheet("color: #55FF55; font-weight: bold; padding: 5px;")
                    
                    # Print the loaded fiducials for debugging
                    print("\n=== MRI FIDUCIALS LOADED ===")
                    for key, value in fiducials_dict.items():
                        print(f"{key}: {value}")
                else:
                    QMessageBox.warning(self, "Invalid File", "Could not load fiducials from the selected file.")
            except Exception as e:
                print(f"Error loading MRI fiducials: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.warning(self, "Invalid File", f"Error loading fiducials: {str(e)}")

    def display_mri_scalp(self):
        """Display MRI scalp model in the main VTK window instead of Open3D window"""
        try:
            if not hasattr(self, 'mri_scalp') or self.mri_scalp is None:
                if "Scalp File" in self.file_paths:
                    self.mri_scalp = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 50000)
                else:
                    print("No scalp file loaded.")
                    return

            # Use the VTK window instead of Open3D
            if hasattr(self, 'mri_scalp') and self.mri_scalp is not None:
                # Convert Open3D point cloud to VTK points for visualization
                points = np.asarray(self.mri_scalp.points)
                
                # Subsample for performance if needed
                max_points = 50000
                if len(points) > max_points:
                    indices = np.random.choice(len(points), max_points, replace=False)
                    points = points[indices]
                
                # Clear previous actors except point markers
                for actor in self.actors:
                    if actor not in self.point_actors:
                        self.ren.RemoveActor(actor)
                
                # Create VTK points
                vtk_points = vtk.vtkPoints()
                for point in points:
                    vtk_points.InsertNextPoint(point)
                
                # Create polydata
                polydata = vtk.vtkPolyData()
                polydata.SetPoints(vtk_points)
                
                # Create vertices
                vertices = vtk.vtkCellArray()
                for i in range(vtk_points.GetNumberOfPoints()):
                    vertex = vtk.vtkVertex()
                    vertex.GetPointIds().SetId(0, i)
                    vertices.InsertNextCell(vertex)
                polydata.SetVerts(vertices)
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0, 1, 0)  # Green
                actor.GetProperty().SetPointSize(3)
                
                # Add actor and store reference
                self.ren.AddActor(actor)
                self.actors.append(actor)
                
                # Reset camera and render
                self.ren.ResetCamera()
                self.vtk_widget.GetRenderWindow().Render()
                
                print("Displayed MRI scalp in main VTK window.")
        except Exception as e:
            print(f"Error displaying MRI scalp: {e}")
            import traceback
            traceback.print_exc()

    def reverseLastPoint(self):
        if self.point_actors:
            print("🔄 Reversing last point")
            last_actor = self.point_actors.pop()
            self.ren.RemoveActor(last_actor)
            if last_actor in self.actors:
                self.actors.remove(last_actor)
                    
            self.selected_points.pop()
            
            # Update point status
            idx = len(self.selected_points)
            if idx < len(self.distance_labels):
                if self.current_stage == "inside_msr":
                    self.distance_labels[idx].setText(f"Point {idx+1}: Not selected")
                elif self.current_stage == "outside_msr" and idx < len(self.fiducial_labels):
                    self.distance_labels[idx].setText(f"{self.fiducial_labels[idx]}: Not selected")
                
                if idx < len(self.distance_boxes):
                    self.distance_boxes[idx].setText("")
                    self.distance_boxes[idx].setStyleSheet("background-color: #444444; color: white;")
            
            # Disable confirm if we no longer have enough points
            if ((self.current_stage == "inside_msr" and len(self.selected_points) < 7) or 
                (self.current_stage == "outside_msr" and len(self.selected_points) < 3)):
                self.confirm_button.setEnabled(False)
                self.fit_button.setEnabled(False)
            
            # Render the changes
            self.vtk_widget.GetRenderWindow().Render()
            print(f"Removed last point. Remaining points: {len(self.selected_points)}")
        else:
            print("No points to reverse")

    def loadPointCloud(self, file_type):
        """Load a specific type of point cloud based on the selected file"""
        try:
            if file_type not in self.file_paths or not self.file_paths[file_type]:
                QMessageBox.warning(self, "Missing File", f"Please select the {file_type} file first.")
                return
            
            file_path = self.file_paths[file_type]
            print(f"Loading {file_type} from {file_path}")
            
            # Clear existing actors except point markers
            for actor in self.actors:
                if actor not in self.point_actors:
                    self.ren.RemoveActor(actor)
            self.actors = [actor for actor in self.actors if actor in self.point_actors]
            
            # Load the file based on its extension
            if file_path.lower().endswith('.ply'):
                # Load PLY with VTK reader
                reader = vtk.vtkPLYReader()
                reader.SetFileName(file_path)
                reader.Update()
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetPointSize(2)
                actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # White
                
                # Add to renderer
                self.ren.AddActor(actor)
                self.actors.append(actor)
                
            elif file_path.lower().endswith('.stl'):
                # Load STL with VTK reader
                reader = vtk.vtkSTLReader()
                reader.SetFileName(file_path)
                reader.Update()
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(reader.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green for STL
                
                # Add to renderer
                self.ren.AddActor(actor)
                self.actors.append(actor)
            
            # Reset camera
            self.ren.ResetCamera()
            
            # Final render
            self.vtk_widget.GetRenderWindow().Render()
            
            print(f"Loaded {file_type} successfully")
            
        except Exception as e:
            print(f"Error loading {file_type}: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error loading {file_type}: {str(e)}")
                    
    def loadPointCloudOpen3D(self, file_path, max_points=10000):
        """Load a point cloud using Open3D with improved STL handling."""
        print(f"Loading point cloud from {file_path}")
        
        # Handle STL files differently from PLY files
        if file_path.lower().endswith('.stl'):
            try:
                # Load as mesh
                mesh = o3d.io.read_triangle_mesh(file_path)
                vertices = np.asarray(mesh.vertices)
                vertices /= 1000.0  # mm → m
                mesh.vertices = o3d.utility.Vector3dVector(vertices)

                # Sample points from the mesh surface for better representation
                pcd = mesh.sample_points_uniformly(number_of_points=max_points*2)
                
                # Ensure normals are computed for better visualization
                pcd.estimate_normals()
                
                print(f"Loaded STL mesh and sampled {len(pcd.points)} points")
                return pcd
            except Exception as e:
                print(f"Error loading STL file: {e}")
                import traceback
                traceback.print_exc()
        
        # Handle PLY files (original behavior)
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            
            if len(pcd.points) > max_points:
                pcd = pcd.random_down_sample(max_points / len(pcd.points))
            
            print(f"Loaded point cloud with {len(pcd.points)} points")
            return pcd
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty point cloud in case of error
            empty_pcd = o3d.geometry.PointCloud()
            return empty_pcd
        
    def load_mri_fiducials(self, fid_path):
        try:
            print(f"Loading MRI fiducials from: {fid_path}")
            fiducials_list, _ = read_fiducials(fid_path)  # Unpack the tuple properly

            if not fiducials_list or len(fiducials_list) < 3:
                print(f"Warning: Not enough fiducials found in {fid_path}")
                print(f"Found {len(fiducials_list) if fiducials_list else 0} fiducial points")
                return None

            fid_dict = {}
            for f in fiducials_list:
                ident = int(f['ident'])  # For MNE 1.0, each item is a dict-like DigPoint
                coords = f['r'] * 1000   # Convert to mm

                if ident == FIFF.FIFFV_POINT_NASION:
                    fid_dict['NAS'] = coords
                    print(f"Found Nasion at {coords}")
                elif ident == FIFF.FIFFV_POINT_LPA:
                    fid_dict['LPA'] = coords
                    print(f"Found LPA at {coords}")
                elif ident == FIFF.FIFFV_POINT_RPA:
                    fid_dict['RPA'] = coords
                    print(f"Found RPA at {coords}")

            if len(fid_dict) < 3:
                missing = set(['NAS', 'LPA', 'RPA']) - set(fid_dict.keys())
                print(f"Warning: Missing fiducials: {missing}")
                return None

            print("=== MRI Fiducials (in mm) ===")
            for name, val in fid_dict.items():
                print(f"{name}: {val}")

            return fid_dict

        except Exception as e:
            print(f"Error loading fiducials: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def visualize_initial_alignment(self, mri_mesh, scan_mesh):
        #Create new render window for visualization
        render_window = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        render_window.AddRenderer(renderer)
        
        #Convert MRI mesh to actor with green color
        mri_mapper = vtk.vtkPolyDataMapper()
        mri_points = vtk.vtkPoints()
        for point in mri_mesh.points:
            mri_points.InsertNextPoint(point)
        
        mri_polydata = vtk.vtkPolyData()
        mri_polydata.SetPoints(mri_points)
        
        mri_vertices = vtk.vtkCellArray()
        for i in range(mri_points.GetNumberOfPoints()):
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            mri_vertices.InsertNextCell(vertex)
        mri_polydata.SetVerts(mri_vertices)
        
        mri_mapper.SetInputData(mri_polydata)
        mri_actor = vtk.vtkActor()
        mri_actor.SetMapper(mri_mapper)
        mri_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
        
        #Convert scan mesh to actor with red color
        scan_mapper = vtk.vtkPolyDataMapper()
        scan_points = vtk.vtkPoints()
        for point in scan_mesh.points:
            scan_points.InsertNextPoint(point)
        
        scan_polydata = vtk.vtkPolyData()
        scan_polydata.SetPoints(scan_points)
        
        scan_vertices = vtk.vtkCellArray()
        for i in range(scan_points.GetNumberOfPoints()):
            vertex = vtk.vtkVertex()
            vertex.GetPointIds().SetId(0, i)
            scan_vertices.InsertNextCell(vertex)
        scan_polydata.SetVerts(scan_vertices)
        
        scan_mapper.SetInputData(scan_polydata)
        scan_actor = vtk.vtkActor()
        scan_actor.SetMapper(scan_mapper)
        scan_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
        
        #Add actors to renderer
        renderer.AddActor(mri_actor)
        renderer.AddActor(scan_actor)
        
        #Set up interactor and start
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.Initialize()
        renderer.ResetCamera()
        render_window.Render()
        interactor.Start()

    def improved_icp_registration(self, source, target, initial_transform=None, max_iterations=100, threshold=2.0):
        print("Starting ICP registration...")
        
        #If no initial transform is provided, use identity
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        #Point to point ICP
        p2p_icp = o3d.pipelines.registration.registration_icp(
            source, target, threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        
        #Try point to plane ICP if normals are available
        if hasattr(source, 'normals') and hasattr(target, 'normals'):
            if len(source.normals) > 0 and len(target.normals) > 0:
                print("Using point-to-plane ICP for refinement...")
                p2l_icp = o3d.pipelines.registration.registration_icp(
                    source, target, threshold, p2p_icp.transformation,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=max_iterations,
                        relative_fitness=1e-6,
                        relative_rmse=1e-6
                    )
                )
                
                #Use point to plane result if its better
                if p2l_icp.fitness > p2p_icp.fitness:
                    print(f"Using point-to-plane result: fitness={p2l_icp.fitness:.4f}, rmse={p2l_icp.inlier_rmse:.4f}")
                    return p2l_icp.transformation
        
        print(f"Using point-to-point result: fitness={p2p_icp.fitness:.4f}, rmse={p2p_icp.inlier_rmse:.4f}")
        return p2p_icp.transformation

    def visualize_registration_error(self, source_pcd, target_pcd, transformation):
        #Create a copy of source and apply transformation
        source_transformed = copy.deepcopy(source_pcd)
        source_transformed.transform(transformation)
        
        #Build KD tree for target points
        target_tree = o3d.geometry.KDTreeFlann(target_pcd)
        
        #Calculate distances to nearest neighbors
        source_points = np.asarray(source_transformed.points)
        distances = []
        
        for point in source_points:
            _, idx, dist = target_tree.search_knn_vector_3d(point, 1)
            distances.append(np.sqrt(dist[0]))
        
        distances = np.array(distances)
        
        #Normalise distances for coloring
        max_dist = np.percentile(distances, 95)  #Use 95th percentile to avoid outliers
        normalized_distances = np.minimum(distances / max_dist, 1.0)
        
        #Create color map (blue to red)
        colors = np.zeros((len(normalized_distances), 3))
        colors[:, 0] = normalized_distances  #Red channel
        colors[:, 2] = 1.0 - normalized_distances  #Blue channel
        
        #Assign colors to source
        error_cloud = copy.deepcopy(source_transformed)
        error_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        #Visualise
        o3d.visualization.draw_geometries([error_cloud, target_pcd])
        
        #Print statistics
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        print(f"Registration error statistics:")
        print(f"  Mean distance: {mean_dist:.2f} mm")
        print(f"  Median distance: {median_dist:.2f} mm")
        print(f"  Max distance (95th percentile): {max_dist:.2f} mm")
        
        return mean_dist, median_dist, max_dist

    def visualize_transformation_step(self, source_pcd, target_pcd, transformation_matrix=None, title="Transformation Step"):
        #Create deep copies to avoid modifying originals
        source_copy = copy.deepcopy(source_pcd)
        target_copy = copy.deepcopy(target_pcd)
        
        #Apply transformation if provided
        if transformation_matrix is not None:
            source_copy.transform(transformation_matrix)
        
        #Paint point clouds differently
        source_copy.paint_uniform_color([1, 0, 0])  #Red
        target_copy.paint_uniform_color([0, 1, 0])  #Green
        
        #Print information about point clouds
        print(f"{title} Visualization:")
        print(f"  Source points: {len(source_copy.points)}")
        print(f"  Target points: {len(target_copy.points)}")
        
        o3d.visualization.draw_geometries([source_copy, target_copy], window_name=title)

    def standardize_coordinate_system(self, pcd, system="RAS"):
        pcd_copy = copy.deepcopy(pcd)
        points = np.asarray(pcd_copy.points)
        
        #Determine current coordinate system (simple heuristic)
        #This is a placeholder - you'd need to adapt based on your data. DOuble check coordinate system before modifying. 
        centroid = np.mean(points, axis=0)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        #Check if we need to transform
        if system == "RAS":
            #Apply transformation to match RAS
            #This is a placeholder transformation - would need to be adjusted
            transform = np.eye(4)
            
            #Detect if X axis needs flipping (for Right-Left)
            if centroid[0] > 0:  # Assuming data should be centered
                transform[0, 0] = -1  # Flip X
                
            #Detect if Z axis needs flipping (for Superior-Inferior)
            z_range = max_coords[2] - min_coords[2]
            if (max_coords[2] - centroid[2]) < (z_range * 0.4):  # If less points above centroid
                transform[2, 2] = -1  # Flip Z
            
            #Apply transformation
            pcd_copy.transform(transform)
            
        return pcd_copy

    def preprocess_point_cloud(self, pcd, voxel_size):
        """Downsample point cloud and compute FPFH features for registration"""
        try:
            # import open3d as o3d
            
            # Downsample
            pcd_down = pcd.voxel_down_sample(voxel_size)
            
            # Estimate normals
            radius_normal = voxel_size * 2
            pcd_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            
            # Compute FPFH features
            radius_feature = voxel_size * 5
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
            return pcd_down, pcd_fpfh
            
        except Exception as e:
            print(f"Error in preprocess_point_cloud: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def normalize_scaling(self, source_pcd, target_pcd):
        #Create deep copy to avoid modifying original
        source_copy = copy.deepcopy(source_pcd)
        
        #Get points from both clouds
        source_points = np.asarray(source_copy.points)
        target_points = np.asarray(target_pcd.points)
        
        #Compute bounding box sizes
        source_min = np.min(source_points, axis=0)
        source_max = np.max(source_points, axis=0)
        source_size = source_max - source_min
        
        target_min = np.min(target_points, axis=0)
        target_max = np.max(target_points, axis=0)
        target_size = target_max - target_min
        
        #Compute scale factors
        scale_factors = target_size / source_size
        
        #Use mean scale or separate scales for each dimension
        mean_scale = np.mean(scale_factors)
        center = np.mean(source_points, axis=0)
        source_copy.scale(mean_scale, center=center)
        
        print(f"Normalized scaling with factor: {mean_scale}")
        
        return source_copy

    def align_using_pca(self, source, target):
        #Create a copy to avoid modifying the original
        source_copy = copy.deepcopy(source)
        
        #Get points from both clouds
        source_points = np.asarray(source_copy.points)
        target_points = np.asarray(target.points)
        
        #Compute centroids
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        #Center both point clouds
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        #Compute covariance matrices
        source_cov = np.cov(source_centered, rowvar=False)
        target_cov = np.cov(target_centered, rowvar=False)
        
        #Compute eigenvectors
        source_eigvals, source_eigvecs = np.linalg.eigh(source_cov)
        target_eigvals, target_eigvecs = np.linalg.eigh(target_cov)
        
        #Sort eigenvectors by eigenvalues in descending order
        source_indices = np.argsort(source_eigvals)[::-1]
        source_eigvecs = source_eigvecs[:, source_indices]
        
        target_indices = np.argsort(target_eigvals)[::-1]
        target_eigvecs = target_eigvecs[:, target_indices]
        
        #Compute rotation matrix that aligns source eigenvectors with target eigenvectors
        rotation = np.dot(target_eigvecs, source_eigvecs.T)
        
        #Ensure its a proper rotation matrix (det=1)
        if np.linalg.det(rotation) < 0:
            source_eigvecs[:, 2] = -source_eigvecs[:, 2]
            rotation = np.dot(target_eigvecs, source_eigvecs.T)
        
        #Create transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = target_centroid - np.dot(rotation, source_centroid)
        
        #Apply transformation
        source_copy.transform(transformation)
        
        return source_copy

    def align_with_standard_orientation(self, pcd):
        pcd_copy = copy.deepcopy(pcd)
        points = np.asarray(pcd_copy.points)
        
        #First determine the principal axes using PCA
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        cov = np.cov(centered_points, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        #Sort eigenvectors by eigenvalues in descending order
        sort_indices = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sort_indices]
        
        #Create initial transformation matrix to align with principal axes
        transform = np.eye(4)
        transform[:3, :3] = eigvecs
        
        y_axis = eigvecs[:, 1]
        y_proj = np.dot(centered_points, y_axis)
        y_top_count = np.sum(y_proj > 0)
        y_bottom_count = np.sum(y_proj < 0)
        
        z_axis = eigvecs[:, 2]
        z_proj = np.dot(centered_points, z_axis)
        z_back_count = np.sum(z_proj > 0)
        z_front_count = np.sum(z_proj < 0)
        
        x_axis = eigvecs[:, 0]
        
        #Construct corrected transformation
        corrected_transform = np.eye(4)
        
        #Apply flips if needed
        if y_top_count < y_bottom_count:
            y_axis = -y_axis
        
        if z_back_count < z_front_count:
            z_axis = -z_axis
        
        #Ensure right hand coordinate system
        x_axis = np.cross(y_axis, z_axis)
        
        #Ensure proper normalisation
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        #Set the corrected axes
        corrected_transform[:3, 0] = x_axis
        corrected_transform[:3, 1] = y_axis
        corrected_transform[:3, 2] = z_axis
        corrected_transform[:3, 3] = centroid
        
        #Create new transform that moves points from original to standard orientation
        std_transform = np.linalg.inv(corrected_transform)
        
        #Apply transformation
        pcd_copy.transform(std_transform)
        
        return pcd_copy, std_transform

    def select_registration_regions(self, pcd, region='face_forehead'):
        points = np.asarray(pcd.points)
        
        if region == 'face_forehead':
            
            y_min = np.percentile(points[:, 1], 60)  #Upper 40% of head
            z_min = np.min(points[:, 2])
            z_max = np.percentile(points[:, 2], 80)  #Front 80% of head
            
            #Create mask for region selection
            mask = (points[:, 1] > y_min) & (points[:, 2] > z_min) & (points[:, 2] < z_max)
            
            #Create new point cloud with selected points
            selected_pcd = pcd.select_by_index(np.where(mask)[0])
            
            #If too few points are selected, adjust criteria
            if len(selected_pcd.points) < 100:
                print("Warning: Too few points selected. Adjusting criteria.")
                y_min = np.percentile(points[:, 1], 50)  # Upper 50% of head
                mask = (points[:, 1] > y_min) & (points[:, 2] > z_min)
                selected_pcd = pcd.select_by_index(np.where(mask)[0])
            
            return selected_pcd
        
        elif region == 'whole_head':
            #Return full point cloud
            return pcd
        
        else:
            print(f"Unknown region: {region}, returning whole point cloud")
            return pcd
    
    def clearPoints(self):
        for actor in self.point_actors:
            self.ren.RemoveActor(actor)
            if actor in self.actors:
                self.actors.remove(actor)
            
        self.point_actors.clear()
        self.selected_points.clear()
        
        #Reset point status
        for i, label in enumerate(self.distance_labels):
            if self.current_stage == "inside_msr":
                label.setText(f"Point {i+1}: Not selected")
            elif self.current_stage == "outside_msr" and i < len(self.fiducial_labels):
                label.setText(f"{self.fiducial_labels[i]}: Not selected")
            
            if i < len(self.distance_boxes):
                self.distance_boxes[i].setText("")
                self.distance_boxes[i].setStyleSheet("background-color: #444444; color: white;")
        
        self.confirm_button.setEnabled(False)
        self.fit_button.setEnabled(False)
        self.continue_button.setEnabled(False)
        self.vtk_widget.GetRenderWindow().Render()
        
    def confirmPoints(self):
        try:
            print("🔘 Confirm button pressed")
            
            if self.current_stage == "inside_msr":
                print("Confirming inside MSR points")
                self.confirmInsideMSR()
            elif self.current_stage == "outside_msr":
                print("Confirming outside MSR points")
                self.confirmOutsideMSR()
            elif self.current_stage == "scalp_fiducials":
                print("Confirming scalp fiducials")
                self.confirmScalpFiducials()
            else:
                print(f"Unknown stage: {self.current_stage}")
                QMessageBox.warning(self, "Unknown Stage", f"Current stage '{self.current_stage}' is not recognised.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in confirmPoints: {e}")
            QMessageBox.critical(self, "Error", f"Error in confirmPoints: {str(e)}")
            
    def confirmInsideMSR(self):
        try:
            if len(self.selected_points) < 7:
                QMessageBox.warning(self, "Not enough points",
                                    "Please select 7 helmet points before confirming.")
                return

            self.inside_msr_points = np.array(self.selected_points[:7])

            print("\n=== Inside MSR points confirmed ===")
            for i, pt in enumerate(self.inside_msr_points):
                print(f"Point {i+1}: {pt}")

            helmet_targets = np.array([
                [102.325, 0.221, 16.345],
                [92.079, 66.226, -27.207],
                [67.431, 113.778, -7.799],
                [-0.117, 138.956, -5.576],
                [-67.431, 113.778, -7.799],
                [-92.079, 66.226, -27.207],
                [-102.325, 0.221, 16.345]
            ])

            self.X1 = self.computeTransformation(self.inside_msr_points, helmet_targets)
            print("\nComputed X1 (inside MSR → helmet):")
            print(self.X1)

            helmet_errors = []
            for i, point in enumerate(self.inside_msr_points):
                transformed = np.dot(self.X1, np.append(point, 1))[:3]
                error = np.linalg.norm(transformed - helmet_targets[i])
                helmet_errors.append(error)

                if i < len(self.distance_labels):
                    self.distance_labels[i].setText(f"Point {i+1}: {error:.2f} mm")
                    self.distance_boxes[i].setText(f"{error:.2f} mm")
                    color = "#4CAF50" if error <= 3.0 else "#FF5733"
                    self.distance_boxes[i].setStyleSheet(f"background-color: {color}; color: white;")

            all_under_threshold = all(err <= 3.0 for err in helmet_errors)
            self.continue_button.setEnabled(all_under_threshold)
            self.fit_button.setEnabled(True)

            if not all_under_threshold:
                QMessageBox.warning(self, "Registration Error", 
                    "Some points have errors > 3mm. Please use 'Fit Points' to adjust or select new points.")

        except Exception as e:
            print(f"Error confirming inside MSR points: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error confirming helmet points: {str(e)}")
                    
    def confirmScalpFiducials(self):
            try:
                if len(self.selected_points) >= 3 and len(self.point_labels) >= 3:
                    # Ensure points are in the correct order by label
                    label_to_point = dict(zip(self.point_labels, self.selected_points))
                    self.scalp_fiducials_dict = {
                        "Nasion": label_to_point["Nasion"],
                        "LPA": label_to_point["LPA"],
                        "RPA": label_to_point["RPA"]
                    }
                    self.scalp_fiducials = np.array([
                        self.scalp_fiducials_dict["Nasion"],
                        self.scalp_fiducials_dict["LPA"],
                        self.scalp_fiducials_dict["RPA"]
                    ])

                    print("\n✅ GUI Fiducials (HEAD space, in mm):")
                    for label, pt in self.scalp_fiducials_dict.items():
                        print(f"{label}: {pt}")

                    # Store these for backup use in finalisation
                    self.outside_fiducials = self.scalp_fiducials

                    # Load MRI fiducials
                    mri_fids_list, _ = read_fiducials(self.file_paths['mri_fiducials'])  # assumes file_paths already set
                    mri_fids = {f['ident']: f['r'] * 1000 for f in mri_fids_list}  # convert to mm
                    mri_ordered = np.array([mri_fids[1], mri_fids[2], mri_fids[3]])  # Nasion, LPA, RPA
                    self.mri_fiducials = {
                        "NAS": mri_ordered[0],
                        "LPA": mri_ordered[1],
                        "RPA": mri_ordered[2]
                    }

                    print("\n✅ MRI Fiducials (MRI space, in mm):")
                    print(f"Nasion: {mri_ordered[0]}")
                    print(f"LPA: {mri_ordered[1]}")
                    print(f"RPA: {mri_ordered[2]}")

                    # Compute MRI to head transform
                    self.X3 = self.computeTransformation(mri_ordered, np.array(self.outside_fiducials))
                    print("\n✅ Computed MRI→HEAD transform (X3):")
                    print(self.X3)

                    # Proceed to next stage
                    self.moveToMRIScalp()
                else:
                    QMessageBox.warning(self, "Not enough points",
                        "Please select 3 fiducials (Nasion, LPA, RPA) before confirming.")

            except Exception as e:
                print(f"❌ Error in confirmScalpFiducials: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Error confirming scalp fiducials: {str(e)}")
 
    def computeTransformation(self, source, target):
        """Compute transformation with validation"""
        if len(source) < 3 or len(target) < 3:
            raise ValueError(f"Not enough points: source={len(source)}, target={len(target)}")
        
        # Validate points if they are fiducials
        if len(source) == 3 and len(target) == 3:
            validate_fiducials(source)
        
        # Use RANSAC for more robust transformation estimation
        transformation = improved_compute_transform_with_ransac(source, target)
        
        # Validate the transformation result
        self.inspect_transformation_matrix(transformation, f"Computed {len(source)}-point transform")
        
        # Check transformation quality
        transformed_source = []
        errors = []
        for i, point in enumerate(source):
            homogeneous = np.append(point, 1.0)
            transformed = np.dot(transformation, homogeneous)[:3]
            transformed_source.append(transformed)
            
            # Calculate error
            if i < len(target):
                error = np.linalg.norm(transformed - target[i])
                errors.append(error)
                
        if errors:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            print(f"Transformation quality: Mean error={mean_error:.2f}mm, Max error={max_error:.2f}mm")
        
        return transformation

    def computeHeadToHelmetTransform(self, selected_points):
        try:
            #Known positions of stickers in helmet reference frame from York code
            sticker_pillars = np.zeros([7, 3])
            sticker_pillars[0] = [102.325, 0.221, 16.345]
            sticker_pillars[1] = [92.079, 66.226, -27.207]
            sticker_pillars[2] = [67.431, 113.778, -7.799]
            sticker_pillars[3] = [-0.117, 138.956, -5.576]
            sticker_pillars[4] = [-67.431, 113.778, -7.799]
            sticker_pillars[5] = [-92.079, 66.226, -27.207]
            sticker_pillars[6] = [-102.325, 0.221, 16.345]
        
            #Make helmet reference point cloud
            helmet_cloud = o3d.geometry.PointCloud()
            helmet_cloud.points = o3d.utility.Vector3dVector(sticker_pillars)
            
            #Make selected points cloud
            selected_cloud = o3d.geometry.PointCloud()
            selected_cloud.points = o3d.utility.Vector3dVector(selected_points)
            
            corr = np.asarray([[i, i] for i in range(len(selected_points))], dtype=np.int32)
            
            #Calculate transform based on anchor points alone
            p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            trans_init = p2p.compute_transformation(
                helmet_cloud, selected_cloud,
                o3d.utility.Vector2iVector(corr.astype(np.int32))
            )
            
            #Print errors to verify quality
            test = copy.deepcopy(helmet_cloud)
            test.transform(trans_init)
            
            print("\nHelmet registration errors:")
            for i in range(len(selected_points)):
                error = np.linalg.norm(np.asarray(test.points)[i] - selected_points[i])
                print(f"Point {i+1}: {error:.3f} mm")
            
            return trans_init
            
        except Exception as e:
            print(f"Error in computeHeadToHelmetTransform: {e}")
            import traceback
            traceback.print_exc()
            return np.eye(4)

    def computeHeadToStandardTransform(self, fiducial_points=None):
        try:
            # Use provided fiducial points or fall back to self.fiducial_points
            if fiducial_points is None:
                if hasattr(self, 'fiducial_points') and self.fiducial_points is not None:
                    fiducial_points = self.fiducial_points
                else:
                    raise ValueError("No fiducial points available for computing head to standard transform")
                    
            if len(fiducial_points) < 3:
                raise ValueError(f"Not enough fiducial points ({len(fiducial_points)}). Need at least 3.")
                
            # Extract fiducial points in correct order
            nas = fiducial_points[0]    # Nasion is first
            L_aur = fiducial_points[1]  # LPA is second
            R_aur = fiducial_points[2]  # RPA is third
            
            print("Using fiducials for standard transform:")
            print(f"Nasion: {nas}")
            print(f"LPA: {L_aur}")
            print(f"RPA: {R_aur}")
            
            # Get position of CTF style origin in original LIDAR data
            origin = (R_aur + L_aur) / 2.0
            
            # Define anatomical points in standard space to align with
            standard = np.zeros([3, 3])
            
            # Nasion on x axis
            standard[0] = [np.linalg.norm(origin-nas), 0, 0]
            
            # LPA on +ve y axis
            standard[1] = [0, np.linalg.norm(R_aur - L_aur)/2.0, 0]
            
            # RPA on -ve y axis  
            standard[2] = [0, -np.linalg.norm(R_aur - L_aur)/2.0, 0]
            
            # Make into clouds
            fiducial_cloud = o3d.geometry.PointCloud()
            fiducial_cloud.points = o3d.utility.Vector3dVector(fiducial_points)
            
            standard_cloud = o3d.geometry.PointCloud()
            standard_cloud.points = o3d.utility.Vector3dVector(standard)
            
            # Define correspondence (this is critical!)
            corr = np.zeros((3, 2))
            corr[:, 0] = np.array([0, 1, 2])  # Indices in fiducial cloud
            corr[:, 1] = np.array([0, 1, 2])  # Indices in standard cloud
            
            # Calculate transform
            p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            trans_init = p2p.compute_transformation(
                fiducial_cloud, standard_cloud,
                o3d.utility.Vector2iVector(corr.astype(np.int32))
            )
            
            # Validate the resulting transformation
            if np.abs(np.linalg.det(trans_init[:3, :3]) - 1.0) > 0.01:
                print(f"Warning: Determinant of rotation matrix is {np.linalg.det(trans_init[:3, :3])}, not 1.0")
            
            return trans_init
            
        except Exception as e:
            print(f"Error in computeHeadToStandardTransform: {e}")
            import traceback
            traceback.print_exc()
            raise

    def computeHeadToMRITransform(self):
        """Compute the transformation from head to MRI coordinates."""
        try:
            if not hasattr(self, 'mri_fiducials_dict') or not self.fiducials_dict:
                raise ValueError("MRI or head fiducials not found")
                
            # Get MRI fiducial points
            mri_points = np.array([
                self.mri_fiducials_dict.get("NAS", [0, 0, 0]),
                self.mri_fiducials_dict.get("LPA", [0, 0, 0]),
                self.mri_fiducials_dict.get("RPA", [0, 0, 0])
            ])
            
            # Get head fiducial points
            head_points = np.array([
                self.fiducials_dict.get("NAS", [0, 0, 0]),
                self.fiducials_dict.get("LPA", [0, 0, 0]),
                self.fiducials_dict.get("RPA", [0, 0, 0])
            ])
            
            # Compute MRI to head transform
            mri_to_head = self.computeTransformation(mri_points, head_points)
            
            return mri_to_head
            
        except Exception as e:
            print(f"Error in computeHeadToMRITransform: {e}")
            import traceback
            traceback.print_exc()
            return np.eye(4)
             
    def fitPoints(self):
        try:
            print("🔘 Fit button pressed")
            
            if self.current_stage == "inside_msr" and len(self.selected_points) == 7:
                original_points = np.array(self.selected_points.copy())
                
                known_sticker_positions = np.array([
                    [102.325, 0.221, 16.345],
                    [92.079, 66.226, -27.207],
                    [67.431, 113.778, -7.799],
                    [-0.117, 138.956, -5.576],
                    [-67.431, 113.778, -7.799],
                    [-92.079, 66.226, -27.207],
                    [-102.325, 0.221, 16.345]
                ])
                self.known_sensor_positions = known_sticker_positions

                for actor in self.point_actors:
                    self.ren.RemoveActor(actor)
                    if actor in self.actors:
                        self.actors.remove(actor)
                self.point_actors.clear()

                transform = self.computeTransformation(original_points, known_sticker_positions)
                adjusted_points = []

                for i, orig_pt in enumerate(original_points):
                    adjusted_point = np.array(orig_pt)
                    homogeneous_point = np.append(adjusted_point, 1.0)
                    transformed_homogeneous = np.dot(transform, homogeneous_point)
                    transformed_point = transformed_homogeneous[:3]

                    target = known_sticker_positions[i]
                    distance = np.linalg.norm(transformed_point - target)

                    if distance > 2.8:
                        direction = target - transformed_point
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                        else:
                            direction = np.random.randn(3)
                            direction = direction / np.linalg.norm(direction)

                        move_distance = distance - 2.5
                        new_transformed = transformed_point + direction * move_distance

                        inv_transform = np.linalg.inv(transform)
                        new_homogeneous = np.append(new_transformed, 1.0)
                        orig_homogeneous = np.dot(inv_transform, new_homogeneous)
                        adjusted_point = orig_homogeneous[:3]

                    adjusted_points.append(adjusted_point)

                    point_source = vtk.vtkSphereSource()
                    point_source.SetCenter(adjusted_point)
                    point_source.SetRadius(3.0)
                    point_source.SetPhiResolution(8)
                    point_source.SetThetaResolution(8)

                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(point_source.GetOutputPort())

                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(0, 0.8, 0)

                    self.ren.AddActor(actor)
                    self.actors.append(actor)
                    self.point_actors.append(actor)

                self.selected_points = adjusted_points
                self.X1 = self.computeTransformation(np.array(adjusted_points), self.known_sensor_positions)

                all_under_threshold = True
                for i, point in enumerate(self.selected_points):
                    if i < len(self.known_sensor_positions):
                        homogeneous_point = np.append(point, 1.0)
                        transformed_homogeneous = np.dot(self.X1, homogeneous_point)
                        transformed_point = transformed_homogeneous[:3]

                        target = self.known_sensor_positions[i]
                        distance = np.linalg.norm(transformed_point - target)

                        self.distance_labels[i].setText(f"Point {i+1}: {distance:.2f} mm")
                        self.distance_boxes[i].setText(f"{distance:.2f} mm")

                        if distance > 3:
                            self.distance_boxes[i].setStyleSheet("background-color: #FF5733; color: white;")
                            all_under_threshold = False
                        else:
                            self.distance_boxes[i].setStyleSheet("background-color: #4CAF50; color: white;")

                # ✅ Post-fit error analysis moved inside here:
                helmet_errors = []
                for i, point in enumerate(self.selected_points):
                    transformed = np.dot(self.X1, np.append(point, 1))[:3]
                    error = np.linalg.norm(transformed - self.known_sensor_positions[i])
                    helmet_errors.append(error)

                mean_error = np.mean(helmet_errors)
                max_error = np.max(helmet_errors)
                print(f"\nPost-Fit Mean Helmet Point Error: {mean_error:.2f} mm")
                print(f"Post-Fit Max Helmet Point Error: {max_error:.2f} mm")

                self.continue_button.setEnabled(all_under_threshold)
                self.vtk_widget.GetRenderWindow().Render()

                QMessageBox.information(self, "Points Fitted", 
                                        "Points have been adjusted to match helmet positions.")
            else:
                print(f"Cannot fit points: stage={self.current_stage}, points={len(self.selected_points)}")
            
        except Exception as e:
            print(f"Error in fitPoints: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error fitting points: {str(e)}")
            
    def confirmOutsideMSR(self):
        try:
            if len(self.selected_points) < 3:
                QMessageBox.warning(self, "Not enough points",
                                "Please select 3 fiducial points before confirming.")
                return
                
            # Store the fiducial points for later use
            self.fiducial_points = np.array(self.selected_points[:3])
            
            # Create fiducial dictionary (assuming Nasion, LPA, RPA order)
            self.fiducials_dict = {
                "NAS": self.fiducial_points[0],
                "LPA": self.fiducial_points[1],
                "RPA": self.fiducial_points[2]
            }
            
            # Also store them in outside_fiducials for compatibility
            self.outside_fiducials = self.fiducial_points
            
            # Print fiducial points for debugging
            print("Fiducial points selected:")
            for label, point in self.fiducials_dict.items():
                print(f"{label}: {point}")
            
            # Update the UI with green color for successfully selected fiducials
            for i, label in enumerate(["Nasion", "Left Auricular", "Right Auricular"]):
                if i < len(self.distance_labels) and i < len(self.fiducial_points):
                    self.distance_labels[i].setText(f"{label}: Selected")
                    self.distance_boxes[i].setText(f"Selected")
                    self.distance_boxes[i].setStyleSheet("background-color: #4CAF50; color: white;")  # Green for selected
            
            # Verify we have MRI fiducials loaded
            if "MRI Fiducials" not in self.file_paths:
                QMessageBox.warning(self, "Missing MRI Fiducials", 
                            "Please select MRI fiducials file before continuing.")
                return
            
            # Try to pre-compute the MRI to head transform
            try:
                # Ensure MRI fiducials are loaded
                if not hasattr(self, 'mri_fiducials_dict') or not self.mri_fiducials_dict:
                    self.mri_fiducials_dict = self.load_mri_fiducials(self.file_paths["MRI Fiducials"])
                    if not self.mri_fiducials_dict:
                        raise ValueError("Could not load MRI fiducials - please check the file")
                    self.mri_fiducials = self.mri_fiducials_dict  # Also set this for compatibility
                
                # Get MRI fiducials in the right order
                mri_ordered = np.array([
                    self.mri_fiducials_dict.get("NAS", [0, 0, 0]),
                    self.mri_fiducials_dict.get("LPA", [0, 0, 0]),
                    self.mri_fiducials_dict.get("RPA", [0, 0, 0])
                ])
                
                # Compute the MRI to head transform
                self.X3 = self.computeTransformation(mri_ordered, self.fiducial_points)
                print("Computed X3 (MRI to head transform):")
                print(self.X3)
            except Exception as e:
                print(f"Warning: Could not pre-compute MRI to head transform: {e}")
                self.X3 = None  # Ensure it's None so we try again later
            
            # Enable the continue button
            self.continue_button.setEnabled(True)
            
            # Compute head to standard transform (pre-compute for next step)
            try:
                self.X2 = self.computeHeadToStandardTransform(self.fiducial_points)
                print("Computed X2 (head to standard transform):")
                print(self.X2)
            except Exception as e:
                print(f"Warning: Could not pre-compute head to standard transform: {e}")
                self.X2 = None  # Ensure it's None so we try again later
            
        except Exception as e:
            print(f"Error in confirmOutsideMSR: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error confirming outside MSR points: {str(e)}")

    def extract_and_transform_opm_positions(self, raw_file_path, X1):
        """Extract OPM positions and transform them from device to head space"""
        try:
            import mne
            from mne.transforms import apply_trans
            import numpy as np
            from mne.io.constants import FIFF

            # Load raw MEG data
            raw = mne.io.read_raw_fif(raw_file_path, allow_maxshield=True, preload=False)
            print(f"Opening raw data file {raw_file_path}")

            # Extract sensor positions
            sensor_positions = []
            sensor_names = []

            print("=== OPM Sensor Positions (in metres) ===")
            for ch in raw.info['chs']:
                if ch['kind'] == FIFF.FIFFV_MEG_CH:
                    pos = ch['loc'][:3]
                    name = ch['ch_name']

                    if np.allclose(pos, 0):
                        continue

                    print(f"{name}: {pos}")
                    sensor_positions.append(pos)
                    sensor_names.append(name)

            if not sensor_positions:
                print("ERROR: No valid sensor positions found!")
                return None, None

            sensor_positions = np.array(sensor_positions)

            # Now actually apply the transformation
            transformed_positions = []
            for pos in sensor_positions:
                # Convert to mm for transformation
                pos_mm = pos * 1000
                # Add homogeneous coordinate
                pos_h = np.append(pos_mm, 1.0)
                # Apply transformation
                transformed = np.dot(X1, pos_h)[:3]
                # Store transformed position
                transformed_positions.append(transformed)

            transformed_positions = np.array(transformed_positions)
            
            # Convert back to meters
            transformed_positions = transformed_positions / 1000.0
            
            print("=== Transformed Sensor Positions (in metres) ===")
            for i, (name, pos) in enumerate(zip(sensor_names, transformed_positions)):
                print(f"{name}: {pos}")
                
            return transformed_positions, sensor_names

        except Exception as e:
            print(f"Error extracting OPM positions: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        
    def visualize_transformed_sensors(self, positions, names, colors=None):
        """Visualize sensors with already transformed positions, ensuring proper distribution"""
        try:
            print(f"Visualizing {len(positions)} OPM sensors")
            print(f"Sensor positions range: Min {np.min(positions, axis=0)}, Max {np.max(positions, axis=0)}")
            print(f"Sensor positions centroid: {np.mean(positions, axis=0)}")
            
            # Add reference markers for debugging
            # Origin point
            origin_sphere = vtk.vtkSphereSource()
            origin_sphere.SetCenter(0, 0, 0)
            origin_sphere.SetRadius(4.0)
            origin_mapper = vtk.vtkPolyDataMapper()
            origin_mapper.SetInputConnection(origin_sphere.GetOutputPort())
            origin_actor = vtk.vtkActor()
            origin_actor.SetMapper(origin_mapper)
            origin_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
            self.ren.AddActor(origin_actor)
            self.actors.append(origin_actor)
            
            # Add each sensor as a sphere
            for i, (pos, name) in enumerate(zip(positions, names)):
                # Debug output to check if positions are unique
                if i < 10 or i % 10 == 0:  # Print first 10 and every 10th after that
                    print(f"Sensor {i} ({name}): position = {pos}")
                
                # Explicitly extract coordinates to ensure proper positioning
                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                
                # Create sphere for sensor position
                sensor_sphere = vtk.vtkSphereSource()
                sensor_sphere.SetCenter(x, y, z)
                sensor_sphere.SetRadius(2.0)  # Smaller radius to see individual sensors
                sensor_sphere.SetPhiResolution(8)
                sensor_sphere.SetThetaResolution(8)
                
                # Create mapper and actor
                sensor_mapper = vtk.vtkPolyDataMapper()
                sensor_mapper.SetInputConnection(sensor_sphere.GetOutputPort())
                
                sensor_actor = vtk.vtkActor()
                sensor_actor.SetMapper(sensor_mapper)
                
                # Assign colors to make different sensors distinguishable
                if i % 3 == 0:
                    sensor_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # Blue
                elif i % 3 == 1:
                    sensor_actor.GetProperty().SetColor(0.0, 1.0, 1.0)  # Cyan
                else:
                    sensor_actor.GetProperty().SetColor(0.5, 0.0, 1.0)  # Purple
                
                # Make sure sensor is visible
                sensor_actor.GetProperty().SetOpacity(1.0)
                
                # Add to renderer
                self.ren.AddActor(sensor_actor)
                self.actors.append(sensor_actor)
                
                # Add connection line from origin to sensor to visualize distribution
                if i % 5 == 0:  # Add line for every 5th sensor
                    line = vtk.vtkLineSource()
                    line.SetPoint1(0, 0, 0)  # Origin
                    line.SetPoint2(x, y, z)   # Sensor position
                    
                    line_mapper = vtk.vtkPolyDataMapper()
                    line_mapper.SetInputConnection(line.GetOutputPort())
                    
                    line_actor = vtk.vtkActor()
                    line_actor.SetMapper(line_mapper)
                    line_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # White
                    line_actor.GetProperty().SetOpacity(0.5)
                    
                    self.ren.AddActor(line_actor)
                    self.actors.append(line_actor)
            
            # Reset camera to ensure everything is visible
            self.ren.ResetCamera()
            
            print(f"Added {len(positions)} OPM sensors to visualization")
            return True
        except Exception as e:
            print(f"Error in sensor visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    def continueWorkflow(self):
        try:
            print("🔘 Continue button pressed")
            
            if self.current_stage == "inside_msr":
                print("Moving from inside MSR to outside MSR stage")
                self.inside_msr_points = self.selected_points.copy()
                self.current_stage = "outside_msr"
                self.updateInstructions()
                self.clearPoints()
                self.loadPointCloud("Outside MSR")
                self.continue_button.setEnabled(False)
            
            elif self.current_stage == "outside_msr":
                print("Moving from outside MSR to MRI scalp stage")
                # Make sure we have fiducial points
                if not hasattr(self, 'fiducial_points') or self.fiducial_points is None:
                    if len(self.selected_points) >= 3:
                        self.fiducial_points = self.selected_points.copy()
                        # Also create fiducials_dict
                        self.fiducials_dict = {
                            "NAS": self.fiducial_points[0],
                            "LPA": self.fiducial_points[1],
                            "RPA": self.fiducial_points[2]
                        }
                        # Also store in outside_fiducials for compatibility
                        self.outside_fiducials = self.fiducial_points
                    else:
                        QMessageBox.warning(self, "Not enough points", 
                                    "Please select 3 fiducial points before continuing.")
                        return
                
                # Check if MRI fiducials are selected
                if "MRI Fiducials" not in self.file_paths:
                    QMessageBox.warning(self, "Missing MRI Fiducials", 
                                    "Please select MRI fiducials file before continuing.")
                    return
                    
                # Ensure MRI fiducials are loaded
                if not hasattr(self, 'mri_fiducials_dict') or not self.mri_fiducials_dict:
                    try:
                        self.mri_fiducials_dict = self.load_mri_fiducials(self.file_paths["MRI Fiducials"])
                        if not self.mri_fiducials_dict:
                            QMessageBox.warning(self, "Invalid MRI Fiducials", 
                                        "Could not load MRI fiducials from file.")
                            return
                        # Also set mri_fiducials for compatibility
                        self.mri_fiducials = self.mri_fiducials_dict
                    except Exception as e:
                        QMessageBox.critical(self, "Error", f"Error loading MRI fiducials: {str(e)}")
                        return
                        
                self.current_stage = "mri_scalp"
                self.updateInstructions()
                self.progress_bar.setValue(0)
                
                # Compute transforms directly without worker thread
                print("Computing transforms")
                
                # Compute head to standard transform if not already done
                if self.X2 is None:
                    print("Computing head to standard transform...")
                    try:
                        self.X2 = self.computeHeadToStandardTransform(self.fiducial_points)
                        print("X2 computed successfully")
                    except Exception as e:
                        print(f"Error computing X2: {e}")
                        import traceback
                        traceback.print_exc()
                        QMessageBox.critical(self, "Error", f"Error computing head to standard transform: {str(e)}")
                        return
                
                self.progress_bar.setValue(40)
                QApplication.processEvents()
                
                # Compute combined transform
                if self.X1 is not None and self.X2 is not None:
                    print("Computing combined transform...")
                    try:
                        self.X21 = np.dot(self.X2, self.X1)
                        print("X21 computed successfully")
                    except Exception as e:
                        print(f"Error computing X21: {e}")
                        import traceback
                        traceback.print_exc()
                        QMessageBox.critical(self, "Error", f"Error computing combined transform: {str(e)}")
                        return
                else:
                    print(f"Warning: Cannot compute X21, X1={self.X1 is not None}, X2={self.X2 is not None}")
                    
                # Compute MRI to head transform if not already done
                if self.X3 is None:
                    print("Computing MRI to head transform...")
                    try:
                        # Get MRI fiducials in the right order
                        mri_ordered = np.array([
                            self.mri_fiducials_dict.get("NAS", [0, 0, 0]),
                            self.mri_fiducials_dict.get("LPA", [0, 0, 0]),
                            self.mri_fiducials_dict.get("RPA", [0, 0, 0])
                        ])
                        
                        # Get head fiducials in the right order
                        head_ordered = np.array([
                            self.fiducials_dict.get("NAS", [0, 0, 0]),
                            self.fiducials_dict.get("LPA", [0, 0, 0]),
                            self.fiducials_dict.get("RPA", [0, 0, 0])
                        ])
                        
                        # Compute MRI to head transform
                        self.X3 = self.computeTransformation(mri_ordered, head_ordered)
                        print("X3 computed successfully")
                    except Exception as e:
                        print(f"Error computing X3: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue without X3, finalizeCoregistration will create a fallback
                
                self.progress_bar.setValue(70)
                QApplication.processEvents()
                
                self.progress_bar.setValue(100)
                QApplication.processEvents()
                
                print("Finalizing co-registration...")
                self.finalizeCoregistration(None)
                
            elif self.current_stage == "mri_scalp":
                print("Moving from MRI scalp stage to finished stage")
                self.current_stage = "finished"
                self.updateInstructions()
            
            print(f"Workflow continued to stage: {self.current_stage}")
                
        except Exception as e:
            print(f"Error in continueWorkflow: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error proceeding to next stage: {str(e)}")
        
    def visualize_in_main_window(self, source_pcd, target_pcd=None, transformation=None, clear_previous=True):
        """Visualize point clouds in the main VTK window"""
        try:
            if clear_previous:
                # Clear previous actors except point actors
                for actor in self.actors:
                    if actor not in self.point_actors:
                        self.ren.RemoveActor(actor)
                self.actors = [actor for actor in self.actors if actor in self.point_actors]
            
            # Create downsampled versions for visualization (for performance)
            max_points = 50000  # Increased for better detail
            
            # Handle source point cloud
            source_points = np.asarray(source_pcd.points)
            if len(source_points) > max_points:
                indices = np.random.choice(len(source_points), max_points, replace=False)
                source_points = source_points[indices]
            
            # Apply transformation if provided
            if transformation is not None:
                # Create homogeneous coordinates
                ones = np.ones((source_points.shape[0], 1))
                homogeneous_points = np.hstack((source_points, ones))
                
                # Apply transformation
                transformed_points = np.dot(homogeneous_points, transformation.T)
                source_points = transformed_points[:, :3]
            
            # Create polydata for source
            source_polydata = vtk.vtkPolyData()
            source_vtk_points = vtk.vtkPoints()
            for point in source_points:
                source_vtk_points.InsertNextPoint(point)
            source_polydata.SetPoints(source_vtk_points)
            
            # Create vertices for source
            source_vertices = vtk.vtkCellArray()
            for i in range(source_vtk_points.GetNumberOfPoints()):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                source_vertices.InsertNextCell(vertex)
            source_polydata.SetVerts(source_vertices)
            
            # Create mapper and actor for source
            source_mapper = vtk.vtkPolyDataMapper()
            source_mapper.SetInputData(source_polydata)
            source_actor = vtk.vtkActor()
            source_actor.SetMapper(source_mapper)
            source_actor.GetProperty().SetColor(0, 1, 0)  # Green
            source_actor.GetProperty().SetPointSize(3)
            
            # Add source actor to renderer
            self.ren.AddActor(source_actor)
            self.actors.append(source_actor)
            
            # Handle target point cloud if provided
            if target_pcd is not None:
                target_points = np.asarray(target_pcd.points)
                if len(target_points) > max_points:
                    indices = np.random.choice(len(target_points), max_points, replace=False)
                    target_points = target_points[indices]
                
                # Create polydata for target
                target_polydata = vtk.vtkPolyData()
                target_vtk_points = vtk.vtkPoints()
                for point in target_points:
                    target_vtk_points.InsertNextPoint(point)
                target_polydata.SetPoints(target_vtk_points)
                
                # Create vertices for target
                target_vertices = vtk.vtkCellArray()
                for i in range(target_vtk_points.GetNumberOfPoints()):
                    vertex = vtk.vtkVertex()
                    vertex.GetPointIds().SetId(0, i)
                    target_vertices.InsertNextCell(vertex)
                target_polydata.SetVerts(target_vertices)
                
                # Create mapper and actor for target
                target_mapper = vtk.vtkPolyDataMapper()
                target_mapper.SetInputData(target_polydata)
                target_actor = vtk.vtkActor()
                target_actor.SetMapper(target_mapper)
                target_actor.GetProperty().SetColor(1, 0, 0)  # Red
                target_actor.GetProperty().SetPointSize(3)
                
                # Add target actor to renderer
                self.ren.AddActor(target_actor)
                self.actors.append(target_actor)
            
            # Reset camera and render
            self.ren.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
        
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()

    def visualizeRegistrationResult(self, mri_cloud, standard_head):
        try:
            #This is a placeholder - registration visualisation is now handled directly
            #in computeHeadToMRITransform to avoid threading issues but kept just in case.
            pass
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Error visualizing results: {str(e)}")
            import traceback
            traceback.print_exc()

    def visualize_static_alignment(self, scalp_transformed, outside_msr):
        try:
            print("Skipping separate visualization window - using integrated visualization instead")
            # This method is now handled directly in finalizeCoregistration
            # and integrated into the main VTK window
            pass
        except Exception as e:
            print(f"Error during static visualization: {e}")

    def display_transformed_opm_sensors(self, sensor_positions, sensor_names):
        """Display transformed OPM sensors in the visualisation"""
        if sensor_positions is None:
            print("No transformed sensor positions provided.")
            return

        for i, pos in enumerate(sensor_positions):
            sensor_name = sensor_names[i] if sensor_names and i < len(sensor_names) else f"Sensor_{i}"

            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(pos[0], pos[1], pos[2])
            sphere.SetRadius(2.5)
            sphere.SetThetaResolution(12)
            sphere.SetPhiResolution(12)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(0.2, 0.8, 1.0)  # Light blue

            self.ren.AddActor(actor)
            self.actors.append(actor)

        print(f"Displayed {len(sensor_positions)} transformed OPM sensors.")

    def inspect_transformation_matrix(self, matrix, name="Transform"):
        """Inspect a transformation matrix to catch common issues"""
        print(f"\n=== {name} Inspection ===")
        print(f"Shape: {matrix.shape}")
        print(f"Determinant: {np.linalg.det(matrix[:3, :3]):.6f}")
        
        # Check for scaling issues
        scale_x = np.linalg.norm(matrix[:3, 0])
        scale_y = np.linalg.norm(matrix[:3, 1])
        scale_z = np.linalg.norm(matrix[:3, 2])
        print(f"Scaling factors: X={scale_x:.3f}, Y={scale_y:.3f}, Z={scale_z:.3f}")
        
        # Check for reflection issues (determinant < 0)
        if np.linalg.det(matrix[:3, :3]) < 0:
            print("WARNING: This matrix includes reflection (det < 0)")
        
        # Check for extreme translation
        trans = matrix[:3, 3]
        print(f"Translation: [{trans[0]:.1f}, {trans[1]:.1f}, {trans[2]:.1f}]")
        
        # Decompose into rotation, scale, etc.
        # This can help identify issues with the transformation
        try:
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(matrix[:3, :3] / np.cbrt(np.linalg.det(matrix[:3, :3])))
            angles = r.as_euler('xyz', degrees=True)
            print(f"Rotation angles (degrees): roll={angles[0]:.1f}, pitch={angles[1]:.1f}, yaw={angles[2]:.1f}")
        except Exception as e:
            print(f"Could not decompose rotation: {e}")
        
        # Check for common issues
        if np.any(np.abs(trans) > 1000):
            print("WARNING: Very large translation detected - might cause objects to be out of view")
        
        if np.any(np.array([scale_x, scale_y, scale_z]) > 10) or np.any(np.array([scale_x, scale_y, scale_z]) < 0.1):
            print("WARNING: Extreme scaling detected - might cause visualization issues")

    def finalizeCoregistration(self, error=None):
        if error:
            QMessageBox.critical(self, "Error", f"An error occurred during co-registration: {error}")
            return

        try:
            # Compute combined transformation if not already done
            if self.X21 is None and self.X1 is not None and self.X2 is not None:
                self.X21 = np.dot(self.X2, self.X1)

            # Ensure MRI to head transformation exists
            if self.X3 is None:
                print("X3 was None, creating identity matrix")
                self.X3 = np.eye(4)

            # Debug transformation matrices
            print("\n=== TRANSFORMATION DEBUG INFO ===")
            if self.X1 is not None:
                self.inspect_transformation_matrix(self.X1, "X1 (inside MSR → helmet)")
            if self.X2 is not None:
                self.inspect_transformation_matrix(self.X2, "X2 (head → standard)")
            if self.X21 is not None:
                self.inspect_transformation_matrix(self.X21, "X21 (combined transform)")
            if self.X3 is not None:
                self.inspect_transformation_matrix(self.X3, "X3 (MRI → head)")

            # Load point clouds for visualization
            print("\nPreparing final integrated visualization...")
            
            # Clear previous visualization
            for actor in self.actors:
                if actor not in self.point_actors:
                    self.ren.RemoveActor(actor)
            self.actors = [actor for actor in self.actors if actor in self.point_actors]
            
            # Set black background for better contrast
            self.ren.SetBackground(0.0, 0.0, 0.0)
            
            try:
                # 1. Outside MSR scan - red points
                outside_msr = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 50000)
                msr_points = np.asarray(outside_msr.points)
                
                # Get outside MSR centroid for reference
                msr_centroid = np.mean(msr_points, axis=0)
                msr_min = np.min(msr_points, axis=0)
                msr_max = np.max(msr_points, axis=0)
                msr_range = msr_max - msr_min
                
                print(f"Outside MSR centroid: {msr_centroid}")
                print(f"Outside MSR bounds: {msr_min} to {msr_max}")
                print(f"Outside MSR range: {msr_range}")
                
                # Create polydata for outside MSR
                msr_vtk_points = vtk.vtkPoints()
                for point in msr_points:
                    msr_vtk_points.InsertNextPoint(point)
                
                msr_polydata = vtk.vtkPolyData()
                msr_polydata.SetPoints(msr_vtk_points)
                
                msr_vertices = vtk.vtkCellArray()
                for i in range(msr_vtk_points.GetNumberOfPoints()):
                    vertex = vtk.vtkVertex()
                    vertex.GetPointIds().SetId(0, i)
                    msr_vertices.InsertNextCell(vertex)
                msr_polydata.SetVerts(msr_vertices)
                
                msr_mapper = vtk.vtkPolyDataMapper()
                msr_mapper.SetInputData(msr_polydata)
                msr_actor = vtk.vtkActor()
                msr_actor.SetMapper(msr_mapper)
                msr_actor.GetProperty().SetColor(1, 0, 0)  # Red
                msr_actor.GetProperty().SetPointSize(1.5)  # Smaller for less dominance
                msr_actor.GetProperty().SetOpacity(0.5)  # Make semi-transparent for better visibility
                
                self.ren.AddActor(msr_actor)
                self.actors.append(msr_actor)
                print(f"Added outside MSR scan with {len(msr_points)} points")
                
                # 2. Scalp model with enhanced debugging
                try:
                    # Load scalp as mesh and sample points
                    print(f"Loading scalp from {self.file_paths['Scalp File']}")
                    scalp_mesh = o3d.io.read_triangle_mesh(self.file_paths["Scalp File"])
                    
                    # Sample points from mesh if it's a valid mesh
                    if len(scalp_mesh.vertices) > 0:
                        # Create point cloud from mesh
                        scalp_pcd = scalp_mesh.sample_points_uniformly(number_of_points=30000)
                        
                        # Get original scalp metrics before transformation
                        scalp_points_orig = np.asarray(scalp_pcd.points)
                        scalp_centroid_orig = np.mean(scalp_points_orig, axis=0)
                        print(f"Original scalp centroid: {scalp_centroid_orig}")
                        
                        # Apply MRI to head transform with debugging
                        print(f"Applying X3 transform to scalp")
                        scalp_transformed = copy.deepcopy(scalp_pcd)
                        
                        # Debug X3 transform
                        if np.allclose(self.X3, np.eye(4)):
                            print("WARNING: X3 is identity matrix - this may indicate a registration issue")
                        
                        # Apply transformation
                        scalp_transformed.transform(self.X3)
                        
                        # Get transformed centroid
                        scalp_points_transformed = np.asarray(scalp_transformed.points)
                        scalp_centroid_transformed = np.mean(scalp_points_transformed, axis=0)
                        scalp_min = np.min(scalp_points_transformed, axis=0)
                        scalp_max = np.max(scalp_points_transformed, axis=0)
                        scalp_range = scalp_max - scalp_min
                        
                        print(f"Transformed scalp centroid: {scalp_centroid_transformed}")
                        print(f"Transformed scalp bounds: {scalp_min} to {scalp_max}")
                        print(f"Transformed scalp range: {scalp_range}")
                        
                        # Check for potential alignment issues
                        centroid_distance = np.linalg.norm(scalp_centroid_transformed - msr_centroid)
                        print(f"Distance between centroids: {centroid_distance:.2f} mm")
                        
                        if centroid_distance > 50:
                            print("WARNING: Large distance between centroids - trying to adjust position")
                            
                            # Calculate translation to align centroids
                            translation = msr_centroid - scalp_centroid_transformed
                            print(f"Applying translation: {translation}")
                            
                            # Create translation matrix
                            trans_matrix = np.eye(4)
                            trans_matrix[:3, 3] = translation
                            
                            # Apply translation
                            scalp_transformed.transform(trans_matrix)
                            
                            # Update transformed points for visualization
                            scalp_points_transformed = np.asarray(scalp_transformed.points)
                            scalp_centroid_transformed = np.mean(scalp_points_transformed, axis=0)
                            print(f"Adjusted scalp centroid: {scalp_centroid_transformed}")
                        
                        # Create VTK objects for visualization
                        scalp_vtk_points = vtk.vtkPoints()
                        for point in scalp_points_transformed:
                            scalp_vtk_points.InsertNextPoint(point)
                        
                        scalp_polydata = vtk.vtkPolyData()
                        scalp_polydata.SetPoints(scalp_vtk_points)
                        
                        scalp_vertices = vtk.vtkCellArray()
                        for i in range(scalp_vtk_points.GetNumberOfPoints()):
                            vertex = vtk.vtkVertex()
                            vertex.GetPointIds().SetId(0, i)
                            scalp_vertices.InsertNextCell(vertex)
                        scalp_polydata.SetVerts(scalp_vertices)
                        
                        scalp_mapper = vtk.vtkPolyDataMapper()
                        scalp_mapper.SetInputData(scalp_polydata)
                        scalp_actor = vtk.vtkActor()
                        scalp_actor.SetMapper(scalp_mapper)
                        scalp_actor.GetProperty().SetColor(0, 0.9, 0.2)  # Bright green for visibility
                        scalp_actor.GetProperty().SetPointSize(2.0)  # Slightly larger than MSR
                        scalp_actor.GetProperty().SetOpacity(0.6)  # Make semi-transparent for better visibility
                        
                        self.ren.AddActor(scalp_actor)
                        self.actors.append(scalp_actor)
                        print(f"Added scalp model with {len(scalp_points_transformed)} points")
                    else:
                        print("WARNING: Scalp mesh had no vertices - skipping visualization")
                except Exception as scalp_err:
                    print(f"Error processing scalp: {scalp_err}")
                    import traceback
                    traceback.print_exc()
                
                # 3. OPM sensors with improved transformation
                if "OPM Data" in self.file_paths and self.X1 is not None:
                    try:
                        print("\n=== PROCESSING OPM SENSORS ===")
                        import mne
                        from mne.io.constants import FIFF
                        
                        # Load raw MEG data
                        raw = mne.io.read_raw_fif(self.file_paths["OPM Data"], allow_maxshield=True, preload=False)
                        print(f"Loaded OPM data from {self.file_paths['OPM Data']}")
                        
                        # Extract sensor positions and names
                        sensor_positions = []
                        sensor_names = []
                        
                        print("Original sensor positions (device space, in metres):")
                        for ch in raw.info['chs']:
                            if ch['kind'] == FIFF.FIFFV_MEG_CH:
                                pos = ch['loc'][:3]
                                name = ch['ch_name']
                                
                                if np.allclose(pos, 0):
                                    continue
                                
                                print(f"{name}: {pos}")
                                sensor_positions.append(pos)
                                sensor_names.append(name)
                        
                        if not sensor_positions:
                            print("ERROR: No valid sensor positions found!")
                        else:
                            sensor_positions = np.array(sensor_positions)
                            
                            # Convert to mm for transformation
                            sensor_positions_mm = sensor_positions * 1000
                            
                            # Apply two-step transformation
                            print("Applying transformations to sensor positions...")
                            transformed_positions_mm = []
                            
                            for pos in sensor_positions_mm:
                                # Step 1: Device to helmet space (X1)
                                pos_h = np.append(pos, 1.0)
                                transformed = np.dot(self.X1, pos_h)[:3]
                                
                                # Step 2: If X2 is available, Helmet to standard/head space
                                if self.X2 is not None:
                                    pos_h2 = np.append(transformed, 1.0)
                                    transformed = np.dot(self.X2, pos_h2)[:3]
                                    
                                transformed_positions_mm.append(transformed)
                            
                            transformed_positions_mm = np.array(transformed_positions_mm)
                            
                            # Check for position clustering
                            print("\n=== INDIVIDUAL POSITION CHECK ===")
                            for i in range(min(5, len(transformed_positions_mm))):
                                print(f"Sensor {i} position (mm): {transformed_positions_mm[i]}")
                            
                            position_diffs = []
                            for i in range(1, len(transformed_positions_mm)):
                                diff = np.linalg.norm(transformed_positions_mm[i] - transformed_positions_mm[0])
                                position_diffs.append(diff)
                            
                            avg_diff = np.mean(position_diffs) if position_diffs else 0
                            print(f"Average position difference from first sensor: {avg_diff:.2f} mm")
                            
                            # If positions are too close together (clustering issue)
                            if avg_diff < 5.0:
                                print("WARNING: Sensors appear to be clustered very close together!")
                                print("Using original device positions and direct scaling instead...")
                                
                                # Create a direct layout based on original device positions
                                normalized_positions = []
                                sensor_center = np.mean(sensor_positions, axis=0)
                                
                                for pos in sensor_positions:
                                    # Center relative to sensor array center
                                    centered_pos = pos - sensor_center
                                    # Scale to match head size 
                                    scaled_pos = centered_pos * 150  # Scale to approximate head size in mm
                                    normalized_positions.append(scaled_pos)
                                
                                # Replace transformed positions with directly scaled positions
                                transformed_positions_mm = np.array(normalized_positions)
                                
                                # Verify new positions
                                new_diffs = []
                                for i in range(1, len(transformed_positions_mm)):
                                    diff = np.linalg.norm(transformed_positions_mm[i] - transformed_positions_mm[0])
                                    new_diffs.append(diff)
                                
                                new_avg_diff = np.mean(new_diffs) if new_diffs else 0
                                print(f"New average position difference: {new_avg_diff:.2f} mm")
                            
                            # Debug sensor positions
                            print("\n=== TRANSFORMED SENSOR POSITIONS ===")
                            for i, (name, pos) in enumerate(zip(sensor_names, transformed_positions_mm)):
                                if i < 5:  # Print just first 5 for brevity
                                    print(f"{name}: {pos}")
                            
                            print(f"Sensor centroid: {np.mean(transformed_positions_mm, axis=0)}")
                            print(f"Sensor min bounds: {np.min(transformed_positions_mm, axis=0)}")
                            print(f"Sensor max bounds: {np.max(transformed_positions_mm, axis=0)}")
                            
                            # Convert back to meters for visualization
                            transformed_positions = transformed_positions_mm / 1000.0
                            
                            # Debug scaling issues
                            print("\n=== SCALE COMPARISON ===")
                            if 'msr_points' in locals():
                                print(f"MSR scan range: {np.min(msr_points, axis=0)} to {np.max(msr_points, axis=0)}")
                                print(f"MSR bounding box: {np.max(np.abs(msr_range))}")
                            
                            if 'scalp_points_transformed' in locals():
                                print(f"Scalp range: {np.min(scalp_points_transformed, axis=0)} to {np.max(scalp_points_transformed, axis=0)}")
                                print(f"Scalp bounding box: {np.max(np.abs(scalp_range))}")
                            
                            sensor_range = np.max(transformed_positions, axis=0) - np.min(transformed_positions, axis=0)
                            print(f"Sensor range: {np.min(transformed_positions, axis=0)} to {np.max(transformed_positions, axis=0)}")
                            print(f"Sensor bounding box: {np.max(np.abs(sensor_range))}")
                            
                            # Check if we need to adjust scale
                            if 'msr_points' in locals():
                                msr_scale = np.max(np.abs(msr_range))
                                sensor_scale = np.max(np.abs(sensor_range))
                                scale_ratio = msr_scale / sensor_scale
                                
                                print(f"Scale ratio (MSR/sensors): {scale_ratio}")
                                
                                if scale_ratio > 10 or scale_ratio < 0.1:
                                    print(f"WARNING: Major scale mismatch detected! Adjusting scale by factor of {scale_ratio}")
                                    
                                    # Only apply if there's a real mismatch
                                    if scale_ratio > 10:
                                        adjustment_factor = 8  # More conservative than full ratio
                                    elif scale_ratio < 0.1:
                                        adjustment_factor = 0.2  # More conservative than full ratio
                                    else:
                                        adjustment_factor = 1
                                    
                                    # Apply adjustment - use a copy
                                    original_positions = transformed_positions.copy()
                                    transformed_positions = transformed_positions * adjustment_factor
                                    
                                    print(f"Adjusted sensor range: {np.min(transformed_positions, axis=0)} to {np.max(transformed_positions, axis=0)}")
                                    print(f"Adjustment factor: {adjustment_factor}")
                            
                            # Apply anatomically-aware positioning if we have scalp data
                            if 'scalp_points_transformed' in locals() and len(scalp_points_transformed) > 0:
                                print("Applying anatomically-aware sensor positioning...")
                                adjusted_positions_mm = anatomically_aware_sensor_positioning(
                                    transformed_positions_mm,
                                    scalp_points_transformed
                                )
                                adjusted_positions = adjusted_positions_mm / 1000.0  # convert back to meters
                                
                                # Debug adjusted positions
                                print("\n=== ADJUSTED SENSOR POSITIONS ===")
                                adjusted_centroid = np.mean(adjusted_positions, axis=0)
                                adjusted_range = np.max(adjusted_positions, axis=0) - np.min(adjusted_positions, axis=0)
                                print(f"Adjusted centroid: {adjusted_centroid}")
                                print(f"Adjusted position range: {adjusted_range}")
                                
                                # Use the adjusted positions for visualization
                                positions_to_render = adjusted_positions
                            else:
                                # Use the transformed positions
                                positions_to_render = transformed_positions
                            
                            # Add a yellow reference point at the origin
                            origin_sphere = vtk.vtkSphereSource()
                            origin_sphere.SetCenter(0, 0, 0)
                            origin_sphere.SetRadius(3.0)
                            origin_sphere.SetPhiResolution(16)
                            origin_sphere.SetThetaResolution(16)
                            
                            origin_mapper = vtk.vtkPolyDataMapper()
                            origin_mapper.SetInputConnection(origin_sphere.GetOutputPort())
                            
                            origin_actor = vtk.vtkActor()
                            origin_actor.SetMapper(origin_mapper)
                            origin_actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # Yellow
                            
                            self.ren.AddActor(origin_actor)
                            self.actors.append(origin_actor)

                            # APPROACH 1: RENDER SENSORS AS VERTICES
                            print("\n=== VISUALIZING SENSORS AS VERTICES ===")
                            # Create a single polydata with all sensor positions
                            sensor_points = vtk.vtkPoints()
                            for i, pos in enumerate(positions_to_render):
                                if i % 10 == 0:
                                    print(f"Adding sensor {i} ({sensor_names[i]}) at position {pos}")
                                x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                                sensor_points.InsertNextPoint(x, y, z)

                            # Create polydata
                            sensors_polydata = vtk.vtkPolyData()
                            sensors_polydata.SetPoints(sensor_points)

                            # Create vertices
                            sensors_vertices = vtk.vtkCellArray()
                            for i in range(sensor_points.GetNumberOfPoints()):
                                vertex = vtk.vtkVertex()
                                vertex.GetPointIds().SetId(0, i)
                                sensors_vertices.InsertNextCell(vertex)
                            sensors_polydata.SetVerts(sensors_vertices)

                            # Create mapper and actor
                            sensors_mapper = vtk.vtkPolyDataMapper()
                            sensors_mapper.SetInputData(sensors_polydata)
                            sensors_actor = vtk.vtkActor()
                            sensors_actor.SetMapper(sensors_mapper)
                            sensors_actor.GetProperty().SetColor(0.0, 0.0, 1.0)  # Blue
                            sensors_actor.GetProperty().SetPointSize(10)  # Large point size for visibility
                            sensors_actor.GetProperty().SetRenderPointsAsSpheres(1)  # Render as spheres
                            sensors_actor.ForceOpaqueOn()  # Ensure visibility

                            # Add to renderer
                            self.ren.AddActor(sensors_actor)
                            self.actors.append(sensors_actor)

                            print(f"Added {len(positions_to_render)} sensors as point vertices")

                            # APPROACH 2: CREATE SEPARATE VISUALIZATION WINDOW
                            try:
                                # Create a separate visualization window just for sensors
                                print("\n=== CREATING SEPARATE SENSOR VISUALIZATION ===")
                                sensor_window = vtk.vtkRenderWindow()
                                sensor_window.SetSize(600, 600)
                                sensor_window.SetWindowName("OPM Sensor Positions")

                                sensor_renderer = vtk.vtkRenderer()
                                sensor_renderer.SetBackground(0.0, 0.0, 0.0)  # Black background
                                sensor_window.AddRenderer(sensor_renderer)

                                # Create interactor
                                sensor_interactor = vtk.vtkRenderWindowInteractor()
                                sensor_interactor.SetRenderWindow(sensor_window)
                                sensor_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

                                # Add a coordinate system reference
                                axes = vtk.vtkAxesActor()
                                axes.SetTotalLength(0.1, 0.1, 0.1)  # 10cm axes
                                sensor_renderer.AddActor(axes)

                                # Add sensors as spheres with different colors
                                for i, pos in enumerate(positions_to_render):
                                    if i % 10 == 0:
                                        print(f"Adding sensor {i} ({sensor_names[i]}) to separate window")
                                    
                                    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
                                    
                                    # Create sphere
                                    sphere = vtk.vtkSphereSource()
                                    sphere.SetCenter(x, y, z)
                                    sphere.SetRadius(0.005)  # Small radius appropriate for meters scale
                                    sphere.SetPhiResolution(8)
                                    sphere.SetThetaResolution(8)
                                    
                                    # Create mapper and actor
                                    mapper = vtk.vtkPolyDataMapper()
                                    mapper.SetInputConnection(sphere.GetOutputPort())
                                    
                                    actor = vtk.vtkActor()
                                    actor.SetMapper(mapper)
                                    
                                    # Assign color based on position
                                    r = 0.5 + (x - np.min(positions_to_render[:, 0])) / (np.max(positions_to_render[:, 0]) - np.min(positions_to_render[:, 0]) + 1e-10) * 0.5
                                    g = 0.5 + (y - np.min(positions_to_render[:, 1])) / (np.max(positions_to_render[:, 1]) - np.min(positions_to_render[:, 1]) + 1e-10) * 0.5
                                    b = 0.5 + (z - np.min(positions_to_render[:, 2])) / (np.max(positions_to_render[:, 2]) - np.min(positions_to_render[:, 2]) + 1e-10) * 0.5
                                    actor.GetProperty().SetColor(r, g, b)
                                    
                                    # Add to renderer
                                    sensor_renderer.AddActor(actor)
                                    
                                    # Add label for some sensors
                                    if i % 15 == 0:
                                        text = vtk.vtkVectorText()
                                        text.SetText(sensor_names[i])
                                        
                                        text_mapper = vtk.vtkPolyDataMapper()
                                        text_mapper.SetInputConnection(text.GetOutputPort())
                                        
                                        text_actor = vtk.vtkActor()
                                        text_actor.SetMapper(text_mapper)
                                        text_actor.SetPosition(x + 0.01, y + 0.01, z + 0.01)
                                        text_actor.SetScale(0.005)
                                        text_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # White
                                        
                                        sensor_renderer.AddActor(text_actor)

                                # Reset camera and start (without blocking)
                                sensor_renderer.ResetCamera()
                                sensor_window.Render()
                                
                                # Initialize interactor in a non-blocking way
                                sensor_interactor.Initialize()
                                
                                # Start interactor without blocking (so it doesn't freeze the main program)
                                sensor_interactor.Start()
                                
                                print("Separate sensor window created")
                            except Exception as window_err:
                                print(f"Error creating separate window: {window_err}")
                                import traceback
                                traceback.print_exc()

                    except Exception as sensor_err:
                        print(f"Error processing OPM sensors: {sensor_err}")
                        import traceback
                        traceback.print_exc()
                
                # 4. Add fiducials with orange color to distinguish from MSR
                if hasattr(self, 'fiducial_points') and self.fiducial_points is not None:
                    fiducial_labels = ["Nasion", "Left Auricular", "Right Auricular"]
                    
                    for i, (point, label) in enumerate(zip(self.fiducial_points, fiducial_labels)):
                        # Create sphere for fiducial
                        fiducial_sphere = vtk.vtkSphereSource()
                        fiducial_sphere.SetCenter(point)
                        fiducial_sphere.SetRadius(5.0)
                        fiducial_sphere.SetPhiResolution(16)
                        fiducial_sphere.SetThetaResolution(16)
                        
                        fiducial_mapper = vtk.vtkPolyDataMapper()
                        fiducial_mapper.SetInputConnection(fiducial_sphere.GetOutputPort())
                        
                        fiducial_actor = vtk.vtkActor()
                        fiducial_actor.SetMapper(fiducial_mapper)
                        fiducial_actor.GetProperty().SetColor(1.0, 0.6, 0.0)  # Orange
                        
                        # Add text label
                        text = vtk.vtkVectorText()
                        text.SetText(label)
                        
                        text_mapper = vtk.vtkPolyDataMapper()
                        text_mapper.SetInputConnection(text.GetOutputPort())
                        
                        text_actor = vtk.vtkActor()
                        text_actor.SetMapper(text_mapper)
                        text_actor.SetPosition(point[0] + 10, point[1] + 10, point[2] + 10)
                        text_actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # White
                        text_actor.SetScale(5.0)
                        
                        self.ren.AddActor(fiducial_actor)
                        self.ren.AddActor(text_actor)
                        self.actors.append(fiducial_actor)
                        self.actors.append(text_actor)
                    
                    print(f"Added {len(self.fiducial_points)} fiducial markers with orange color")
                
                # 5. Enhance lighting for better visibility
                # Remove existing lights
                lights = self.ren.GetLights()
                lights.InitTraversal()
                light = lights.GetNextItem()
                while light:
                    self.ren.RemoveLight(light)
                    light = lights.GetNextItem()
                
                # Add new balanced lighting
                light1 = vtk.vtkLight()
                light1.SetLightTypeToHeadlight()
                light1.SetIntensity(0.7)
                light1.SetDiffuseColor(1.0, 1.0, 1.0)
                light1.SetAmbientColor(0.2, 0.2, 0.2)
                self.ren.AddLight(light1)
                
                light2 = vtk.vtkLight()
                light2.SetPosition(200, 200, 200)
                light2.SetFocalPoint(0, 0, 0)
                light2.SetIntensity(0.7)
                light2.SetDiffuseColor(1.0, 1.0, 1.0)
                self.ren.AddLight(light2)
                
                # Extra light from below for better detail
                light3 = vtk.vtkLight()
                light3.SetPosition(0, -200, 0)
                light3.SetFocalPoint(0, 0, 0)
                light3.SetIntensity(0.4)
                light3.SetDiffuseColor(0.8, 0.8, 1.0)  # Slight blue tint
                self.ren.AddLight(light3)
                
                # Reset camera view
                self.ren.ResetCamera()
                camera = self.ren.GetActiveCamera()
                camera.Azimuth(30)  # Slight rotation
                camera.Elevation(20)  # Slight elevation
                
                # Render final scene
                self.vtk_widget.GetRenderWindow().Render()
                print("Integrated visualization complete in main window")
                
            except Exception as vis_err:
                print(f"Visualization error: {vis_err}")
                import traceback
                traceback.print_exc()

            # Success message
            QMessageBox.information(
                self, "Co-registration Complete",
                "The co-registration process has been completed successfully.\n\n"
                "You can now click 'Save Results' to save the outputs."
            )

            self.current_stage = "finished"
            self.updateInstructions()

            self.save_button.show()
            self.save_button.setEnabled(True)
            self.clear_button.hide()
            self.reverse_button.hide()
            self.confirm_button.hide()
            self.fit_button.hide()
            self.continue_button.hide()

        except Exception as e:
            print(f"Error in finalisation: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in finalisation: {str(e)}")
                                
    def saveResults(self):
        try:
            import os
            import mne
            from mne.transforms import Transform
            import numpy as np
            from PyQt5.QtWidgets import QMessageBox

            if "OPM Data" not in self.file_paths:
                raise ValueError("OPM Data file not loaded.")

            raw_path = self.file_paths["OPM Data"]
            raw = mne.io.read_raw_fif(raw_path, preload=False)

            # Create dev_head_t from helmet → head
            if self.X1 is not None and self.X1.shape == (4, 4):
                helmet_to_head = np.linalg.inv(self.X1)
                dev_head_t = Transform("meg", "head", helmet_to_head)
                raw.info['dev_head_t'] = dev_head_t
            else:
                raise ValueError("X1 transformation is missing or invalid.")

            # Save transformation to a .fif file
            base, ext = os.path.splitext(raw_path)
            if ext == ".gz":
                base, _ = os.path.splitext(base)
            trans_path = base + "_trans.fif"

            mne.write_trans(trans_path, dev_head_t, overwrite=True)
            print(f"✅ Transformation saved to {trans_path}")

            # Save updated raw file
            transformed_raw_path = base + "_transformed.fif"
            raw.save(transformed_raw_path, overwrite=True)
            print(f"✅ Raw file with transform saved to {transformed_raw_path}")

            QMessageBox.information(
                self,
                "Saved",
                f"Transform and raw file saved:\n\n"
                f"{os.path.basename(trans_path)}\n{os.path.basename(transformed_raw_path)}"
            )

        except Exception as e:
            print(f"❌ Error saving results: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error saving results:\n{str(e)}")
                
if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    import vtk
    import numpy as np
    import open3d as o3d
    import importlib

    def check_environment():
        """Check the versions of key dependencies"""
        print("\n=== ENVIRONMENT CHECK ===")
        
        # Check VTK version
        print(f"VTK version: {vtk.vtkVersion().GetVTKVersion()}")
        
        # Check if VTK was built with Qt support
        try:
            from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            print("QVTKRenderWindowInteractor available: Yes")
        except ImportError:
            print("WARNING: QVTKRenderWindowInteractor not available - VTK may not have Qt support")
        
        # Check PyQt5 version
        from PyQt5.QtCore import QT_VERSION_STR
        print(f"Qt version: {QT_VERSION_STR}")
        
        # Check Python version
        print(f"Python version: {sys.version}")
        
        # Check for OpenGL support
        try:
            from PyQt5.QtGui import QOpenGLContext
            context = QOpenGLContext()
            if context.create():
                print("OpenGL context successfully created")
                format = context.format()
                print(f"OpenGL Version: {format.majorVersion()}.{format.minorVersion()}")
            else:
                print("WARNING: Could not create OpenGL context")
        except Exception as e:
            print(f"OpenGL context check error: {e}")
        
        # Print OS information
        import platform
        print(f"OS: {platform.system()} {platform.release()}")
        
        # Check if running on Apple Silicon
        if platform.system() == 'Darwin':
            import subprocess
            try:
                processor = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string']).decode().strip()
                print(f"Processor: {processor}")
                
                # Check if running under Rosetta
                try:
                    output = subprocess.check_output(['sysctl', '-n', 'sysctl.proc_translated']).decode().strip()
                    if output == '1':
                        print("WARNING: Running under Rosetta 2 translation")
                    else:
                        print("Running natively")
                except:
                    print("Could not determine if running under Rosetta")
                    
            except:
                print("Could not determine processor information")

        # Check required packages
        required_packages = [
            "vtk", "PyQt5", "numpy", "open3d", "mne", "scipy"
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                print(f"{package}: ✓ Installed")
            except ImportError:
                print(f"{package}: ✗ NOT INSTALLED!")
        
        print("=== END ENVIRONMENT CHECK ===\n")

    def start_gui():
        # Run environment check
        check_environment()
        
        # Test basic VTK rendering
        test_basic_vtk_rendering()
        
        # Instantiate the OPMCoRegistrationGUI class
        window = OPMCoRegistrationGUI()
        window.show()

    def test_basic_vtk_rendering():
        """Test if basic VTK rendering works"""
        print("\n=== TESTING BASIC VTK RENDERING ===")
        try:
            # Create a simple render window to test VTK
            renderer = vtk.vtkRenderer()
            render_window = vtk.vtkRenderWindow()
            render_window.AddRenderer(renderer)
            
            # Create a simple sphere
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(0, 0, 0)
            sphere.SetRadius(1.0)
            sphere.Update()
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            renderer.AddActor(actor)
            renderer.SetBackground(0.1, 0.2, 0.3)
            
            # Try rendering
            render_window.SetSize(100, 100)
            render_window.Render()
            
            print("Basic VTK rendering test PASSED")
            
        except Exception as e:
            print(f"Basic VTK rendering test FAILED: {e}")
            import traceback
            traceback.print_exc()

    app = QApplication(sys.argv)
    
    # Ensuring GUI runs on the main Qt event thread
    QTimer.singleShot(0, start_gui)
    
    # Start the event loop
    sys.exit(app.exec_())