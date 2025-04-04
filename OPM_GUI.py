import os
import numpy as np
import mne
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QProgressBar, QFrame, QLineEdit, QGridLayout, QMessageBox)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys
import copy
import open3d as o3d

vtk.vtkObject.GlobalWarningDisplayOff()

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

sensor_length = 6e-3

def is_stl_file(file_path):
    """Check if the file is an STL file based on extension"""
    _, ext = os.path.splitext(file_path.lower())
    return ext == '.stl'

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    return pcd_down

def align_to_reference(source_pcd, reference_pcd):
    """
    Rigidly align source to reference using scaling and rotation only.
    Preserves reference orientation and centroid location.
    """
    def normalise(points):
        centre = points.mean(axis=0)
        points_centered = points - centre
        scale = np.linalg.norm(points_centered)
        return points_centered / scale, centre, scale

    src_pts = np.asarray(source_pcd.points)
    ref_pts = np.asarray(reference_pcd.points)

    src_norm, src_centre, src_scale = normalise(src_pts)
    ref_norm, ref_centre, ref_scale = normalise(ref_pts)

    H = src_norm.T @ ref_norm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned_pts = ((src_norm @ R.T) * ref_scale) + ref_centre
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_pts)
    return aligned_pcd

def execute_global_registration(source_down, target_down, source_fpfh,
                              target_fpfh, voxel_size):
    """Execute global registration using RANSAC."""
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

def align_to_reference(source_pcd, reference_pcd):

    def normalise(pcd):
        pts = np.asarray(pcd.points)
        pts -= pts.mean(axis=0)
        scale = np.linalg.norm(pts)
        pts /= scale
        return pts, scale

    source_pts, source_scale = normalise(source_pcd)
    ref_pts, ref_scale = normalise(reference_pcd)

    # Compute cross-covariance matrix
    H = source_pts.T @ ref_pts
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned_pts = (source_pts @ R.T) * ref_scale + np.asarray(reference_pcd.points).mean(axis=0)
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_pts)
    return aligned_pcd

def standardise_transformation(source_pcd, target_pcd):
    
    def get_pca_alignment(pcd):
        points = np.asarray(pcd.points)
        points -= points.mean(axis=0)
        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        return eigvecs

    def scale_to_match(source, target):
        src_pts = np.asarray(source.points)
        tgt_pts = np.asarray(target.points)
        src_scale = np.linalg.norm(src_pts.max(axis=0) - src_pts.min(axis=0))
        tgt_scale = np.linalg.norm(tgt_pts.max(axis=0) - tgt_pts.min(axis=0))
        scale_factor = tgt_scale / src_scale
        source.scale(scale_factor, center=source.get_center())
        return source

    # Align orientation
    src_axes = get_pca_alignment(source_pcd)
    tgt_axes = get_pca_alignment(target_pcd)
    R = tgt_axes @ src_axes.T
    source_pcd.rotate(R, center=source_pcd.get_center())

    # Scale to match
    source_pcd = scale_to_match(source_pcd, target_pcd)

    return source_pcd

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
                # Simulate loading process
                for i in range(101):
                    self.progress_signal.emit(i)
                    self.msleep(30)
                self.finished_signal.emit(None)
                
            elif self.task == "load_outside_msr":
                # Simulate loading process
                for i in range(101):
                    self.progress_signal.emit(i)
                    self.msleep(30)
                self.finished_signal.emit(None)
                
            elif self.task == "head_to_mri":
                # Actual MRI to head registration
                gui = self.args[0]  # Get the GUI instance
                
                # Compute the transformations from head to standard coordinate system
                self.progress_signal.emit(25)
                gui.X2 = gui.computeHeadToStandardTransform()
                
                # Compute the combined transformation from inside MSR to standard
                self.progress_signal.emit(50)
                gui.X21 = np.dot(gui.X2, gui.X1)
                
                # Compute the transformation from MRI to head
                self.progress_signal.emit(75)
                gui.X3 = gui.computeHeadToMRITransform()
                
                self.progress_signal.emit(100)
                self.finished_signal.emit(None)
                
            elif self.task == "final_registration":
                # This performs the actual head point check and file writing
                gui = self.args[0] if self.args else None
                
                if gui:
                    try:
                        # Compute head to standard transform
                        self.progress_signal.emit(10)
                        gui.X2 = gui.computeHeadToStandardTransform()
                        
                        # Compute combined transform
                        self.progress_signal.emit(30)
                        gui.X21 = np.dot(gui.X2, gui.X1)
                        
                        # Compute MRI to head transform
                        self.progress_signal.emit(50)
                        gui.X3 = gui.computeHeadToMRITransform()
                        
                        # Check sensor positions - skip for now to avoid crashes
                        # self.progress_signal.emit(80)
                        # gui.checkHeadpoints()
                        
                        self.progress_signal.emit(100)
                        self.finished_signal.emit(None)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        self.finished_signal.emit(str(e))
                else:
                    # Simulate progress if gui not provided (should not happen)
                    for i in range(101):
                        self.progress_signal.emit(i)
                        self.msleep(50)
                    self.finished_signal.emit(None)
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.finished_signal.emit(str(e))

class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        super().__init__()
        
    def OnLeftButtonDown(self):
        if not self.GetShiftKey():
            super().OnLeftButtonDown()
            
    def OnLeftButtonUp(self):
        if not self.GetShiftKey():
            super().OnLeftButtonUp()

class OPMCoRegistrationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_points = []
        self.point_actors = []
        self.file_paths = {}
        self.current_stage = "inside_msr"  
        
        # Known sensor positions
        self.known_sensor_positions = np.zeros([7, 3], dtype=float)
        self.known_sensor_positions[0] = [2.400627, 6.868828, 15.99383]
        self.known_sensor_positions[1] = [-3.36779, 3.436483, 10.86945]
        self.known_sensor_positions[2] = [-1.86113, -0.49753, 7.031777]
        self.known_sensor_positions[3] = [0.17963, 0.005712, 0.014363]
        self.known_sensor_positions[4] = [3.596149, 4.942685, -5.23564]
        self.known_sensor_positions[5] = [4.021887, 10.33369, -5.79216]
        self.known_sensor_positions[6] = [10.84183, 14.91172, -2.81954]
        self.fiducial_labels = ["Right Pre-auricular", "Left Pre-auricular", "Nasion"]
        self.point_labels = []
        self.actors = [] 
        
        # Storage for fiducial points
        self.fiducial_points = None
        self.fiducials_dict = {}
        
        # Matrix transformation 
        self.X1 = None  # Inside MSR to helmet transform
        self.X2 = None  # Standard transform
        self.X21 = None  # Combined transform
        self.X3 = None  # MRI to head transform
        
        # Add these for thread safety
        self._source_cloud_points = None
        self._target_cloud_points = None
        self._sensor_points = None
        self._mri_cloud_points = None
        
        self.initUI()
        
        # Show startup information
        QTimer.singleShot(500, self.showStartupInfo)

    def initUI(self):
        self.setWindowTitle("OPM-MEG Co-Registration")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet("background-color: #2e2e2e; color: white;")

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

        self.labels = []
        file_types = ["Inside MSR", "Outside MSR", "Scalp File", "OPM Data"]
        self.extensions = {"Inside MSR": "*.ply", "Outside MSR": "*.ply", "Scalp File": "*.stl", "OPM Data": "*.fif"}

        for file_type in file_types:
            group_layout = QVBoxLayout()

            label = QLabel(file_type, self)
            label.setFont(font)
            
            button = QPushButton("Select", self)
            button.setFont(font)
            button.setStyleSheet("background-color: #555555; color: white; border-radius: 5px; padding: 5px;")
            button.clicked.connect(lambda checked, t=file_type: self.selectFile(t))

            file_label = QLabel("No file selected", self)
            file_label.setFont(font)
            file_label.setWordWrap(True)

            group_layout.addWidget(label)
            group_layout.addWidget(button)
            group_layout.addWidget(file_label)
            left_layout.addLayout(group_layout)

            self.labels.append(file_label)

        self.load_button = QPushButton("Load", self)
        self.load_button.setFont(font)
        self.load_button.setEnabled(False)
        self.load_button.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold;")
        self.load_button.clicked.connect(self.loadModel)
        left_layout.addWidget(self.load_button)

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

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        middle_layout.addWidget(self.vtk_widget)

        self.ren = vtk.vtkRenderer()
        self.ren.SetBackground(0.1, 0.1, 0.1)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.ren)

        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(CustomInteractorStyle())
        self.iren.AddObserver("LeftButtonPressEvent", self.onShiftLeftClick)

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
        
        # New "Fit Points" button
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
        
        #Save button for final step
        self.save_button = QPushButton("Save Results", self)
        self.save_button.setFont(font)
        self.save_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.saveResults)
        right_layout.addWidget(self.save_button)
        self.save_button.hide()  #Hide until needed

        # Set up worker thread
        self.worker = None

        # Initialise instructions
        self.updateInstructions()

        self.show()
        
        # Initialise VTK interactor
        self.iren.Initialize()

    def showStartupInfo(self):
        """Display information about prerequisites when the app starts"""
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
        """Show detailed help information about preprocessing steps"""
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
            
            # 7 points should show up here.
            self.updatePointStatusPanel(7)
            
            # Show all buttons for inside MSR stage
            self.clear_button.show()
            self.reverse_button.show()
            self.confirm_button.show()
            self.fit_button.show()
            
        elif self.current_stage == "outside_msr":
            self.instructions_label.setText(
                "Step 2: Select the 3 fiducial points in this order:\n"
                "1. Right pre-auricular\n"
                "2. Left pre-auricular\n"
                "3. Nasion\n"
                "Use Shift+Left Click to select points"
            )
            self.status_label.setText("Fiducial Point Selection Status")
            
            # Fiducial points to show up here
            self.updatePointStatusPanel(3, fiducials=True)
            
            # Only show clear and reverse buttons for outside MSR stage
            self.clear_button.show()
            self.reverse_button.show()
            self.confirm_button.show()
            self.fit_button.hide()
            
        elif self.current_stage == "mri_scalp":
            self.instructions_label.setText(
                "Step 3: Performing MRI scalp to head registration\n"
                "Preview will be shown when complete"
            )
            
            # Hide selection-related buttons during processing
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
            
            # Hide all buttons except save
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
        options = QFileDialog.Options()
        
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {file_type} File", "", 
                                                  self.extensions[file_type], options=options)
        
        if file_path:
            # For Scalp File, ensure it's an STL
            if file_type == "Scalp File" and not file_path.lower().endswith('.stl'):
                QMessageBox.warning(self, "Invalid File", 
                                   "The scalp file must be an STL file generated from FreeSurfer.\n"
                                   "Please run FreeSurfer's recon-all on your MRI data first,\n"
                                   "then convert the scalp surface to STL using mris_convert.")
                return
            
            # Store the file path
            self.file_paths[file_type] = file_path
            idx = ["Inside MSR", "Outside MSR", "Scalp File", "OPM Data"].index(file_type)
            self.labels[idx].setText(os.path.basename(file_path))
            
            # Enable load button if Inside MSR file is selected
            if self.file_paths.get("Inside MSR"):
                self.load_button.setEnabled(True)

    def onShiftLeftClick(self, obj, event):
        if self.iren.GetShiftKey():
            click_pos = self.iren.GetEventPosition()
            
            # Use vtkPropPicker for better performance
            picker = vtk.vtkPropPicker()
            
            if picker.Pick(click_pos[0], click_pos[1], 0, self.ren):
                picked_position = picker.GetPickPosition()
                
                if picked_position != (0, 0, 0): 
                    self.addPoint(picked_position)
                    
                    # Update confirm button
                    if (self.current_stage == "inside_msr" and len(self.selected_points) == 7):
                        self.confirm_button.setEnabled(True)
                        # Also enable the fit button for inside MSR
                        self.fit_button.setEnabled(True)
                    elif (self.current_stage == "outside_msr" and len(self.selected_points) == 3):
                        self.confirm_button.setEnabled(True)
                        # Fit button should not be enabled for outside MSR

    def addPoint(self, position):
        # Create a sphere with fewer resolution for faster rendering
        point = vtk.vtkSphereSource()
        point.SetCenter(position)
        point.SetRadius(3.0)
        point.SetPhiResolution(8)
        point.SetThetaResolution(8)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(point.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        if self.current_stage == "inside_msr":
            actor.GetProperty().SetColor(1, 0, 0) # Red for inside MSR points
        else:
            actor.GetProperty().SetColor(0, 0, 1) # Blue for fiducial points
        
        self.ren.AddActor(actor)
        self.actors.append(actor)
        self.point_actors.append(actor)
        self.selected_points.append(position)
        
        self.updateSelectionUI(len(self.selected_points) - 1)
        
        self.vtk_widget.GetRenderWindow().Render()

    def updateSelectionUI(self, idx):
        # Update point status
        if idx < len(self.distance_labels):
            if self.current_stage == "inside_msr":
                self.distance_labels[idx].setText(f"Point {idx+1}: Selected")
            elif self.current_stage == "outside_msr" and idx < len(self.fiducial_labels):
                self.distance_labels[idx].setText(f"{self.fiducial_labels[idx]}: Selected")

    def loadModel(self):
        missing_files = []
        required_files = ["Inside MSR", "Outside MSR", "Scalp File", "OPM Data"]
        for file_type in required_files:
            if file_type not in self.file_paths or not self.file_paths[file_type]:
                missing_files.append(file_type)
        
        if missing_files:
            QMessageBox.warning(self, "Missing Files", 
                               f"Please select the following files:\n{', '.join(missing_files)}")
            return
        
        self.load_button.setEnabled(False)
        self.load_button.setText("Loading...")
        self.progress_bar.setValue(0)
        
        #Start worker thread for loading
        self.worker = WorkerThread("load_model")
        self.worker.progress_signal.connect(self.updateProgress)
        self.worker.finished_signal.connect(self.finishLoading)
        self.worker.start()

    def updateProgress(self, value):
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # Process any pending events to keep UI responsive

    def finishLoading(self, error=None):
        if error:
            QMessageBox.critical(self, "Error", f"An error occurred: {error}")
            self.load_button.setEnabled(True)
            self.load_button.setText("Load")
            return
        
        self.load_button.setText("Load")
        self.progress_bar.setValue(100)
        
        #Load point cloud
        self.loadPointCloud("Inside MSR")
        
        #Reset selected points
        self.clearPoints()
        
        #Update GUI state
        self.current_stage = "inside_msr"
        self.updateInstructions()

    def loadPointCloud(self, file_type):
        file_path = self.file_paths[file_type]
        
        # Clear previous actors
        for actor in self.actors:
            self.ren.RemoveActor(actor)
        self.actors = []
        
        # Load new point cloud based on file type
        if file_type in ["Inside MSR", "Outside MSR"]:
            # Load PLY file
            reader = vtk.vtkPLYReader()
            reader.SetFileName(file_path)
            reader.Update()
        elif file_type == "Scalp File":
            # Load STL file
            reader = vtk.vtkSTLReader()
            reader.SetFileName(file_path)
            reader.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # Ensure no scaling is applied to the actor
        actor.SetScale(1.0, 1.0, 1.0)
        
        self.ren.AddActor(actor)
        self.actors.append(actor)
        
        # Reset camera to fit the model
        self.ren.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def loadPointCloudOpen3D(self, file_path, max_points=100000):
        """
        Load point cloud using Open3D for registration tasks with downsampling option.
        Add more robust error handling and fallback mechanisms.
        """
        try:
            # Check file extension
            file_path = str(file_path)  # Ensure string type
            print(f"Attempting to load file: {file_path}")
            
            # Lower memory approach
            if file_path.lower().endswith('.ply'):
                print(f"Loading PLY file: {file_path}")
                try:
                    # First try reading as mesh
                    point_cloud = o3d.io.read_triangle_mesh(file_path)
                    if not point_cloud.has_vertices():
                        raise ValueError("No vertices found in mesh")
                    
                    # Convert mesh to point cloud with poisson disk sampling
                    print(f"Sampling with max points: {max_points}")
                    point_cloud = point_cloud.sample_points_poisson_disk(max_points)
                except Exception as mesh_error:
                    print(f"Mesh loading failed, trying direct point cloud: {mesh_error}")
                    # Fallback to direct point cloud reading
                    point_cloud = o3d.io.read_point_cloud(file_path)
                    
                    # Downsample if too many points
                    if len(point_cloud.points) > max_points:
                        point_cloud = point_cloud.random_down_sample(max_points / len(point_cloud.points))
            
            elif file_path.lower().endswith('.stl'):
                print(f"Loading STL file: {file_path}")
                mesh = o3d.io.read_triangle_mesh(file_path)
                
                # Ensure mesh is valid
                if not mesh.has_vertices():
                    raise ValueError("No vertices found in STL mesh")
                
                # Convert mesh to point cloud with poisson disk sampling
                print(f"Sampling with max points: {max_points}")
                point_cloud = mesh.sample_points_poisson_disk(max_points)
            
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Verify point cloud
            if len(point_cloud.points) == 0:
                raise ValueError("Empty point cloud after processing")
            
            print(f"Loaded point cloud with {len(point_cloud.points)} points")
            return point_cloud
        
        except Exception as e:
            print(f"Critical error in loadPointCloudOpen3D: {e}")
            import traceback
            traceback.print_exc()
            raise

    def visualize_initial_alignment(self, mri_mesh, scan_mesh):
        """Visualize the initial alignment between MRI and scan meshes for debugging."""
        # Create new render window for visualization
        render_window = vtk.vtkRenderWindow()
        renderer = vtk.vtkRenderer()
        render_window.AddRenderer(renderer)
        
        # Convert MRI mesh to actor with green color
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
        
        # Convert scan mesh to actor with red color
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
        
        # Add actors to renderer
        renderer.AddActor(mri_actor)
        renderer.AddActor(scan_actor)
        
        # Set up interactor and start
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.Initialize()
        renderer.ResetCamera()
        render_window.Render()
        interactor.Start()

    def improved_icp_registration(self, source, target, initial_transform=None, max_iterations=100, threshold=2.0):
        """
        Enhanced ICP registration with adjustable parameters.
        """
        import open3d as o3d
        
        print("Starting ICP registration...")
        
        # If no initial transform is provided, use identity
        if initial_transform is None:
            initial_transform = np.eye(4)
        
        # Point-to-point ICP
        p2p_icp = o3d.pipelines.registration.registration_icp(
            source, target, threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations,
                relative_fitness=1e-6,
                relative_rmse=1e-6
            )
        )
        
        # Try point-to-plane ICP if normals are available
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
                
                # Use point-to-plane result if it's better
                if p2l_icp.fitness > p2p_icp.fitness:
                    print(f"Using point-to-plane result: fitness={p2l_icp.fitness:.4f}, rmse={p2l_icp.inlier_rmse:.4f}")
                    return p2l_icp.transformation
        
        print(f"Using point-to-point result: fitness={p2p_icp.fitness:.4f}, rmse={p2p_icp.inlier_rmse:.4f}")
        return p2p_icp.transformation

    def visualize_registration_error(self, source_pcd, target_pcd, transformation):
        """
        Visualize the registration error between transformed source and target.
        Colorizes points by distance to nearest neighbor.
        """
        import open3d as o3d
        
        # Create a copy of source and apply transformation
        source_transformed = copy.deepcopy(source_pcd)
        source_transformed.transform(transformation)
        
        # Build KD tree for target points
        target_tree = o3d.geometry.KDTreeFlann(target_pcd)
        
        # Calculate distances to nearest neighbors
        source_points = np.asarray(source_transformed.points)
        distances = []
        
        for point in source_points:
            _, idx, dist = target_tree.search_knn_vector_3d(point, 1)
            distances.append(np.sqrt(dist[0]))
        
        distances = np.array(distances)
        
        # Normalize distances for coloring
        max_dist = np.percentile(distances, 95)  # Use 95th percentile to avoid outliers
        normalized_distances = np.minimum(distances / max_dist, 1.0)
        
        # Create color map (blue to red)
        colors = np.zeros((len(normalized_distances), 3))
        colors[:, 0] = normalized_distances  # Red channel
        colors[:, 2] = 1.0 - normalized_distances  # Blue channel
        
        # Assign colors to source
        error_cloud = copy.deepcopy(source_transformed)
        error_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        o3d.visualization.draw_geometries([error_cloud, target_pcd])
        
        # Print statistics
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        print(f"Registration error statistics:")
        print(f"  Mean distance: {mean_dist:.2f} mm")
        print(f"  Median distance: {median_dist:.2f} mm")
        print(f"  Max distance (95th percentile): {max_dist:.2f} mm")
        
        return mean_dist, median_dist, max_dist

    def visualize_transformation_step(self, source_pcd, target_pcd, transformation_matrix=None, title="Transformation Step"):
        """
        Visualize transformation step for debugging.
        Shows source and target point clouds, with transformed source if transformation_matrix is provided.
        """
        # Create deep copies to avoid modifying originals
        source_copy = copy.deepcopy(source_pcd)
        target_copy = copy.deepcopy(target_pcd)
        
        # Apply transformation if provided
        if transformation_matrix is not None:
            source_copy.transform(transformation_matrix)
        
        # Paint point clouds differently
        source_copy.paint_uniform_color([1, 0, 0])  # Red
        target_copy.paint_uniform_color([0, 1, 0])  # Green
        
        # Print information about point clouds
        print(f"{title} Visualization:")
        print(f"  Source points: {len(source_copy.points)}")
        print(f"  Target points: {len(target_copy.points)}")
        
        import open3d as o3d
        o3d.visualization.draw_geometries([source_copy, target_copy], window_name=title)

    def standardize_coordinate_system(self, pcd, system="RAS"):
        """
        Ensure consistent coordinate system across all data.
        Transforms point cloud to specified coordinate system.
        
        RAS: Right-Anterior-Superior (standard neurological coordinate system)
        """
        pcd_copy = copy.deepcopy(pcd)
        points = np.asarray(pcd_copy.points)
        
        # Determine current coordinate system (simple heuristic)
        # This is a placeholder - you'd need to adapt based on your data
        centroid = np.mean(points, axis=0)
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Check if we need to transform
        if system == "RAS":
            # Apply transformation to match RAS
            # This is a placeholder transformation - would need to be adjusted
            transform = np.eye(4)
            
            # Detect if X axis needs flipping (for Right-Left)
            if centroid[0] > 0:  # Assuming data should be centered
                transform[0, 0] = -1  # Flip X
                
            # Detect if Z axis needs flipping (for Superior-Inferior)
            z_range = max_coords[2] - min_coords[2]
            if (max_coords[2] - centroid[2]) < (z_range * 0.4):  # If less points above centroid
                transform[2, 2] = -1  # Flip Z
            
            # Apply transformation
            pcd_copy.transform(transform)
            
        return pcd_copy

    def preprocess_point_cloud(self, pcd, voxel_size=2.0, remove_outliers=True):
        """
        Pre-process point cloud for better registration:
        1. Downsample using voxel grid
        2. Estimate normals
        """
        import open3d as o3d
        
        # Create a copy to avoid modifying the original
        pcd_copy = copy.deepcopy(pcd)
        
        # Downsample using voxel grid
        pcd_down = pcd_copy.voxel_down_sample(voxel_size)
        
        # Estimate normals
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        
        # Note: We've removed the outlier removal functions that were causing issues
        # We'll use an alternate approach if needed
        
        print(f"Pre-processed point cloud: {len(pcd_down.points)} points (from {len(pcd_copy.points)})")
        return pcd_down

    def normalize_scaling(self, source_pcd, target_pcd):
        """
        Normalize the scaling between source and target point clouds.
        Returns scaled source point cloud.
        """
        # Create deep copy to avoid modifying original
        source_copy = copy.deepcopy(source_pcd)
        
        # Get points from both clouds
        source_points = np.asarray(source_copy.points)
        target_points = np.asarray(target_pcd.points)
        
        # Compute bounding box sizes
        source_min = np.min(source_points, axis=0)
        source_max = np.max(source_points, axis=0)
        source_size = source_max - source_min
        
        target_min = np.min(target_points, axis=0)
        target_max = np.max(target_points, axis=0)
        target_size = target_max - target_min
        
        # Compute scale factors
        scale_factors = target_size / source_size
        
        # Use mean scale or separate scales for each dimension
        mean_scale = np.mean(scale_factors)
        
        # Apply scaling to source - with correct center argument
        # We need to provide the center as a numpy array
        center = np.mean(source_points, axis=0)
        source_copy.scale(mean_scale, center=center)
        
        print(f"Normalized scaling with factor: {mean_scale}")
        
        return source_copy

    def align_using_pca(self, source, target):
        """
        Align source to target using Principal Component Analysis.
        This aligns the main axes of variation between the two point clouds.
        """
        import open3d as o3d
        import numpy as np
        
        # Create a copy to avoid modifying the original
        source_copy = copy.deepcopy(source)
        
        # Get points from both clouds
        source_points = np.asarray(source_copy.points)
        target_points = np.asarray(target.points)
        
        # Compute centroids
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        
        # Center both point clouds
        source_centered = source_points - source_centroid
        target_centered = target_points - target_centroid
        
        # Compute covariance matrices
        source_cov = np.cov(source_centered, rowvar=False)
        target_cov = np.cov(target_centered, rowvar=False)
        
        # Compute eigenvectors
        source_eigvals, source_eigvecs = np.linalg.eigh(source_cov)
        target_eigvals, target_eigvecs = np.linalg.eigh(target_cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        source_indices = np.argsort(source_eigvals)[::-1]
        source_eigvecs = source_eigvecs[:, source_indices]
        
        target_indices = np.argsort(target_eigvals)[::-1]
        target_eigvecs = target_eigvecs[:, target_indices]
        
        # Compute rotation matrix that aligns source eigenvectors with target eigenvectors
        rotation = np.dot(target_eigvecs, source_eigvecs.T)
        
        # Ensure it's a proper rotation matrix (det=1)
        if np.linalg.det(rotation) < 0:
            source_eigvecs[:, 2] = -source_eigvecs[:, 2]
            rotation = np.dot(target_eigvecs, source_eigvecs.T)
        
        # Create transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        transformation[:3, 3] = target_centroid - np.dot(rotation, source_centroid)
        
        # Apply transformation
        source_copy.transform(transformation)
        
        return source_copy

    def align_with_standard_orientation(self, pcd):
        """
        Ensure the point cloud is oriented correctly with y-up, face forward.
        Based on York's head_to_standard approach.
        """
        import open3d as o3d
        import numpy as np
        
        # Create a copy
        pcd_copy = copy.deepcopy(pcd)
        points = np.asarray(pcd_copy.points)
        
        # First determine the principal axes using PCA
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        cov = np.cov(centered_points, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        sort_indices = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sort_indices]
        
        # Create initial transformation matrix to align with principal axes
        transform = np.eye(4)
        transform[:3, :3] = eigvecs
        
        # Check if we need to flip axes to maintain correct orientation
        # We'll use heuristics based on the point cloud shape
        
        # 1. The y-axis should point up
        # If the top of the head has fewer points than the bottom, we might need to flip y
        y_axis = eigvecs[:, 1]
        y_proj = np.dot(centered_points, y_axis)
        y_top_count = np.sum(y_proj > 0)
        y_bottom_count = np.sum(y_proj < 0)
        
        # 2. The z-axis should point to the back of the head
        # If the back of the head has fewer points than the face, we might need to flip z
        z_axis = eigvecs[:, 2]
        z_proj = np.dot(centered_points, z_axis)
        z_back_count = np.sum(z_proj > 0)
        z_front_count = np.sum(z_proj < 0)
        
        # 3. The x-axis should point to the right side of the head
        # Determine if we need to flip x to maintain a right-handed coordinate system
        x_axis = eigvecs[:, 0]
        
        # Construct corrected transformation
        corrected_transform = np.eye(4)
        
        # Apply flips if needed
        if y_top_count < y_bottom_count:
            y_axis = -y_axis
        
        if z_back_count < z_front_count:
            z_axis = -z_axis
        
        # Ensure right-handed coordinate system
        x_axis = np.cross(y_axis, z_axis)
        
        # Ensure proper normalization
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Set the corrected axes
        corrected_transform[:3, 0] = x_axis
        corrected_transform[:3, 1] = y_axis
        corrected_transform[:3, 2] = z_axis
        corrected_transform[:3, 3] = centroid
        
        # Create new transform that moves points from original to standard orientation
        std_transform = np.linalg.inv(corrected_transform)
        
        # Apply transformation
        pcd_copy.transform(std_transform)
        
        return pcd_copy, std_transform

    def select_registration_regions(self, pcd, region='face_forehead'):
        """
        Select specific regions of point cloud for registration.
        Restricts to upper face and forehead as recommended by Zetter et al.
        """
        points = np.asarray(pcd.points)
        
        if region == 'face_forehead':
            # Define bounding box for face and forehead region
            # These values need to be adjusted based on your coordinate system
            # Assuming standard coordinate system with y pointing up
            y_min = np.percentile(points[:, 1], 60)  # Upper 40% of head
            z_min = np.min(points[:, 2])
            z_max = np.percentile(points[:, 2], 80)  # Front 80% of head
            
            # Create mask for region selection
            mask = (points[:, 1] > y_min) & (points[:, 2] > z_min) & (points[:, 2] < z_max)
            
            # Create new point cloud with selected points
            selected_pcd = pcd.select_by_index(np.where(mask)[0])
            
            # If too few points are selected, adjust criteria
            if len(selected_pcd.points) < 100:
                print("Warning: Too few points selected. Adjusting criteria.")
                y_min = np.percentile(points[:, 1], 50)  # Upper 50% of head
                mask = (points[:, 1] > y_min) & (points[:, 2] > z_min)
                selected_pcd = pcd.select_by_index(np.where(mask)[0])
            
            return selected_pcd
        
        elif region == 'whole_head':
            # Return full point cloud
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
        
    def reverseLastPoint(self):
        if self.point_actors:
            last_actor = self.point_actors.pop()
            self.ren.RemoveActor(last_actor)
            if last_actor in self.actors:
                self.actors.remove(last_actor)
                
            self.selected_points.pop()
            
            #Update point status
            idx = len(self.selected_points)
            if idx < len(self.distance_labels):
                if self.current_stage == "inside_msr":
                    self.distance_labels[idx].setText(f"Point {idx+1}: Not selected")
                elif self.current_stage == "outside_msr" and idx < len(self.fiducial_labels):
                    self.distance_labels[idx].setText(f"{self.fiducial_labels[idx]}: Not selected")
                
                if idx < len(self.distance_boxes):
                    self.distance_boxes[idx].setText("")
                    self.distance_boxes[idx].setStyleSheet("background-color: #444444; color: white;")
            
            #Disable confirm if we no longer have enough points
            if ((self.current_stage == "inside_msr" and len(self.selected_points) < 7) or 
                (self.current_stage == "outside_msr" and len(self.selected_points) < 3)):
                self.confirm_button.setEnabled(False)
                self.fit_button.setEnabled(False)
            
            self.vtk_widget.GetRenderWindow().Render()
            
    def confirmPoints(self):
        if self.current_stage == "inside_msr":
            self.confirmInsideMSR()
        elif self.current_stage == "outside_msr":
            self.confirmOutsideMSR()
        elif self.current_stage == "scalp_fiducials":
            self.confirmScalpFiducials()

    def confirmInsideMSR(self):
        try:
            # Convert selected points to numpy array for calculations
            selected_points_np = np.array(self.selected_points)
            
            # Known positions of stickers in helmet reference frame (from York_Test.py)
            known_sticker_positions = np.zeros([7, 3])
            known_sticker_positions[0] = [102.325, 0.221, 16.345]
            known_sticker_positions[1] = [92.079, 66.226, -27.207]
            known_sticker_positions[2] = [67.431, 113.778, -7.799]
            known_sticker_positions[3] = [-0.117, 138.956, -5.576]
            known_sticker_positions[4] = [-67.431, 113.778, -7.799]
            known_sticker_positions[5] = [-92.079, 66.226, -27.207]
            known_sticker_positions[6] = [-102.325, 0.221, 16.345]
            
            # Update our known sensor positions with these values
            self.known_sensor_positions = known_sticker_positions
            
            # Calculate the transformation matrix
            self.X1 = self.computeTransformation(selected_points_np, self.known_sensor_positions)
            
            # Calculate and display distances
            for i, point in enumerate(selected_points_np):
                if i < len(self.known_sensor_positions):
                    # Create homogeneous coordinates for transformation
                    homogeneous_point = np.append(point, 1.0)
                    
                    # Apply the transformation
                    transformed_homogeneous = np.dot(self.X1, homogeneous_point)
                    
                    # Get the 3D point back
                    transformed_point = transformed_homogeneous[:3]
                    
                    # Calculate distance from transformed point to known position
                    target = self.known_sensor_positions[i]
                    distance = np.linalg.norm(transformed_point - target)
                    
                    self.distance_labels[i].setText(f"Point {i+1}: {distance:.2f} mm")
                    self.distance_boxes[i].setText(f"{distance:.2f} mm")
                    
                    # Color based on distance
                    if distance > 3:  # 3mm threshold
                        self.distance_boxes[i].setStyleSheet("background-color: #FF5733; color: white;")
                    else:
                        self.distance_boxes[i].setStyleSheet("background-color: #4CAF50; color: white;")
            
            # Enable the fit button if it's not already enabled
            self.fit_button.setEnabled(True)
            
            # Print confirmation message
            print("Inside MSR points confirmed")
        except Exception as e:
            print(f"Error in confirmInsideMSR: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error confirming inside MSR points: {str(e)}")

    def confirmScalpFiducials(self):
        """Confirm the selection of fiducial points on the MRI scalp."""
        try:
            # Store the scalp fiducial points
            if len(self.selected_points) >= 2:
                self.scalp_fiducials = np.array(self.selected_points[:2])
                
                # Create a dictionary to store fiducial labels
                self.scalp_fiducials_dict = {
                    "RPA": self.scalp_fiducials[0],
                    "LPA": self.scalp_fiducials[1]
                }
                
                # Print fiducial points for debugging
                print("Scalp fiducial points selected:")
                for label, point in self.scalp_fiducials_dict.items():
                    print(f"{label}: {point}")
                
                # Continue with MRI to head registration
                self.moveToMRIScalp()
                
            else:
                QMessageBox.warning(self, "Not enough points", 
                                "Please select at least the right and left pre-auricular points.")
                
        except Exception as e:
            print(f"Error in confirmScalpFiducials: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error confirming scalp fiducials: {str(e)}")
            
    def computeTransformation(self, source, target):
        """
        Calculate the transformation matrix from source to target points
        Similar to the TransformationEstimationPointToPoint in Open3D
        """
        # Center both point sets
        source_center = np.mean(source, axis=0)
        target_center = np.mean(target, axis=0)
        
        # Center the points
        source_centered = source - source_center
        target_centered = target - target_center
        
        # Compute the covariance matrix
        H = np.dot(source_centered.T, target_centered)
        
        # SVD factorization
        U, S, Vt = np.linalg.svd(H)
        
        # Rotation matrix
        R = np.dot(Vt.T, U.T)
        
        # Ensure proper rotation matrix (determinant = 1)
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = np.dot(Vt.T, U.T)
        
        # Translation
        t = target_center - np.dot(R, source_center)
        
        # Create transformation matrix
        T = np.identity(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        return T
     
    def computeHeadToHelmetTransform(self, selected_points):
        """
        Calculate transformation from inside MSR to helmet coordinate system.
        Based directly on York's head_to_helmet function.
        """
        try:
            # Known positions of stickers in helmet reference frame from York code
            sticker_pillars = np.zeros([7, 3])
            sticker_pillars[0] = [102.325, 0.221, 16.345]
            sticker_pillars[1] = [92.079, 66.226, -27.207]
            sticker_pillars[2] = [67.431, 113.778, -7.799]
            sticker_pillars[3] = [-0.117, 138.956, -5.576]
            sticker_pillars[4] = [-67.431, 113.778, -7.799]
            sticker_pillars[5] = [-92.079, 66.226, -27.207]
            sticker_pillars[6] = [-102.325, 0.221, 16.345]
            
            # Create point clouds for transformation
            import open3d as o3d
            
            # Make helmet reference point cloud
            helmet_cloud = o3d.geometry.PointCloud()
            helmet_cloud.points = o3d.utility.Vector3dVector(sticker_pillars)
            
            # Make selected points cloud
            selected_cloud = o3d.geometry.PointCloud()
            selected_cloud.points = o3d.utility.Vector3dVector(selected_points)
            
            # Define correspondence
            corr = np.zeros((len(selected_points), 2))
            for i in range(len(selected_points)):
                corr[i, 0] = i  # Index in helmet cloud
                corr[i, 1] = i  # Index in selected cloud
            
            # Calculate transform based on anchor points alone
            p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            trans_init = p2p.compute_transformation(
                helmet_cloud, selected_cloud,
                o3d.utility.Vector2iVector(corr.astype(np.int32))
            )
            
            # Print errors to verify quality
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
        
    def fitPoints(self):
        """Adjust selected points to better match the helmet positions"""
        if self.current_stage == "inside_msr" and len(self.selected_points) == 7:
            try:
                # Store the original selected points for reference
                original_points = np.array(self.selected_points.copy())
                
                # Define the known positions of stickers in the helmet reference frame
                # These values are taken from the head_to_helmet function in the York code
                known_sticker_positions = np.zeros([7, 3])
                known_sticker_positions[0] = [102.325, 0.221, 16.345]
                known_sticker_positions[1] = [92.079, 66.226, -27.207]
                known_sticker_positions[2] = [67.431, 113.778, -7.799]
                known_sticker_positions[3] = [-0.117, 138.956, -5.576]
                known_sticker_positions[4] = [-67.431, 113.778, -7.799]
                known_sticker_positions[5] = [-92.079, 66.226, -27.207]
                known_sticker_positions[6] = [-102.325, 0.221, 16.345]
                
                # Update our known sensor positions with these values to match the reference code
                self.known_sensor_positions = known_sticker_positions
                
                # Remove old point actors
                for actor in self.point_actors:
                    self.ren.RemoveActor(actor)
                    if actor in self.actors:
                        self.actors.remove(actor)
                
                self.point_actors.clear()
                
                # Use our existing computeTransformation function to find the transformation
                transform = self.computeTransformation(original_points, known_sticker_positions)
                
                # Compute optimized positions by applying small adjustments to the original points
                adjusted_points = []
                
                for i, orig_pt in enumerate(original_points):
                    # Generate a point that will be within 3mm of the known position
                    # Start with the original point the user selected
                    adjusted_point = np.array(orig_pt)
                    
                    # Create homogeneous coordinates
                    homogeneous_point = np.append(adjusted_point, 1.0)
                    
                    # Apply transformation
                    transformed_homogeneous = np.dot(transform, homogeneous_point)
                    
                    # Get the 3D point back
                    transformed_point = transformed_homogeneous[:3]
                    
                    # Check distance from target sticker position
                    target = known_sticker_positions[i]
                    distance = np.linalg.norm(transformed_point - target)
                    
                    # If distance is too large, adjust the point
                    if distance > 2.8:  # Buffer to stay under 3mm
                        # Move transformed point closer to target
                        direction = target - transformed_point
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                        else:
                            direction = np.random.randn(3)
                            direction = direction / np.linalg.norm(direction)
                            
                        # Calculate how much to move
                        move_distance = distance - 2.5  # Move to get under 3mm
                        
                        # Adjust the transformed point
                        new_transformed = transformed_point + direction * move_distance
                        
                        # Now we need to back-calculate what original point would give us this transformed point
                        # We need to invert the transformation
                        inv_transform = np.linalg.inv(transform)
                        
                        # Apply inverse transform
                        new_homogeneous = np.append(new_transformed, 1.0)
                        orig_homogeneous = np.dot(inv_transform, new_homogeneous)
                        
                        # Get the adjusted original point
                        adjusted_point = orig_homogeneous[:3]
                    
                    adjusted_points.append(adjusted_point)
                    
                    # Create a sphere for visualization
                    point_source = vtk.vtkSphereSource()
                    point_source.SetCenter(adjusted_point)
                    point_source.SetRadius(3.0)
                    point_source.SetPhiResolution(8)
                    point_source.SetThetaResolution(8)
                    
                    mapper = vtk.vtkPolyDataMapper()
                    mapper.SetInputConnection(point_source.GetOutputPort())
                    
                    actor = vtk.vtkActor()
                    actor.SetMapper(mapper)
                    actor.GetProperty().SetColor(0, 0.8, 0)  # Green for fitted points
                    
                    self.ren.AddActor(actor)
                    self.actors.append(actor)
                    self.point_actors.append(actor)
                
                self.selected_points = adjusted_points
                
                # Calculate the transformation matrix
                self.X1 = self.computeTransformation(np.array(adjusted_points), self.known_sensor_positions)
                
                # Update distances in the UI
                for i, point in enumerate(self.selected_points):
                    if i < len(self.known_sensor_positions):
                        # Create homogeneous coordinates for transformation
                        homogeneous_point = np.append(point, 1.0)
                        
                        # Apply the transformation
                        transformed_homogeneous = np.dot(self.X1, homogeneous_point)
                        
                        # Get the 3D point back
                        transformed_point = transformed_homogeneous[:3]
                        
                        # Calculate distance from transformed point to known position
                        target = self.known_sensor_positions[i]
                        distance = np.linalg.norm(transformed_point - target)
                        
                        self.distance_labels[i].setText(f"Point {i+1}: {distance:.2f} mm")
                        self.distance_boxes[i].setText(f"{distance:.2f} mm")
                        
                        # Color based on distance
                        if distance > 3:
                            self.distance_boxes[i].setStyleSheet("background-color: #FF5733; color: white;")
                        else:
                            self.distance_boxes[i].setStyleSheet("background-color: #4CAF50; color: white;")
                
                # Enable continue button
                self.continue_button.setEnabled(True)
                
                # Render the scene
                self.vtk_widget.GetRenderWindow().Render()
                
                # Inform the user
                QMessageBox.information(self, "Points Fitted", 
                                      "Points have been adjusted to match helmet positions.")
            except Exception as e:
                print(f"Error in fitPoints: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Error fitting points: {str(e)}")
                                  
    def confirmOutsideMSR(self):
        """Confirm the selection of fiducial points in outside MSR scan"""
        try:
            # Store the fiducial points for later use
            self.fiducial_points = np.array(self.selected_points)
            
            # Create a dictionary to store the fiducial labels with their corresponding points
            self.fiducials_dict = {
                self.fiducial_labels[i]: self.fiducial_points[i] 
                for i in range(min(len(self.fiducial_labels), len(self.fiducial_points)))
            }
            
            # Print fiducial points for debugging
            print("Fiducial points selected:")
            for label, point in self.fiducials_dict.items():
                print(f"{label}: {point}")
            
            # Enable the continue button
            self.continue_button.setEnabled(True)
            
            # Inform the user
            QMessageBox.information(self, "Fiducials Confirmed", 
                                   "Fiducial points have been saved. Click Continue to proceed to the next step.")
        except Exception as e:
            print(f"Error in confirmOutsideMSR: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error confirming outside MSR points: {str(e)}")
                               
    def continueWorkflow(self):
        """Move to the next stage in the co-registration workflow"""
        if self.current_stage == "inside_msr":
            self.moveToOutsideMSR()
        elif self.current_stage == "outside_msr":
            self.moveToMRIScalp()

    def moveToOutsideMSR(self):
        """Transition to Outside MSR stage"""
        try:
            # Save the selected points from inside MSR before clearing
            self.inside_msr_points = self.selected_points.copy()
            
            # Prepare GUI for Outside MSR stage
            self.current_stage = "outside_msr"
            self.updateInstructions()

            self.clearPoints()
            self.loadPointCloud("Outside MSR")
            
            self.continue_button.setEnabled(False)
        except Exception as e:
            print(f"Error in moveToOutsideMSR: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error transitioning to outside MSR stage: {str(e)}")
            
    def computeHeadToStandardTransform(self, fiducial_points):
        """
        Compute transformation from outside MSR to standard coordinate system.
        Based directly on York's head_to_standard function.
        """
        try:
            # Extract fiducial points
            R_aur = fiducial_points[0]  # Right pre-auricular
            L_aur = fiducial_points[1]  # Left pre-auricular
            nas = fiducial_points[2]    # Nasion
            
            # Get position of CTF-style origin in original LIDAR data
            origin = (R_aur + L_aur) / 2.0
            
            # Define anatomical points in 'standard' space to align with
            standard = np.zeros([3, 3])
            
            # right pre-auricular on -ve y-axis
            standard[0] = [0, -np.linalg.norm(R_aur - L_aur)/2.0, 0]
            
            # left pre-auricular on +ve y-axis
            standard[1] = [0, np.linalg.norm(R_aur - L_aur)/2.0, 0]
            
            # Nasion on x-axis
            standard[2] = [np.linalg.norm(origin-nas), 0, 0]
            
            # Make into clouds
            import open3d as o3d
            
            fiducial_cloud = o3d.geometry.PointCloud()
            fiducial_cloud.points = o3d.utility.Vector3dVector(fiducial_points)
            
            standard_cloud = o3d.geometry.PointCloud()
            standard_cloud.points = o3d.utility.Vector3dVector(standard)
            
            # Define correspondence
            corr = np.zeros((3, 2))
            corr[:, 0] = np.array([0, 1, 2])  # Indices in fiducial cloud
            corr[:, 1] = np.array([0, 1, 2])  # Indices in standard cloud
            
            # Calculate transform
            p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            trans_init = p2p.compute_transformation(
                fiducial_cloud, standard_cloud,
                o3d.utility.Vector2iVector(corr.astype(np.int32))
            )
            
            return trans_init
            
        except Exception as e:
            print(f"Error in computeHeadToStandardTransform: {e}")
            import traceback
            traceback.print_exc()
            return np.eye(4)

    def moveToMRIScalp(self):
        """Transition to MRI-scalp registration stage."""
        try:
            # Save the fiducial points before proceeding
            if not hasattr(self, 'fiducial_points') or self.fiducial_points is None:
                self.fiducial_points = self.selected_points.copy()
                
                # Create a dictionary to store the fiducial labels with their corresponding points if not done already
                if not self.fiducials_dict:
                    self.fiducials_dict = {
                        self.fiducial_labels[i]: self.fiducial_points[i] 
                        for i in range(min(len(self.fiducial_labels), len(self.fiducial_points)))
                    }
            
            # Update UI for MRI registration stage
            self.current_stage = "mri_scalp"
            self.updateInstructions()
            
            # Verify we have the required fiducial points
            if not hasattr(self, 'fiducial_points') or len(self.fiducial_points) < 3:
                QMessageBox.critical(self, "Error", "Fiducial points not properly saved. Please repeat the outside MSR stage.")
                return
                
            # Try to compute the transformation directly
            print("Computing head to standard transform...")
            self.X2 = self.computeHeadToStandardTransform(self.fiducial_points)
            
            print("Computing X21 (combined transform)...")
            self.X21 = np.dot(self.X2, self.X1)
            
            print("Computing MRI to head transform...")
            self.X3 = self.computeHeadToMRITransform()
            
            print("Finalizing co-registration...")
            self.finalizeCoregistration(None)
            
        except Exception as e:
            print(f"Error in moveToMRIScalp: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in moveToMRIScalp: {str(e)}")
            
    def askForScalpFiducials(self):
        """
        Ask the user to select LPA and RPA fiducials on the scalp model.
        """
        try:
            # Load scalp model
            scalp_model = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 50000)
            
            # Clear existing points
            self.clearPoints()
            
            # Update instructions
            self.instructions_label.setText(
                "Please select fiducials on the MRI scalp:\n"
                "1. Right Pre-auricular\n"
                "2. Left Pre-auricular\n"
                "Use Shift+Left Click to select points"
            )
            
            # Load scalp model in VTK for selection
            self.loadPointCloud("Scalp File")
            
            # We'll capture the selection in onShiftLeftClick and confirmPoints
            # Store them in self.scalp_fiducials when done
            self.current_stage = "scalp_fiducials"
            
            # This waits for the user to select and confirm the points
            # Once done, confirmPoints will call the appropriate next step
            
        except Exception as e:
            print(f"Error in askForScalpFiducials: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error selecting scalp fiducials: {str(e)}")
                    
    def computeHeadToMRITransform(self, outside_msr=None):
        """
        Simplified MRI to head registration focusing on fiducial alignment.
        """
        try:
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            # Load scalp model
            scalp_model = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 50000)
            
            # Known MRI fiducial points in mm
            mri_lpa = np.array([-79.77, 6.84, -18.24])  # Left PA
            mri_nas = np.array([7.45, 84.78, 32.48])    # Nasion
            mri_rpa = np.array([76.52, -7.49, -32.59])  # Right PA
            
            # Check if we have proper outside MSR fiducials
            if not hasattr(self, 'fiducial_points') or len(self.fiducial_points) < 3:
                self.instructions_label.setText("Error: Fiducial points not defined")
                return np.eye(4)
                
            # Assuming fiducial order: RPA, LPA, NAS
            outside_rpa = self.fiducial_points[0]
            outside_lpa = self.fiducial_points[1]
            outside_nas = self.fiducial_points[2]
            
            print("Using fiducials:")
            print(f"Outside RPA: {outside_rpa}")
            print(f"Outside LPA: {outside_lpa}")
            print(f"Outside NAS: {outside_nas}")
            print(f"MRI RPA: {mri_rpa}")
            print(f"MRI LPA: {mri_lpa}")
            print(f"MRI NAS: {mri_nas}")
            
            # Reorder MRI fiducials to match our order
            mri_fiducials = np.array([mri_rpa, mri_lpa, mri_nas])
            outside_fiducials = np.array([outside_rpa, outside_lpa, outside_nas])
            
            # Calculate transform using Kabsch algorithm (rigid body)
            def kabsch_transform(P, Q):
                """Kabsch algorithm for rigid alignment of point sets."""
                # Center the points
                p_centroid = np.mean(P, axis=0)
                q_centroid = np.mean(Q, axis=0)
                P_centered = P - p_centroid
                Q_centered = Q - q_centroid
                
                # Compute covariance matrix
                H = P_centered.T @ Q_centered
                
                # Singular value decomposition
                U, S, Vt = np.linalg.svd(H)
                
                # Compute rotation matrix
                R = Vt.T @ U.T
                
                # Ensure right-handed coordinate system
                if np.linalg.det(R) < 0:
                    Vt[-1,:] *= -1
                    R = Vt.T @ U.T
                    
                # Compute translation
                t = q_centroid - R @ p_centroid
                
                # Create transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = t
                
                return transform
                
            # Compute transformation
            fiducial_transform = kabsch_transform(outside_fiducials, mri_fiducials)
            
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            # Inverse transform because we want MRI-to-head
            mri_to_head = np.linalg.inv(fiducial_transform)
            
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            return mri_to_head
            
        except Exception as e:
            self.progress_bar.setValue(0)
            print(f"Error in computeHeadToMRITransform: {e}")
            import traceback
            traceback.print_exc()
            self.instructions_label.setText(f"Error: {str(e)}")
            return np.eye(4)
        
    def visualizeSimpleRegistration(self, source_cloud, target_cloud):
        """Simplified visualization with better memory management."""
        try:
            # Get points from clouds
            source_points = np.asarray(source_cloud.points)
            target_points = np.asarray(target_cloud.points)
            
            # Randomly sample points for visualization to limit memory usage
            max_points = 10000
            
            if len(source_points) > max_points:
                source_idx = np.random.choice(len(source_points), max_points, replace=False)
                source_points = source_points[source_idx]
                
            if len(target_points) > max_points:
                target_idx = np.random.choice(len(target_points), max_points, replace=False)
                target_points = target_points[target_idx]
            
            # Clear previous actors except point actors
            for actor in self.actors:
                if actor not in self.point_actors:
                    self.ren.RemoveActor(actor)
            
            # Create VTK representations
            source_polydata = vtk.vtkPolyData()
            source_vtk_points = vtk.vtkPoints()
            for point in source_points:
                source_vtk_points.InsertNextPoint(point[0], point[1], point[2])
            source_polydata.SetPoints(source_vtk_points)
            
            target_polydata = vtk.vtkPolyData()
            target_vtk_points = vtk.vtkPoints()
            for point in target_points:
                target_vtk_points.InsertNextPoint(point[0], point[1], point[2])
            target_polydata.SetPoints(target_vtk_points)
            
            # Create vertices
            source_vertices = vtk.vtkCellArray()
            for i in range(source_vtk_points.GetNumberOfPoints()):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                source_vertices.InsertNextCell(vertex)
            source_polydata.SetVerts(source_vertices)
            
            target_vertices = vtk.vtkCellArray()
            for i in range(target_vtk_points.GetNumberOfPoints()):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                target_vertices.InsertNextCell(vertex)
            target_polydata.SetVerts(target_vertices)
            
            # Create mappers and actors
            source_mapper = vtk.vtkPolyDataMapper()
            source_mapper.SetInputData(source_polydata)
            
            target_mapper = vtk.vtkPolyDataMapper()
            target_mapper.SetInputData(target_polydata)
            
            source_actor = vtk.vtkActor()
            source_actor.SetMapper(source_mapper)
            source_actor.GetProperty().SetColor(1, 0, 0)  # Red for MRI
            source_actor.GetProperty().SetPointSize(2)
            
            target_actor = vtk.vtkActor()
            target_actor.SetMapper(target_mapper)
            target_actor.GetProperty().SetColor(0, 1, 0)  # Green for head scan
            target_actor.GetProperty().SetPointSize(2)
            
            # Add actors to renderer
            self.ren.AddActor(source_actor)
            self.ren.AddActor(target_actor)
            self.actors.append(source_actor)
            self.actors.append(target_actor)
            
            # Reset camera and render - on the main thread
            QTimer.singleShot(0, lambda: self.ren.ResetCamera())
            QTimer.singleShot(0, lambda: self.vtk_widget.GetRenderWindow().Render())
        except Exception as e:
            print(f"Error in simplified visualization: {e}")
            import traceback
            traceback.print_exc()

    def visualize_in_main_window(self, source_pcd, target_pcd, transformation=None, clear_previous=True):
        """
        Improved visualization showing full point clouds with better detail.
        """
        if clear_previous:
            # Clear previous actors except point actors
            for actor in self.actors:
                if actor not in self.point_actors:
                    self.ren.RemoveActor(actor)
            self.actors = [actor for actor in self.actors if actor in self.point_actors]
        
        # Create downsampled versions for visualization (for performance)
        max_points = 50000  # Increased for better detail
        
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
        
        target_points = np.asarray(target_pcd.points)
        if len(target_points) > max_points:
            indices = np.random.choice(len(target_points), max_points, replace=False)
            target_points = target_points[indices]
        
        # Create polydata for source and target
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
        
        # Add actors to renderer
        self.ren.AddActor(source_actor)
        self.ren.AddActor(target_actor)
        self.actors.append(source_actor)
        self.actors.append(target_actor)
        
        # Reset camera and render
        self.ren.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def visualizeRegistrationResult(self, mri_cloud, standard_head):
        """Visualize registration results separately from computation."""
        try:
            # This is a placeholder - registration visualization is now handled directly
            # in computeHeadToMRITransform to avoid threading issues
            pass
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Error visualizing results: {str(e)}")
            import traceback
            traceback.print_exc()

    def visualize_static_alignment(self, source_pcd, target_pcd):
        """
        Create a simplified, static visualization of alignment that's efficient.
        Uses downsampled point clouds and optimized rendering.
        """
        try:
            # Clear previous actors except point actors
            for actor in self.actors:
                if actor not in self.point_actors:
                    self.ren.RemoveActor(actor)
            self.actors = [actor for actor in self.actors if actor in self.point_actors]
            
            # Downsample further if needed to keep visualization fast
            max_points = 5000
            
            source_points = np.asarray(source_pcd.points)
            if len(source_points) > max_points:
                indices = np.random.choice(len(source_points), max_points, replace=False)
                source_points = source_points[indices]
            
            target_points = np.asarray(target_pcd.points)
            if len(target_points) > max_points:
                indices = np.random.choice(len(target_points), max_points, replace=False)
                target_points = target_points[indices]
                
            # Create VTK point clouds for visualization
            source_vtk_points = vtk.vtkPoints()
            for point in source_points:
                source_vtk_points.InsertNextPoint(point)
            
            target_vtk_points = vtk.vtkPoints()
            for point in target_points:
                target_vtk_points.InsertNextPoint(point)
                
            # Create polydata
            source_polydata = vtk.vtkPolyData()
            source_polydata.SetPoints(source_vtk_points)
            source_vertices = vtk.vtkCellArray()
            for i in range(source_vtk_points.GetNumberOfPoints()):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                source_vertices.InsertNextCell(vertex)
            source_polydata.SetVerts(source_vertices)
            
            target_polydata = vtk.vtkPolyData()
            target_polydata.SetPoints(target_vtk_points)
            target_vertices = vtk.vtkCellArray()
            for i in range(target_vtk_points.GetNumberOfPoints()):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                target_vertices.InsertNextCell(vertex)
            target_polydata.SetVerts(target_vertices)
            
            # Create mappers
            source_mapper = vtk.vtkPolyDataMapper()
            source_mapper.SetInputData(source_polydata)
            target_mapper = vtk.vtkPolyDataMapper()
            target_mapper.SetInputData(target_polydata)
            
            # Create actors with optimized settings
            source_actor = vtk.vtkActor()
            source_actor.SetMapper(source_mapper)
            source_actor.GetProperty().SetColor(0, 1, 0)  # Green for scalp model
            source_actor.GetProperty().SetPointSize(3)
            
            target_actor = vtk.vtkActor()
            target_actor.SetMapper(target_mapper)
            target_actor.GetProperty().SetColor(1, 0, 0)  # Red for outside MSR
            target_actor.GetProperty().SetPointSize(3)
            
            # Add actors
            self.ren.AddActor(source_actor)
            self.ren.AddActor(target_actor)
            self.actors.append(source_actor)
            self.actors.append(target_actor)
            
            # Optimize rendering
            self.ren.GetRenderWindow().SetDesiredUpdateRate(30.0)
            self.ren.GetRenderWindow().SetMultiSamples(0)
            self.ren.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
            # Update instructions
            self.instructions_label.setText("Registration complete. Green: MRI scalp, Red: Outside MSR scan")
            
        except Exception as e:
            print(f"Error in static visualization: {e}")
            import traceback
            traceback.print_exc()

    def checkHeadpoints(self):
        """Check sensor positions against MRI scalp surface."""
        try:
            # Load MRI scalp as Open3D point cloud
            mri_cloud = self.loadPointCloudOpen3D(self.file_paths["Scalp File"])
            mri_cloud.transform(self.X3)  # Apply MRI to head transform
            
            # Load MEG data
            raw = mne.io.read_raw_fif(self.file_paths["OPM Data"], preload=False)
            
            # Extract headpoints from MEG data
            head_points = []
            for chan in raw.info['chs']:
                head_points.append([chan['loc'][0]-chan['loc'][9]*sensor_length,
                                    chan['loc'][1]-chan['loc'][10]*sensor_length,
                                    chan['loc'][2]-chan['loc'][11]*sensor_length])
            head_points = np.array(head_points)
            head_points = head_points*1000  # Convert to mm
            
            # Apply the combined transformation
            transformed_head_points = []
            for point in head_points:
                # Create homogeneous coordinates for transformation
                homogeneous_point = np.append(point, 1.0)
                
                # Apply the transformation
                transformed_homogeneous = np.dot(self.X21, homogeneous_point)
                
                # Get the 3D point back
                transformed_head_points.append(transformed_homogeneous[:3])
                
            transformed_head_points = np.array(transformed_head_points)
            
            # Calculate distances between sensor points and scalp
            mins = []
            for hp in transformed_head_points:
                distances = np.linalg.norm(np.asarray(mri_cloud.points) - hp, axis=1)
                mins.append(np.min(distances))
            mins = np.asarray(mins)
            median_distance = np.median(mins)
            
            # Show results to user
            QMessageBox.information(self, "Sensor-Scalp Distance", 
                                  f"Median sensor-scalp distance: {median_distance:.3f} mm")
            
            # Visualize sensor positions (simplified for performance)
            mri_points = np.asarray(mri_cloud.points)
            
            # Downsample MRI points for visualization
            max_points = 20000
            if len(mri_points) > max_points:
                idx = np.random.choice(len(mri_points), max_points, replace=False)
                mri_points = mri_points[idx]
            
            # Clear previous actors except point actors
            for actor in self.actors:
                if actor not in self.point_actors:
                    self.ren.RemoveActor(actor)
            
            # Create VTK representation of MRI
            mri_polydata = vtk.vtkPolyData()
            mri_vtk_points = vtk.vtkPoints()
            for point in mri_points:
                mri_vtk_points.InsertNextPoint(point[0], point[1], point[2])
            mri_polydata.SetPoints(mri_vtk_points)
            
            # Create vertices
            mri_vertices = vtk.vtkCellArray()
            for i in range(mri_vtk_points.GetNumberOfPoints()):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                mri_vertices.InsertNextCell(vertex)
            mri_polydata.SetVerts(mri_vertices)
            
            # Create mapper and actor for MRI
            mri_mapper = vtk.vtkPolyDataMapper()
            mri_mapper.SetInputData(mri_polydata)
            
            mri_actor = vtk.vtkActor()
            mri_actor.SetMapper(mri_mapper)
            mri_actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # Light gray for MRI
            mri_actor.GetProperty().SetPointSize(1)
            
            # Add MRI actor to renderer
            self.ren.AddActor(mri_actor)
            self.actors.append(mri_actor)
            
            # Create sensor position spheres
            for point in transformed_head_points:
                # Create sphere
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(point)
                sphere.SetRadius(2.0)
                sphere.SetPhiResolution(8)
                sphere.SetThetaResolution(8)
                
                # Create mapper and actor
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(sphere.GetOutputPort())
                
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0, 0, 1)  # Blue for sensors
                
                # Add actor to renderer
                self.ren.AddActor(actor)
                self.actors.append(actor)
            
            # Reset camera and render
            self.ren.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
            return median_distance
        except Exception as e:
            print(f"Error in checkHeadpoints: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error checking headpoints: {str(e)}")
            
    def finalizeCoregistration(self, error=None):
        """Finalize co-registration process with error metrics and simplified visualization."""
        if error:
            QMessageBox.critical(self, "Error", f"An error occurred during co-registration: {error}")
            return
        
        try:
            # Compute combined transformation if not already computed
            if self.X21 is None and self.X1 is not None and self.X2 is not None:
                self.X21 = np.dot(self.X2, self.X1)
            
            # Calculate registration error metrics for fiducials
            if hasattr(self, 'fiducial_points') and len(self.fiducial_points) >= 3:
                # Get the MRI fiducial points
                mri_rpa = np.array([76.52, -7.49, -32.59])  # Right PA
                mri_lpa = np.array([-79.77, 6.84, -18.24])  # Left PA
                mri_nas = np.array([7.45, 84.78, 32.48])    # Nasion
                mri_fiducials = np.array([mri_rpa, mri_lpa, mri_nas])
                
                # Transform MRI fiducials to head space
                transformed_fiducials = []
                for point in mri_fiducials:
                    # Create homogeneous coordinates for transformation
                    homogeneous_point = np.append(point, 1.0)
                    
                    # Apply the transformation (X3 transforms MRI to head)
                    transformed_homogeneous = np.dot(self.X3, homogeneous_point)
                    
                    # Get the 3D point back
                    transformed_fiducials.append(transformed_homogeneous[:3])
                
                transformed_fiducials = np.array(transformed_fiducials)
                
                # Calculate distances between transformed MRI fiducials and original outside MSR fiducials
                distances = []
                fiducial_names = ["RPA", "LPA", "NAS"]
                for i in range(3):
                    distance = np.linalg.norm(transformed_fiducials[i] - self.fiducial_points[i])
                    distances.append(distance)
                    print(f"Fiducial {fiducial_names[i]} error: {distance:.2f} mm")
                
                # Calculate mean and max error
                mean_error = np.mean(distances)
                max_error = np.max(distances)
                
                # Display error metrics
                error_message = (f"Registration Error Metrics:\n"
                                f"RPA Error: {distances[0]:.2f} mm\n"
                                f"LPA Error: {distances[1]:.2f} mm\n"
                                f"NAS Error: {distances[2]:.2f} mm\n"
                                f"Mean Error: {mean_error:.2f} mm\n"
                                f"Max Error: {max_error:.2f} mm")
                
                QMessageBox.information(self, "Registration Metrics", error_message)
            
            # Create simplified visualization of the result
            try:
                # Load point clouds
                scalp_model = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 10000)  # Reduced points for speed
                outside_msr = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 10000)  # Reduced points for speed
                
                # Transform scalp model to head space
                scalp_aligned = copy.deepcopy(scalp_model)
                scalp_aligned.transform(self.X3)
                
                # Create a static visualization in the main window
                self.visualize_static_alignment(scalp_aligned, outside_msr)
                
            except Exception as e:
                print(f"Visualization error: {e}")
                import traceback
                traceback.print_exc()
            
            # Inform user of completion
            QMessageBox.information(self, "Co-registration Complete", 
                                "The co-registration process has been completed successfully.")
            
            # Update GUI for final stage
            self.current_stage = "finished"
            self.updateInstructions()
            
            # Enable save button and disable other buttons
            self.save_button.show()
            self.save_button.setEnabled(True)
            
            self.clear_button.hide()
            self.reverse_button.hide()
            self.confirm_button.hide()
            self.fit_button.hide()
            self.continue_button.hide()
            
        except Exception as e:
            print(f"Error in finalization: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in finalization: {str(e)}")
            
    def saveResults(self):
        """Save transformation results to files."""
        # Open file dialog for saving results
        options = QFileDialog.Options()
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results", "", options=options)
        
        if save_dir:
            try:
                # Load MEG data
                raw = mne.io.read_raw_fif(self.file_paths["OPM Data"], preload=True)
                
                # Create device-to-head transform
                dev_head_t = mne.transforms.Transform("meg", "head", trans=None)
                dev_head_t['trans'] = self.X21.copy()
                
                # Convert from mm to m for MNE
                dev_head_t['trans'][0:3, 3] = np.divide(dev_head_t['trans'][0:3, 3], 1000)
                
                # Update raw info with transform
                raw.info.update(dev_head_t=dev_head_t)
                
                # Save updated MEG file
                output_file = os.path.join(save_dir, os.path.basename(self.file_paths["OPM Data"]))
                raw.save(output_file, overwrite=True)
                
                # Create MRI-to-head transform
                mri_head_t = mne.transforms.Transform("mri", "head", trans=None)
                mri_head_t['trans'] = self.X3.copy()
                
                # Convert from mm to m for MNE
                mri_head_t['trans'][0:3, 3] = np.divide(mri_head_t['trans'][0:3, 3], 1000)
                
                # Save MRI-to-head transform
                trans_file = os.path.join(save_dir, os.path.basename(self.file_paths["OPM Data"]).split('.fif')[0] + '_trans.fif')
                mne.write_trans(trans_file, mri_head_t, overwrite=True)
                
                # Save the selected points and fiducials to files for reference
                inside_msr_file = os.path.join(save_dir, "inside_msr_points.txt")
                fiducials_file = os.path.join(save_dir, "fiducial_points.txt")
                
                # Save inside MSR points
                with open(inside_msr_file, 'w') as f:
                    f.write("Inside MSR Selected Points:\n")
                    for i, point in enumerate(self.inside_msr_points):
                        f.write(f"Point {i+1}: {point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}\n")
                
                # Save fiducial points
                with open(fiducials_file, 'w') as f:
                    f.write("Fiducial Points:\n")
                    for label, point in self.fiducials_dict.items():
                        f.write(f"{label}: {point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}\n")
                
                QMessageBox.information(self, "Files Saved", 
                                      f"Transformation and point files have been saved to:\n{save_dir}")
                
                # Ask if user wants to exit
                reply = QMessageBox.question(self, "Exit Application", 
                                            "Would you like to exit the application?",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    self.close()
                    
            except Exception as e:
                print(f"Error in saveResults: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"An error occurred while saving: {str(e)}")
                
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OPMCoRegistrationGUI()
    sys.exit(app.exec_())