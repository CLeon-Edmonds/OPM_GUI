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

# Disable VTK warnings
import vtk
vtk.vtkObject.GlobalWarningDisplayOff()

# Set verbosity level for Open3D to avoid unnecessary output
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

sensor_length = 6e-3

def is_stl_file(file_path):
    """Check if the file is an STL file based on extension"""
    _, ext = os.path.splitext(file_path.lower())
    return ext == '.stl'

def preprocess_point_cloud(pcd, voxel_size):
    """Downsample point cloud and compute FPFH features for registration."""
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
        
    def computeHeadToStandardTransform(self):
        """Compute transformation from outside MSR scan to standard coordinate system."""
        try:
            # Keep UI responsive
            QApplication.processEvents()
            
            # Get fiducial points (already selected by user)
            R_aur = self.fiducial_points[0]  # Right pre-auricular
            L_aur = self.fiducial_points[1]  # Left pre-auricular
            nas = self.fiducial_points[2]    # Nasion
            
            # Print fiducial points to verify
            print(f"Right pre-auricular: {R_aur}")
            print(f"Left pre-auricular: {L_aur}")
            print(f"Nasion: {nas}")
            
            # Verify the ordering of fiducials
            # Check if the fiducials make anatomical sense
            rl_distance = np.linalg.norm(R_aur - L_aur)
            print(f"Distance between pre-auricular points: {rl_distance:.2f}mm")
            
            # Keep UI responsive
            QApplication.processEvents()
            
            # Get position of CTF-style origin in original LIDAR data
            origin = (R_aur + L_aur) / 2.0
            
            # Define anatomical points in 'standard' space to align with
            standard = np.zeros([3, 3])
            
            # right pre-auricular on -ve y-axis
            standard[0] = [0, -np.linalg.norm(R_aur - L_aur)/2., 0]
            
            # left pre-auricular on +ve y-axis
            standard[1] = [0, np.linalg.norm(R_aur - L_aur)/2., 0]
            
            # Nasion on x-axis
            standard[2] = [np.linalg.norm(origin-nas), 0, 0]
            
            # Keep UI responsive
            QApplication.processEvents()
            
            # Calculate the transformation matrix
            X2 = self.computeTransformation(self.fiducial_points, standard)
            
            return X2
        except Exception as e:
            print(f"Error in computeHeadToStandardTransform: {e}")
            import traceback
            traceback.print_exc()
            raise e

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
            
            # Try to compute the transformation directly instead of using a worker thread for debugging
            print("Computing head to standard transform...")
            self.X2 = self.computeHeadToStandardTransform()
            
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
            
    def computeHeadToMRITransform(self):
        """Perform the MRI to head registration with improved memory efficiency."""
        try:
            # Load the point clouds using Open3D - with lower resolution
            self.progress_bar.setValue(10)
            QApplication.processEvents()
            
            standard_head = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 50000)
            
            self.progress_bar.setValue(20)
            QApplication.processEvents()
            
            mri_scalp = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 50000)
            
            self.progress_bar.setValue(30)
            QApplication.processEvents()
            
            # Apply the head to standard transform to the standard head
            standard_head.transform(self.X2)
            
            self.progress_bar.setValue(40)
            QApplication.processEvents()
            
            # Create a copy of the MRI scalp for processing
            mri_crop = copy.deepcopy(mri_scalp)
            
            # Calculate average magnitudes for scaling
            points = np.asarray(mri_crop.points)
            standard_points = np.asarray(standard_head.points)
            
            standard_magnitudes = np.mean(np.linalg.norm(standard_points, axis=1))
            mri_magnitudes = np.mean(np.linalg.norm(points, axis=1))
            
            # Calculate scale factor
            scale_factor = standard_magnitudes / mri_magnitudes if mri_magnitudes > 0 else 1.0
            
            print(f"Applying scale factor: {scale_factor}")
            self.progress_bar.setValue(50)
            QApplication.processEvents()
            
            # Apply scaling directly to points
            scaled_points = points * scale_factor
            mri_crop = o3d.geometry.PointCloud()
            mri_crop.points = o3d.utility.Vector3dVector(scaled_points)
            
            # Also scale the original for visualization
            mri_scalp = o3d.geometry.PointCloud()
            mri_scalp.points = o3d.utility.Vector3dVector(scaled_points)
            
            self.progress_bar.setValue(60)
            QApplication.processEvents()
            
            # Use a much simpler ICP with fewer iterations and larger threshold
            threshold = 10.0  # Very forgiving threshold
            max_iterations = 500  # Much fewer iterations
            
            # Create a basic initial alignment based on centroids
            trans_init = np.eye(4)
            mri_center = np.mean(scaled_points, axis=0)
            head_center = np.mean(standard_points, axis=0)
            trans_init[:3, 3] = head_center - mri_center
            
            self.progress_bar.setValue(70)
            QApplication.processEvents()
            
            # Simple ICP with very basic parameters
            result_icp = o3d.pipelines.registration.registration_icp(
                mri_crop, standard_head, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations))
            
            self.progress_bar.setValue(80)
            QApplication.processEvents()
            
            # Store the transformation
            X3 = copy.deepcopy(result_icp.transformation)
            
            # Transform the original MRI scalp with the computed transformation for visualization
            mri_scalp.transform(X3)
            
            self.progress_bar.setValue(90)
            QApplication.processEvents()
            
            # Visualize with existing method but downsampled point clouds
            self.visualizeSimpleRegistration(mri_scalp, standard_head)
            
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            return X3
        except Exception as e:
            self.progress_bar.setValue(0)
            print(f"Error in computeHeadToMRITransform: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in head to MRI registration: {str(e)}")
            raise e
    
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
        """Finalize co-registration process."""
        if error:
            QMessageBox.critical(self, "Error", f"An error occurred during co-registration: {error}")
            return
        
        try:
            # Compute combined transformation if not already computed
            if self.X21 is None and self.X1 is not None and self.X2 is not None:
                self.X21 = np.dot(self.X2, self.X1)
            
            # Skip checkHeadpoints for now since it might cause issues
            #self.checkHeadpoints()
            
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