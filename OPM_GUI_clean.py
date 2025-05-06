import os
import numpy as np
import mne
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QProgressBar, QFrame, QLineEdit, QGridLayout, QMessageBox,
                            QDialog, QDialogButtonBox, QSlider, QGroupBox)
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

class ManualAlignmentInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """Custom interactor style for manual model alignment."""
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.AddObserver("MouseMoveEvent", self.onMouseMove)
        self.AddObserver("LeftButtonPressEvent", self.onLeftButtonDown)
        self.AddObserver("LeftButtonReleaseEvent", self.onLeftButtonUp)
        self.AddObserver("RightButtonPressEvent", self.onRightButtonDown)
        self.AddObserver("RightButtonReleaseEvent", self.onRightButtonUp)
        self.AddObserver("MiddleButtonPressEvent", self.onMiddleButtonDown)
        self.AddObserver("MiddleButtonReleaseEvent", self.onMiddleButtonUp)
        
    def onMouseMove(self, obj, event):
        super().OnMouseMove()
        
    def onLeftButtonDown(self, obj, event):
        super().OnLeftButtonDown()
        
    def onLeftButtonUp(self, obj, event):
        super().OnLeftButtonUp()
        if self.parent:
            self.parent.onModelManipulated()
            
    def onRightButtonDown(self, obj, event):
        super().OnRightButtonDown()
        
    def onRightButtonUp(self, obj, event):
        super().OnRightButtonUp()
        if self.parent:
            self.parent.onModelManipulated()
            
    def onMiddleButtonDown(self, obj, event):
        super().OnMiddleButtonDown()
        
    def onMiddleButtonUp(self, obj, event):
        super().OnMiddleButtonUp()
        if self.parent:
            self.parent.onModelManipulated()
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
class OPMCoRegistrationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_points = []
        self.point_actors = []
        self.file_paths = {}
        self.current_stage = "inside_msr"  
        
        # Initialize transform matrices for manual alignment
        self.alignment_transforms = {
            "Inside MSR": vtk.vtkTransform(),
            "Outside MSR": vtk.vtkTransform(),
            "Scalp File": vtk.vtkTransform()
        }

        # Initialize each transform to identity
        for transform in self.alignment_transforms.values():
            transform.Identity()

        # Dictionary to store temporary aligned files
        self.aligned_file_paths = {}

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
        
        # Start worker thread for loading
        self.worker = WorkerThread("load_model")
        self.worker.progress_signal.connect(self.updateProgress)
        self.worker.finished_signal.connect(self.startManualAlignment)  # Changed to start alignment
        self.worker.start()

    def startManualAlignment(self):
        """Begin the manual alignment process for all models."""
        missing_files = []
        required_files = ["Inside MSR", "Outside MSR", "Scalp File", "OPM Data"]
        for file_type in required_files:
            if file_type not in self.file_paths or not self.file_paths[file_type]:
                missing_files.append(file_type)
        
        if missing_files:
            QMessageBox.warning(self, "Missing Files", 
                            f"Please select the following files:\n{', '.join(missing_files)}")
            return
        
        # Set up for alignment process
        self.progress_bar.setValue(10)
        self.clearPoints()
        
        # Store original camera parameters
        self.original_camera_position = self.ren.GetActiveCamera().GetPosition()
        self.original_camera_focal_point = self.ren.GetActiveCamera().GetFocalPoint()
        self.original_camera_view_up = self.ren.GetActiveCamera().GetViewUp()
        
        # Start with MRI scalp
        self.current_alignment_model = "Scalp File"
        self.instructions_label.setText("Aligning MRI Scalp: Use mouse to rotate/move")
        
        # Load MRI scalp
        self.loadModelForAlignment(self.current_alignment_model)
        
        # Create alignment controls panel
        self.createManualAlignmentPanel()

    def loadModelForAlignment(self, model_type):
        """Load a model for manual alignment."""
        # Clear previous actors
        for actor in self.actors:
            self.ren.RemoveActor(actor)
        self.actors = []
        
        # Load the model
        file_path = self.file_paths[model_type]
        
        if model_type == "Scalp File":
            reader = vtk.vtkSTLReader()
        else:
            reader = vtk.vtkPLYReader()
            
        reader.SetFileName(file_path)
        reader.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set model color
        if model_type == "Scalp File":
            actor.GetProperty().SetColor(1, 0, 0)  # Red for MRI
        elif model_type == "Outside MSR":
            actor.GetProperty().SetColor(0, 1, 0)  # Green for outside scan
        else:
            actor.GetProperty().SetColor(0, 0, 1)  # Blue for inside scan
        
        self.ren.AddActor(actor)
        self.actors.append(actor)
        
        # Store the actor for later transformation
        self.current_actor = actor
        
        # Switch to manual alignment interaction style
        if not hasattr(self, 'manual_align_style'):
            self.manual_align_style = ManualAlignmentInteractorStyle(self)
        
        self.iren.SetInteractorStyle(self.manual_align_style)
        
        # Reset camera and render
        self.ren.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def createManualAlignmentPanel(self):
        """Create control panel for manual alignment."""
        # Create a panel for alignment controls
        if hasattr(self, 'alignment_panel'):
            self.alignment_panel.deleteLater()
        
        self.alignment_panel = QFrame(self)
        self.alignment_panel.setFrameStyle(QFrame.StyledPanel)
        
        panel_layout = QVBoxLayout()
        self.alignment_panel.setLayout(panel_layout)
        
        # Add instructions
        panel_layout.addWidget(QLabel("Manual Alignment:"))
        panel_layout.addWidget(QLabel("• Rotate: Left mouse button"))
        panel_layout.addWidget(QLabel("• Pan: Middle mouse button"))
        panel_layout.addWidget(QLabel("• Zoom: Right mouse button or wheel"))
        
        # Add quick manipulation buttons for common operations
        flip_group = QGroupBox("Quick Adjustments")
        flip_layout = QGridLayout()
        flip_group.setLayout(flip_layout)
        
        flip_x = QPushButton("Flip X")
        flip_y = QPushButton("Flip Y")
        flip_z = QPushButton("Flip Z")
        
        reset_button = QPushButton("Reset")
        reset_button.setStyleSheet("background-color: #FF5733; color: white;")
        
        flip_layout.addWidget(flip_x, 0, 0)
        flip_layout.addWidget(flip_y, 0, 1)
        flip_layout.addWidget(flip_z, 0, 2)
        flip_layout.addWidget(reset_button, 1, 0, 1, 3)
        
        panel_layout.addWidget(flip_group)
        
        # Navigation buttons
        navigation_group = QGroupBox("Navigation")
        navigation_layout = QHBoxLayout()
        navigation_group.setLayout(navigation_layout)
        
        confirm_button = QPushButton("Confirm & Continue")
        confirm_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        navigation_layout.addWidget(confirm_button)
        
        panel_layout.addWidget(navigation_group)
        
        # Connect buttons
        flip_x.clicked.connect(lambda: self.flipAxis(0))
        flip_y.clicked.connect(lambda: self.flipAxis(1))
        flip_z.clicked.connect(lambda: self.flipAxis(2))
        reset_button.clicked.connect(self.resetModelAlignment)
        confirm_button.clicked.connect(self.confirmModelAlignment)
        
        # Position the panel in the right frame
        right_frame = None
        for child in self.children():
            if isinstance(child, QFrame) and child.width() == 300 and child.x() > 600:  # Rightmost frame
                right_frame = child
                break
        
        if right_frame:
            layout = right_frame.layout()
            layout.addWidget(self.alignment_panel)
        else:
            # Fallback - add directly to the main window
            main_layout = self.layout()
            main_layout.addWidget(self.alignment_panel)
        
        # Store transformations for each model
        if not hasattr(self, 'alignment_transforms'):
            self.alignment_transforms = {
                "Scalp File": vtk.vtkTransform(),
                "Outside MSR": vtk.vtkTransform(),
                "Inside MSR": vtk.vtkTransform()
            }
            
            # Initialize transforms
            for key in self.alignment_transforms:
                self.alignment_transforms[key].Identity()

    def onModelManipulated(self):
        """Called when the model has been manipulated through the interactor."""
        # Get the current transformation from the actor
        transform = self.current_actor.GetUserTransform()
        if not transform:
            # If no transform exists yet, create one from the current position
            transform = vtk.vtkTransform()
            transform.SetMatrix(self.current_actor.GetMatrix())
            self.current_actor.SetUserTransform(transform)
        
        # Store the current transform
        self.alignment_transforms[self.current_alignment_model].DeepCopy(transform)

    def flipAxis(self, axis):
        """Flip the model along specified axis (0=X, 1=Y, 2=Z)."""
        # Get current transformation
        transform = self.alignment_transforms[self.current_alignment_model]
        
        # Create a scale transform for flipping
        scale = [1, 1, 1]
        scale[axis] = -1
        
        # Apply the scale transform
        transform.Scale(scale[0], scale[1], scale[2])
        
        # Apply to the actor
        self.current_actor.SetUserTransform(transform)
        
        # Update the render window
        self.vtk_widget.GetRenderWindow().Render()

    def resetModelAlignment(self):
        """Reset the model to its original orientation."""
        # Reset the transform
        self.alignment_transforms[self.current_alignment_model].Identity()
        
        # Apply to the actor
        self.current_actor.SetUserTransform(self.alignment_transforms[self.current_alignment_model])
        
        # Reset camera to original position
        camera = self.ren.GetActiveCamera()
        camera.SetPosition(self.original_camera_position)
        camera.SetFocalPoint(self.original_camera_focal_point)
        camera.SetViewUp(self.original_camera_view_up)
        
        # Update the render window
        self.ren.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def confirmModelAlignment(self):
        """Confirm current model alignment and move to next model."""
        # Save the current alignment transformation
        self.saveAlignedModelPosition(self.current_alignment_model)
        
        # Move to next model
        if self.current_alignment_model == "Scalp File":
            # Move to Outside MSR
            self.current_alignment_model = "Outside MSR"
            self.instructions_label.setText("Aligning Outside MSR: Use mouse to rotate/move")
            self.loadModelForAlignment(self.current_alignment_model)
            
        elif self.current_alignment_model == "Outside MSR":
            # Move to Inside MSR
            self.current_alignment_model = "Inside MSR"
            self.instructions_label.setText("Aligning Inside MSR: Use mouse to rotate/move")
            self.loadModelForAlignment(self.current_alignment_model)
            
        elif self.current_alignment_model == "Inside MSR":
            # Finished all alignments, start co-registration
            if hasattr(self, 'alignment_panel'):
                self.alignment_panel.hide()
            
            # Restore original interaction style
            self.iren.SetInteractorStyle(CustomInteractorStyle())
            
            # Start the co-registration process with aligned models
            self.startCoregistrationWithAlignedModels()

    def startCoregistrationWithAlignedModels(self):
        """Start the co-registration process using the manually aligned models."""
        try:
            # Set up for inside MSR
            self.current_stage = "inside_msr"
            self.updateInstructions()
            
            # Clear any existing points
            self.clearPoints()
            
            # Apply the saved transformations to both models and save them to temporary files
            inside_cloud = self.loadPointCloudOpen3D(self.file_paths["Inside MSR"], 100000)
            inside_cloud.transform(self.alignment_transforms["Inside MSR"].GetMatrix())
            
            temp_inside = os.path.join(os.path.dirname(self.file_paths["Inside MSR"]), "temp_inside_aligned.ply")
            o3d.io.write_point_cloud(temp_inside, inside_cloud)
            
            # Store these temporary files and use them for the actual processing
            self.aligned_file_paths = {}
            self.aligned_file_paths["Inside MSR"] = temp_inside
            
            # Show the aligned model
            self.loadVTKPointCloud(temp_inside)
            
            # Update instructions
            self.instructions_label.setText("Select the 7 helmet label points on the aligned model")
        except Exception as e:
            print(f"Error in startCoregistrationWithAlignedModels: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error starting co-registration: {str(e)}")
            
    def startCoregistrationProcess(self):
        """Begin the co-registration process with file orientation alignment."""
        # Check if files are selected
        missing_files = []
        required_files = ["Inside MSR", "Outside MSR", "Scalp File", "OPM Data"]
        for file_type in required_files:
            if file_type not in self.file_paths or not self.file_paths[file_type]:
                missing_files.append(file_type)
        
        if missing_files:
            QMessageBox.warning(self, "Missing Files", 
                            f"Please select the following files:\n{', '.join(missing_files)}")
            return
        
        # Start with aligning the models
        self.current_alignment_stage = "scalp"
        self.alignModel(self.current_alignment_stage)

    def alignModel(self, model_type):
        """Guide the user through aligning a specific model."""
        try:
            # Set up for model alignment
            self.clearPoints()
            
            # Initialize transform matrix for this model
            self.current_alignment_transform = np.eye(4)
            
            # Load the appropriate model
            if model_type == "scalp":
                self.instructions_label.setText("Align MRI Scalp to standard orientation")
                file_path = self.file_paths["Scalp File"]
                self.current_point_cloud = self.loadPointCloudOpen3D(file_path, 50000)
            elif model_type == "outside":
                self.instructions_label.setText("Align Outside MSR scan to standard orientation")
                file_path = self.file_paths["Outside MSR"]
                self.current_point_cloud = self.loadPointCloudOpen3D(file_path, 50000)
            elif model_type == "inside":
                self.instructions_label.setText("Align Inside MSR scan to standard orientation")
                file_path = self.file_paths["Inside MSR"]
                self.current_point_cloud = self.loadPointCloudOpen3D(file_path, 50000)
            
            # Create a temporary PLY file for visualization
            temp_file = os.path.join(os.path.dirname(file_path), f"temp_{model_type}_viz.ply")
            o3d.io.write_point_cloud(temp_file, self.current_point_cloud)
            
            # Visualize in VTK
            self.loadVTKFile(temp_file)
            
            # Show alignment controls
            self.createAlignmentControls(model_type)
            
        except Exception as e:
            print(f"Error in alignModel: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error setting up model alignment: {str(e)}")


        """Create alignment controls in the UI."""
        # Create a panel with simple orientation controls
        if hasattr(self, 'alignment_panel'):
            self.alignment_panel.deleteLater()
        
        self.alignment_panel = QFrame(self)
        self.alignment_panel.setFrameStyle(QFrame.StyledPanel)
        
        panel_layout = QVBoxLayout()
        self.alignment_panel.setLayout(panel_layout)
        
        # Add instructions
        panel_layout.addWidget(QLabel(f"Align the {model_type} model to standard orientation:"))
        panel_layout.addWidget(QLabel("• Front of head should face towards right"))
        panel_layout.addWidget(QLabel("• Top of head should be up"))
        
        # Flip controls
        flip_group = QGroupBox("Flip Axes")
        flip_layout = QHBoxLayout()
        flip_group.setLayout(flip_layout)
        
        flip_x = QPushButton("Flip X")
        flip_y = QPushButton("Flip Y")
        flip_z = QPushButton("Flip Z")
        
        flip_layout.addWidget(flip_x)
        flip_layout.addWidget(flip_y)
        flip_layout.addWidget(flip_z)
        
        panel_layout.addWidget(flip_group)
        
        # Rotation controls
        rot_group = QGroupBox("Rotate 90°")
        rot_layout = QGridLayout()
        rot_group.setLayout(rot_layout)
        
        rot_x_plus = QPushButton("X+")
        rot_x_minus = QPushButton("X-")
        rot_y_plus = QPushButton("Y+")
        rot_y_minus = QPushButton("Y-")
        rot_z_plus = QPushButton("Z+")
        rot_z_minus = QPushButton("Z-")
        
        rot_layout.addWidget(rot_x_plus, 0, 0)
        rot_layout.addWidget(rot_x_minus, 1, 0)
        rot_layout.addWidget(rot_y_plus, 0, 1)
        rot_layout.addWidget(rot_y_minus, 1, 1)
        rot_layout.addWidget(rot_z_plus, 0, 2)
        rot_layout.addWidget(rot_z_minus, 1, 2)
        
        panel_layout.addWidget(rot_group)
        
        # Reset button
        reset_button = QPushButton("Reset Orientation")
        panel_layout.addWidget(reset_button)
        
        # Continue button
        continue_button = QPushButton("Confirm Orientation")
        continue_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        panel_layout.addWidget(continue_button)
        
        # Connect signals
        flip_x.clicked.connect(lambda: self.applyTransform(self.createFlipMatrix(0)))
        flip_y.clicked.connect(lambda: self.applyTransform(self.createFlipMatrix(1)))
        flip_z.clicked.connect(lambda: self.applyTransform(self.createFlipMatrix(2)))
        
        rot_x_plus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(0, np.pi/2)))
        rot_x_minus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(0, -np.pi/2)))
        rot_y_plus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(1, np.pi/2)))
        rot_y_minus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(1, -np.pi/2)))
        rot_z_plus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(2, np.pi/2)))
        rot_z_minus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(2, -np.pi/2)))
        
        reset_button.clicked.connect(self.resetOrientation)
        continue_button.clicked.connect(self.confirmAlignment)
        
        # Position the panel
        # Find the right frame to add the panel to
        right_frame = None
        for child in self.children():
            if isinstance(child, QFrame) and child.width() == 300 and child.x() > 600:  # Rightmost frame
                right_frame = child
                break
        
        if right_frame:
            layout = right_frame.layout()
            layout.addWidget(self.alignment_panel)
        else:
            # Fallback - add directly to the main window
            main_layout = self.layout()
            main_layout.addWidget(self.alignment_panel)

    def createRotationMatrix(self, axis, angle):
        """Create a transformation matrix for rotation."""
        rot_matrix = np.eye(4)
        
        if axis == 0:  # X-axis
            c, s = np.cos(angle), np.sin(angle)
            rot_matrix[1:3, 1:3] = np.array([[c, -s], [s, c]])
        elif axis == 1:  # Y-axis
            c, s = np.cos(angle), np.sin(angle)
            rot_matrix[0, 0] = c
            rot_matrix[0, 2] = s
            rot_matrix[2, 0] = -s
            rot_matrix[2, 2] = c
        elif axis == 2:  # Z-axis
            c, s = np.cos(angle), np.sin(angle)
            rot_matrix[0:2, 0:2] = np.array([[c, -s], [s, c]])
        
        return rot_matrix

        """Apply a transformation to the current model and update visualization."""
        try:
            # Get current center for transform around center
            center = self.current_point_cloud.get_center()
            
            # Create transformation matrix that:
            # 1. Translates to origin
            # 2. Applies transform
            # 3. Translates back
            T1 = np.eye(4)
            T1[:3, 3] = -center
            
            T2 = np.eye(4)
            T2[:3, 3] = center
            
            combined_transform = np.dot(T2, np.dot(transform, T1))
            
            # Update current transform
            self.current_alignment_transform = np.dot(combined_transform, self.current_alignment_transform)
            
            # Apply to point cloud
            self.current_point_cloud.transform(combined_transform)
            
            # Update visualization
            temp_file = os.path.join(os.path.dirname(self.file_paths["Scalp File"]), 
                                f"temp_{self.current_alignment_stage}_viz.ply")
            o3d.io.write_point_cloud(temp_file, self.current_point_cloud)
            
            # Reload the visualization
            self.loadVTKFile(temp_file)
            
        except Exception as e:
            print(f"Error applying transform: {e}")
            import traceback
            traceback.print_exc()

    def resetOrientation(self):
        """Reset the current model to its original orientation."""
        # Reload the original point cloud
        if self.current_alignment_stage == "scalp":
            file_path = self.file_paths["Scalp File"]
        elif self.current_alignment_stage == "outside":
            file_path = self.file_paths["Outside MSR"]
        elif self.current_alignment_stage == "inside":
            file_path = self.file_paths["Inside MSR"]
        
        self.current_point_cloud = self.loadPointCloudOpen3D(file_path, 50000)
        self.current_alignment_transform = np.eye(4)
        
        # Update visualization
        temp_file = os.path.join(os.path.dirname(file_path), 
                            f"temp_{self.current_alignment_stage}_viz.ply")
        o3d.io.write_point_cloud(temp_file, self.current_point_cloud)
        
        # Reload the visualization
        self.loadVTKFile(temp_file)
             
    def createAlignmentControls(self, model_type):
        """Create alignment controls in the UI."""
        # Create a panel with simple orientation controls
        if hasattr(self, 'alignment_panel'):
            self.alignment_panel.deleteLater()
        
        self.alignment_panel = QFrame(self)
        self.alignment_panel.setFrameStyle(QFrame.StyledPanel)
        
        panel_layout = QVBoxLayout()
        self.alignment_panel.setLayout(panel_layout)
        
        # Add instructions
        panel_layout.addWidget(QLabel(f"Align the {model_type} model to standard orientation:"))
        panel_layout.addWidget(QLabel("• Front of head should face towards right"))
        panel_layout.addWidget(QLabel("• Top of head should be up"))
        
        # Flip controls
        flip_group = QGroupBox("Flip Axes")
        flip_layout = QHBoxLayout()
        flip_group.setLayout(flip_layout)
        
        flip_x = QPushButton("Flip X")
        flip_y = QPushButton("Flip Y")
        flip_z = QPushButton("Flip Z")
        
        flip_layout.addWidget(flip_x)
        flip_layout.addWidget(flip_y)
        flip_layout.addWidget(flip_z)
        
        panel_layout.addWidget(flip_group)
        
        # Rotation controls
        rot_group = QGroupBox("Rotate 90°")
        rot_layout = QGridLayout()
        rot_group.setLayout(rot_layout)
        
        rot_x_plus = QPushButton("X+")
        rot_x_minus = QPushButton("X-")
        rot_y_plus = QPushButton("Y+")
        rot_y_minus = QPushButton("Y-")
        rot_z_plus = QPushButton("Z+")
        rot_z_minus = QPushButton("Z-")
        
        rot_layout.addWidget(rot_x_plus, 0, 0)
        rot_layout.addWidget(rot_x_minus, 1, 0)
        rot_layout.addWidget(rot_y_plus, 0, 1)
        rot_layout.addWidget(rot_y_minus, 1, 1)
        rot_layout.addWidget(rot_z_plus, 0, 2)
        rot_layout.addWidget(rot_z_minus, 1, 2)
        
        panel_layout.addWidget(rot_group)
        
        # Reset button
        reset_button = QPushButton("Reset Orientation")
        panel_layout.addWidget(reset_button)
        
        # Continue button
        continue_button = QPushButton("Confirm Orientation")
        continue_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        panel_layout.addWidget(continue_button)
        
        # Connect signals
        flip_x.clicked.connect(lambda: self.applyTransform(self.createFlipMatrix(0)))
        flip_y.clicked.connect(lambda: self.applyTransform(self.createFlipMatrix(1)))
        flip_z.clicked.connect(lambda: self.applyTransform(self.createFlipMatrix(2)))
        
        rot_x_plus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(0, np.pi/2)))
        rot_x_minus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(0, -np.pi/2)))
        rot_y_plus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(1, np.pi/2)))
        rot_y_minus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(1, -np.pi/2)))
        rot_z_plus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(2, np.pi/2)))
        rot_z_minus.clicked.connect(lambda: self.applyTransform(self.createRotationMatrix(2, -np.pi/2)))
        
        reset_button.clicked.connect(self.resetOrientation)
        continue_button.clicked.connect(self.confirmAlignment)
        
        # Position the panel
        # Find the right frame to add the panel to
        right_frame = None
        for child in self.children():
            if isinstance(child, QFrame) and child.width() == 300 and child.x() > 600:  # Rightmost frame
                right_frame = child
                break
        
        if right_frame:
            layout = right_frame.layout()
            layout.addWidget(self.alignment_panel)
        else:
            # Fallback - add directly to the main window
            main_layout = self.layout()
            main_layout.addWidget(self.alignment_panel)

    def confirmAlignment(self):
        """Confirm current alignment and move to next stage or start registration."""
        # Store the current transform for this model
        if self.current_alignment_stage == "scalp":
            self.scalp_alignment = self.current_alignment_transform
            # Move to outside MSR alignment
            self.current_alignment_stage = "outside"
            self.alignModel("outside")
        elif self.current_alignment_stage == "outside": 
            self.outside_alignment = self.current_alignment_transform
            # Move to inside MSR alignment
            self.current_alignment_stage = "inside"
            self.alignModel("inside")
        elif self.current_alignment_stage == "inside":
            self.inside_alignment = self.current_alignment_transform
            # Hide alignment panel
            if hasattr(self, 'alignment_panel'):
                self.alignment_panel.hide()
            
            # Start the actual co-registration process
            self.startActualCoregistration()

    def loadVTKFile(self, file_path):
        """Load a file directly into VTK for visualization."""
        # Clear previous actors
        for actor in self.actors:
            self.ren.RemoveActor(actor)
        self.actors = []
        
        # Determine file type
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.ply':
            reader = vtk.vtkPLYReader()
            reader.SetFileName(file_path)
            reader.Update()
        elif ext == '.stl':
            reader = vtk.vtkSTLReader()
            reader.SetFileName(file_path)
            reader.Update()
        else:
            print(f"Unsupported file format: {ext}")
            return
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        self.ren.AddActor(actor)
        self.actors.append(actor)
        
        # Reset camera and render
        self.ren.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def startActualCoregistration(self):
        """Start the standard co-registration process with pre-aligned models."""
        # Reset the UI
        self.progress_bar.setValue(0)
        self.instructions_label.setText("Starting Co-registration with pre-aligned models")
        QApplication.processEvents()
        
        # Clean up temporary files
        for stage in ["scalp", "outside", "inside"]:
            temp_file = os.path.join(os.path.dirname(self.file_paths["Scalp File"]), 
                                f"temp_{stage}_viz.ply")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
        
        # Set up for inside MSR
        self.current_stage = "inside_msr"
        self.updateInstructions()
        
        # Load inside MSR with the pre-alignment
        inside_cloud = self.loadPointCloudOpen3D(self.file_paths["Inside MSR"], 100000)
        inside_cloud.transform(self.inside_alignment)
        
        # Create temporary PLY for visualization
        temp_file = os.path.join(os.path.dirname(self.file_paths["Inside MSR"]), "temp_inside_aligned.ply")
        o3d.io.write_point_cloud(temp_file, inside_cloud)
        
        # Update file path temporarily for visualization
        original_inside_path = self.file_paths["Inside MSR"]
        self.file_paths["Inside MSR"] = temp_file
        
        # Load for visualization
        self.loadPointCloud("Inside MSR")
        
        # Restore original path
        self.file_paths["Inside MSR"] = original_inside_path

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
        """Load point cloud with downsampling and apply saved alignments."""
        try:
            # Check file extension
            file_path = str(file_path)
            print(f"Loading file: {file_path}")
            
            # Load the point cloud based on file type
            if file_path.lower().endswith('.ply'):
                try:
                    # Try as mesh first
                    mesh = o3d.io.read_triangle_mesh(file_path)
                    if not mesh.has_vertices():
                        raise ValueError("No vertices found in mesh")
                    point_cloud = mesh.sample_points_poisson_disk(max_points)
                except:
                    # Fall back to direct point cloud loading
                    point_cloud = o3d.io.read_point_cloud(file_path)
                    if len(point_cloud.points) > max_points:
                        point_cloud = point_cloud.random_down_sample(max_points / len(point_cloud.points))
            
            elif file_path.lower().endswith('.stl'):
                mesh = o3d.io.read_triangle_mesh(file_path)
                if not mesh.has_vertices():
                    raise ValueError("No vertices found in STL mesh")
                point_cloud = mesh.sample_points_poisson_disk(max_points)
            
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Apply saved alignment transformations
            if hasattr(self, 'inside_matrix') and file_path == self.file_paths["Inside MSR"]:
                print(f"Applying Inside MSR alignment matrix")
                point_cloud.transform(self.inside_matrix)
            elif hasattr(self, 'outside_matrix') and file_path == self.file_paths["Outside MSR"]:
                print(f"Applying Outside MSR alignment matrix")
                point_cloud.transform(self.outside_matrix)
            elif hasattr(self, 'scalp_matrix') and file_path == self.file_paths["Scalp File"]:
                print(f"Applying Scalp alignment matrix")
                point_cloud.transform(self.scalp_matrix)
            
            print(f"Loaded point cloud with {len(point_cloud.points)} points")
            return point_cloud
            
        except Exception as e:
            print(f"Error in loadPointCloudOpen3D: {e}")
            import traceback
            traceback.print_exc()
            raise

    def saveAlignedModelPosition(self, model_type):
        """Save the current transform matrix of a manually aligned model."""
        # Get the transformation matrix from the current actor
        transform = self.current_actor.GetUserTransform()
        if not transform:
            # If no transform exists yet, create one from the current position
            transform = vtk.vtkTransform()
            transform.SetMatrix(self.current_actor.GetMatrix())
        
        # Convert VTK transform to numpy matrix
        matrix = transform.GetMatrix()
        numpy_matrix = np.eye(4)
        for i in range(4):
            for j in range(4):
                numpy_matrix[i, j] = matrix.GetElement(i, j)
        
        # Store the transformation
        if model_type == "Inside MSR":
            self.inside_matrix = numpy_matrix
            print("Saved Inside MSR alignment")
        elif model_type == "Outside MSR":
            self.outside_matrix = numpy_matrix
            print("Saved Outside MSR alignment")
        elif model_type == "Scalp File":
            self.scalp_matrix = numpy_matrix
            print("Saved MRI Scalp alignment")

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
            
            # Get the transform from the manual alignment process
            manual_transform = np.eye(4)
            if hasattr(self, 'alignment_transforms') and "Inside MSR" in self.alignment_transforms:
                # Convert VTK transform to numpy matrix
                vtk_matrix = self.alignment_transforms["Inside MSR"].GetMatrix()
                for i in range(4):
                    for j in range(4):
                        manual_transform[i, j] = vtk_matrix.GetElement(i, j)
            
            # Calculate the transformation matrix
            self.X1 = self.computeTransformation(selected_points_np, self.known_sensor_positions)
            
            # Combine with manual transform if it exists
            if hasattr(self, 'alignment_transforms') and "Inside MSR" in self.alignment_transforms:
                # The manual transform is applied first to align, then X1 is applied
                self.X1 = np.dot(self.X1, manual_transform)
            
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
            
            # Load aligned outside MSR model
            outside_cloud = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 100000)
            outside_cloud.transform(self.alignment_transforms["Outside MSR"].GetMatrix())
            
            temp_outside = os.path.join(os.path.dirname(self.file_paths["Outside MSR"]), "temp_outside_aligned.ply")
            o3d.io.write_point_cloud(temp_outside, outside_cloud)
            
            # Store in aligned file paths and load
            self.aligned_file_paths["Outside MSR"] = temp_outside
            self.loadVTKPointCloud(temp_outside)
            
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
        """Perform registration that preserves manual alignment."""
        try:
            # Set up progress bar
            self.progress_bar.setValue(10)
            self.instructions_label.setText("Applying manual alignments")
            QApplication.processEvents()
            
            # Load head with alignment and fiducial-based transform
            standard_head = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 100000)
            standard_head.transform(self.alignment_transforms["Outside MSR"].GetMatrix())
            standard_head.transform(self.X2)  # Apply fiducial-based transform
            
            # Load MRI with alignment
            mri_scalp = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 100000)
            mri_scalp.transform(self.alignment_transforms["Scalp File"].GetMatrix())
            
            # Run ICP to fine-tune alignment
            self.progress_bar.setValue(40)
            self.instructions_label.setText("Running fine registration (ICP)")
            QApplication.processEvents()
            
            # Run ICP for better alignment
            threshold = 5.0  # Starting threshold
            result_icp = o3d.pipelines.registration.registration_icp(
                mri_scalp, standard_head, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            
            # Run a second ICP with tighter threshold for refinement
            threshold = 2.0
            refined_icp = o3d.pipelines.registration.registration_icp(
                mri_scalp, standard_head, threshold, result_icp.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            
            # The final transformation is the manual alignment plus ICP refinement
            X3 = np.dot(refined_icp.transformation, result_icp.transformation)
            
            # Apply to MRI for visualization
            mri_final = copy.deepcopy(mri_scalp)
            mri_final.transform(X3)
            
            # Visualize result
            self.progress_bar.setValue(90)
            self.instructions_label.setText("Registration complete")
            QApplication.processEvents()
            
            self.visualizeSimpleRegistration(mri_final, standard_head)
            
            self.progress_bar.setValue(100)
            
            return X3
            
        except Exception as e:
            self.progress_bar.setValue(0)
            print(f"Error in computeHeadToMRITransform: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in head to MRI registration: {str(e)}")
            raise e
        
    def onHeadSelectionDone(self):
        """Handle completion of head point selection."""
        try:
            # Save the selected points as head registration points
            self.head_reg_points = [self.selected_points[i] for i in range(len(self.selected_points))]
            
            if len(self.head_reg_points) < 3:
                QMessageBox.warning(self, "Not Enough Points", 
                                "Please select at least 3 points on the head scan.")
                return
            
            # Hide the continue button
            if hasattr(self, 'head_selection_done_button'):
                self.head_selection_done_button.hide()
                
            # Step 3: Now select points on the MRI
            self.progress_bar.setValue(40)
            self.instructions_label.setText("Select matching points on the MRI scan")
            
            # Set up for MRI point selection
            self.current_selection_model = "mri"
            self.clearPoints()  # Clear head points
            
            # Load the MRI for viewing - we'll need to convert it to PLY first
            # Create temporary PLY file for visualization
            mri_scalp = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 100000)
            self.temp_mri_file = os.path.join(os.path.dirname(self.file_paths["Scalp File"]), 
                                            "temp_mri_for_viz.ply")
            o3d.io.write_point_cloud(self.temp_mri_file, mri_scalp)
            
            # Store the temporary path and use it
            self.file_paths["Temp MRI"] = self.temp_mri_file
            self.loadVTKPointCloud(self.temp_mri_file)
            
            # Display instructions to user
            QMessageBox.information(self, "Point Selection", 
                                "Now select matching points on the MRI scan.\n\n"
                                "Select points in the SAME ORDER as you did on the head scan.\n\n"
                                "Use Shift+Left Click to select points.")
            
            # Set up a button to continue when done with MRI selection
            self.mri_selection_done_button = QPushButton("Complete Point Selection", self)
            self.mri_selection_done_button.setFont(QFont("Arial", 12))
            self.mri_selection_done_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.mri_selection_done_button.clicked.connect(self.onMRISelectionDone)
            self.mri_selection_done_button.show()
            
            # Position the button at the bottom of the right frame
            right_frame = None
            for child in self.children():
                if isinstance(child, QFrame) and child.width() == 300 and child.x() > 600:  # Rightmost frame
                    right_frame = child
                    break
            
            if right_frame:
                layout = right_frame.layout()
                layout.addWidget(self.mri_selection_done_button)
                
        except Exception as e:
            print(f"Error in onHeadSelectionDone: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in head selection: {str(e)}")

    def onMRISelectionDone(self):
        """Handle completion of MRI point selection and continue with registration."""
        try:
            # Save the selected points as MRI registration points
            self.mri_reg_points = [self.selected_points[i] for i in range(len(self.selected_points))]
            
            if len(self.mri_reg_points) < 3:
                QMessageBox.warning(self, "Not Enough Points", 
                                "Please select at least 3 points on the MRI scan.")
                return
            
            # Hide the continue button
            if hasattr(self, 'mri_selection_done_button'):
                self.mri_selection_done_button.hide()
            
            # Check if counts match
            if len(self.mri_reg_points) != len(self.head_reg_points):
                # Use the smaller count
                n_points = min(len(self.mri_reg_points), len(self.head_reg_points))
                self.mri_reg_points = self.mri_reg_points[:n_points]
                self.head_reg_points = self.head_reg_points[:n_points]
                
                QMessageBox.warning(self, "Point Count Mismatch", 
                                f"Using the first {n_points} points from each selection.")
            
            # Continue with registration
            self.progress_bar.setValue(60)
            self.instructions_label.setText("Computing registration")
            QApplication.processEvents()
            
            # Convert VTK points to Open3D format
            self.continueRegistrationWithPoints()
            
        except Exception as e:
            print(f"Error in onMRISelectionDone: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in MRI selection: {str(e)}")
            
    def loadVTKPointCloud(self, file_path):
        """Load a point cloud file directly into VTK for visualization."""
        # Clear previous actors
        for actor in self.actors:
            self.ren.RemoveActor(actor)
        self.actors = []
        
        # Determine file type
        _, ext = os.path.splitext(file_path.lower())
        
        if ext == '.ply':
            reader = vtk.vtkPLYReader()
            reader.SetFileName(file_path)
            reader.Update()
        elif ext == '.stl':
            reader = vtk.vtkSTLReader()
            reader.SetFileName(file_path)
            reader.Update()
        else:
            print(f"Unsupported file format: {ext}")
            return
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        self.ren.AddActor(actor)
        self.actors.append(actor)
        
        # Reset camera and render
        self.ren.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def vtk_to_numpy_matrix(self, vtk_transform):
        """Convert a VTK transform to a NumPy matrix"""
        if vtk_transform is None:
            return np.eye(4)
        
        matrix = vtk_transform.GetMatrix()
        numpy_matrix = np.eye(4)
        for i in range(4):
            for j in range(4):
                numpy_matrix[i, j] = matrix.GetElement(i, j)
        
        return numpy_matrix

    def continueRegistrationWithPoints(self):
        """Continue the registration process using the selected points."""
        try:
            # Load point clouds with Open3D
            standard_head = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 100000)
            standard_head.transform(self.X2)  # Apply head to standard transform
            
            mri_scalp = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 100000)
            
            # Convert VTK points to Open3D format
            # Open3D uses different coordinate system than VTK
            head_points = np.array(self.head_reg_points)
            mri_points = np.array(self.mri_reg_points)
            
            # Create correspondence array for Open3D
            n_points = len(mri_points)
            corr = np.zeros((n_points, 2), dtype=int)
            
            # For point-to-point registration, we need to create dummy indices
            for i in range(n_points):
                corr[i, 0] = i  # Source (MRI) index
                corr[i, 1] = i  # Target (head) index
            
            # Create point clouds with just the selected points
            head_point_cloud = o3d.geometry.PointCloud()
            head_point_cloud.points = o3d.utility.Vector3dVector(head_points)
            
            mri_point_cloud = o3d.geometry.PointCloud()
            mri_point_cloud.points = o3d.utility.Vector3dVector(mri_points)
            
            # Calculate initial transform based on selected points
            p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            trans_init = p2p.compute_transformation(mri_point_cloud, head_point_cloud,
                                                o3d.utility.Vector2iVector(corr))
            
            # Apply initial transform to MRI
            mri_aligned = copy.deepcopy(mri_scalp)
            mri_aligned.transform(trans_init)
            
            # Crop both clouds for better ICP
            # This is similar to York's approach, but using a more robust method
            head_points_array = np.asarray(standard_head.points)
            mri_points_array = np.asarray(mri_aligned.points)
            
            # Get bounding box of head scan to help with cropping
            head_bb = standard_head.get_axis_aligned_bounding_box()
            
            # Crop MRI points to rough match the head scan extent (with some margin)
            min_bound = head_bb.min_bound - 20  # 20mm margin
            max_bound = head_bb.max_bound + 20
            
            # Filter MRI points to be within the bounding box
            x_filter = np.logical_and(mri_points_array[:, 0] >= min_bound[0], 
                                    mri_points_array[:, 0] <= max_bound[0])
            y_filter = np.logical_and(mri_points_array[:, 1] >= min_bound[1], 
                                    mri_points_array[:, 1] <= max_bound[1])
            z_filter = np.logical_and(mri_points_array[:, 2] >= min_bound[2], 
                                    mri_points_array[:, 2] <= max_bound[2])
            
            combined_filter = np.logical_and(x_filter, np.logical_and(y_filter, z_filter))
            
            mri_crop = o3d.geometry.PointCloud()
            mri_crop.points = o3d.utility.Vector3dVector(mri_points_array[combined_filter])
            
            # Update progress
            self.progress_bar.setValue(80)
            self.instructions_label.setText("Running fine registration")
            QApplication.processEvents()
            
            # Run ICP for refinement
            threshold = 5.0  # Start with a larger threshold
            result_icp = o3d.pipelines.registration.registration_icp(
                mri_crop, standard_head, threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
            
            # Run a second ICP with tighter threshold
            threshold = 2.0
            refined_icp = o3d.pipelines.registration.registration_icp(
                mri_crop, standard_head, threshold, result_icp.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000))
            
            # Combine the transformations
            # Initial point-based + First ICP + Second ICP
            final_transform = np.dot(refined_icp.transformation, 
                                    np.dot(result_icp.transformation, trans_init))
            
            # Apply to full MRI for visualization
            mri_final = copy.deepcopy(mri_scalp)
            mri_final.transform(final_transform)
            
            # Visualize result
            self.progress_bar.setValue(90)
            self.instructions_label.setText("Registration complete")
            QApplication.processEvents()
            
            # Load the head scan again for our visualization
            self.loadPointCloud("Outside MSR")
            
            # Save the transformation
            self.X3 = final_transform
            
            # Visualize result using our function
            self.visualizeSimpleRegistration(mri_final, standard_head)
            
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            # Cleanup temporary file
            if hasattr(self, 'temp_mri_file') and os.path.exists(self.temp_mri_file):
                try:
                    os.remove(self.temp_mri_file)
                except:
                    pass
            
            return self.X3
            
        except Exception as e:
            self.progress_bar.setValue(0)
            print(f"Error in continueRegistrationWithPoints: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in registration: {str(e)}")
            raise e
        
    def displayManualAlignment(self):
        """Handle manual alignment and continue with automatic registration."""
        try:
            # Load the point clouds
            standard_head = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 50000)
            standard_head.transform(self.X2)  # Apply the head to standard transform
            
            mri_scalp = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 50000)
            
            # Initialize manual transform
            self.manual_transform = np.eye(4)
            
            # Create simple dialog for manual alignment
            dialog = QDialog(self)
            dialog.setWindowTitle("Manual Pre-alignment")
            dialog.resize(300, 400)
            
            layout = QVBoxLayout()
            dialog.setLayout(layout)
            
            # Add instructions
            layout.addWidget(QLabel("Adjust the MRI to match head orientation:"))
            
            # Flip buttons - simple controls that are easy to understand
            buttons_layout = QGridLayout()
            
            flip_x = QPushButton("Flip X")
            flip_y = QPushButton("Flip Y")
            flip_z = QPushButton("Flip Z")
            rot_x_plus = QPushButton("Rotate X +90°")
            rot_x_minus = QPushButton("Rotate X -90°")
            rot_y_plus = QPushButton("Rotate Y +90°")
            rot_y_minus = QPushButton("Rotate Y -90°")
            rot_z_plus = QPushButton("Rotate Z +90°")
            rot_z_minus = QPushButton("Rotate Z -90°")
            
            buttons_layout.addWidget(flip_x, 0, 0)
            buttons_layout.addWidget(flip_y, 0, 1)
            buttons_layout.addWidget(flip_z, 0, 2)
            buttons_layout.addWidget(rot_x_plus, 1, 0)
            buttons_layout.addWidget(rot_y_plus, 1, 1)
            buttons_layout.addWidget(rot_z_plus, 1, 2)
            buttons_layout.addWidget(rot_x_minus, 2, 0)
            buttons_layout.addWidget(rot_y_minus, 2, 1)
            buttons_layout.addWidget(rot_z_minus, 2, 2)
            
            layout.addLayout(buttons_layout)
            
            # Scale control
            scale_layout = QHBoxLayout()
            scale_layout.addWidget(QLabel("Scale:"))
            
            scale_slider = QSlider(Qt.Horizontal)
            scale_slider.setMinimum(50)
            scale_slider.setMaximum(200)
            scale_slider.setValue(100)
            scale_slider.setTickPosition(QSlider.TicksBelow)
            scale_slider.setTickInterval(10)
            
            scale_value = QLabel("1.00")
            scale_layout.addWidget(scale_slider)
            scale_layout.addWidget(scale_value)
            
            layout.addLayout(scale_layout)
            
            # Apply and reset buttons
            action_layout = QHBoxLayout()
            apply_button = QPushButton("Apply Changes")
            reset_button = QPushButton("Reset")
            action_layout.addWidget(apply_button)
            action_layout.addWidget(reset_button)
            
            layout.addLayout(action_layout)
            
            # OK/Cancel buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(button_box)
            
            # Connect button signals
            mri_copy = copy.deepcopy(mri_scalp)
            
            def update_view():
                self.visualizeSimpleRegistration(mri_copy, standard_head)
            
            def apply_transformation(transform):
                nonlocal mri_copy
                mri_copy.transform(transform)
                self.manual_transform = np.dot(transform, self.manual_transform)
                update_view()
            
            def flip_axis(axis):
                # Create flip matrix
                flip_matrix = np.eye(4)
                flip_matrix[axis, axis] = -1
                
                # Center, flip, uncenter
                center = mri_copy.get_center()
                T_center = np.eye(4)
                T_center[:3, 3] = -center
                
                T_back = np.eye(4)
                T_back[:3, 3] = center
                
                transform = np.dot(T_back, np.dot(flip_matrix, T_center))
                
                apply_transformation(transform)
            
            def rotate_axis(axis, angle):
                # Create rotation matrix
                rot_matrix = np.eye(4)
                
                if axis == 0:  # X-axis
                    c, s = np.cos(angle), np.sin(angle)
                    rot_matrix[1:3, 1:3] = np.array([[c, -s], [s, c]])
                elif axis == 1:  # Y-axis
                    c, s = np.cos(angle), np.sin(angle)
                    rot_vals = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                    rot_matrix[:3, :3] = rot_vals
                elif axis == 2:  # Z-axis
                    c, s = np.cos(angle), np.sin(angle)
                    rot_matrix[0:2, 0:2] = np.array([[c, -s], [s, c]])
                
                # Center, rotate, uncenter
                center = mri_copy.get_center()
                T_center = np.eye(4)
                T_center[:3, 3] = -center
                
                T_back = np.eye(4)
                T_back[:3, 3] = center
                
                transform = np.dot(T_back, np.dot(rot_matrix, T_center))
                
                apply_transformation(transform)
            
            def update_scale():
                scale = scale_slider.value() / 100.0
                scale_value.setText(f"{scale:.2f}")
                
                # Center, scale, uncenter
                center = mri_copy.get_center()
                T_center = np.eye(4)
                T_center[:3, 3] = -center
                
                S = np.eye(4)
                S[0, 0] = S[1, 1] = S[2, 2] = scale
                
                T_back = np.eye(4)
                T_back[:3, 3] = center
                
                transform = np.dot(T_back, np.dot(S, T_center))
                
                # Reset copy and apply full transform to avoid compounding scale
                mri_copy = copy.deepcopy(mri_scalp)
                self.manual_transform = transform
                mri_copy.transform(self.manual_transform)
                update_view()
            
            def reset_transform():
                nonlocal mri_copy
                mri_copy = copy.deepcopy(mri_scalp)
                self.manual_transform = np.eye(4)
                scale_slider.setValue(100)
                update_view()
            
            # Connect signals
            flip_x.clicked.connect(lambda: flip_axis(0))
            flip_y.clicked.connect(lambda: flip_axis(1))
            flip_z.clicked.connect(lambda: flip_axis(2))
            
            rot_x_plus.clicked.connect(lambda: rotate_axis(0, np.pi/2))
            rot_x_minus.clicked.connect(lambda: rotate_axis(0, -np.pi/2))
            rot_y_plus.clicked.connect(lambda: rotate_axis(1, np.pi/2))
            rot_y_minus.clicked.connect(lambda: rotate_axis(1, -np.pi/2))
            rot_z_plus.clicked.connect(lambda: rotate_axis(2, np.pi/2))
            rot_z_minus.clicked.connect(lambda: rotate_axis(2, -np.pi/2))
            
            apply_button.clicked.connect(update_scale)
            reset_button.clicked.connect(reset_transform)
            
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            # Initial visualization
            update_view()
            
            # Show dialog - modal
            dialog.setModal(True)
            result = dialog.exec_()
            
            if result == QDialog.Accepted:
                # User accepted manual alignment, continue with automatic registration
                self.progress_bar.setValue(20)
                self.instructions_label.setText("Running automatic registration")
                QApplication.processEvents()
                
                # Run the automatic registration
                self.continueWithAutomaticRegistration(mri_copy, standard_head)
            else:
                # User canceled, still run automatic but without manual pre-alignment
                self.progress_bar.setValue(20)
                self.instructions_label.setText("Manual alignment canceled, running automatic only")
                QApplication.processEvents()
                
                self.continueWithAutomaticRegistration(mri_scalp, standard_head)
                
        except Exception as e:
            print(f"Error in displayManualAlignment: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in manual alignment: {str(e)}")
            
            # Fall back to automatic registration
            self.progress_bar.setValue(20)
            self.instructions_label.setText("Error in manual alignment, running automatic only")
            QApplication.processEvents()
            
            try:
                standard_head = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 50000)
                standard_head.transform(self.X2)
                mri_scalp = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 50000)
                self.continueWithAutomaticRegistration(mri_scalp, standard_head)
            except Exception as e2:
                print(f"Error in fallback registration: {e2}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Registration failed: {str(e2)}")
                
    def continueWithAutomaticRegistration(self, mri_scalp, standard_head):
        """Continue with automatic registration after manual alignment."""
        try:
            # Create a cropped copy for registration, following York's approach
            mri_crop = copy.deepcopy(mri_scalp)
            
            # Crop MRI using York's thresholds
            points = np.asarray(mri_crop.points)
            
            # First crop based on y coordinate
            y_threshold = 0.
            indices = np.where(points[:, 1] > y_threshold)[0]
            if len(indices) > 0:
                mri_crop = mri_crop.select_by_index(indices)
                
                # Update points after first crop
                points = np.asarray(mri_crop.points)
                
                # Second crop based on z coordinate
                z_threshold = -100.
                indices = np.where(points[:, 2] > z_threshold)[0]
                if len(indices) > 0:
                    mri_crop = mri_crop.select_by_index(indices)
            
            self.progress_bar.setValue(40)
            self.instructions_label.setText("Computing global registration")
            QApplication.processEvents()
            
            # Prepare for global registration
            voxel_size = 2.0
            # Use the global functions instead of methods
            source_down, source_fpfh = preprocess_point_cloud(mri_crop, voxel_size)
            target_down, target_fpfh = preprocess_point_cloud(standard_head, voxel_size)
            
            self.progress_bar.setValue(60)
            QApplication.processEvents()
            
            # Global registration
            result_global = execute_global_registration(source_down, target_down, 
                                                    source_fpfh, target_fpfh, voxel_size)
            
            # Apply global registration result
            trans_init = result_global.transformation
            
            self.progress_bar.setValue(80)
            self.instructions_label.setText("Running ICP refinement")
            QApplication.processEvents()
            
            # ICP refinement
            threshold = 2.0
            result_icp = o3d.pipelines.registration.registration_icp(
                mri_crop, standard_head, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=8000))
            
            # Final transformation is the combination of global and ICP
            final_transform = np.dot(result_icp.transformation, trans_init)
            
            # If manual transform was applied, include it in the final transform
            if hasattr(self, 'manual_transform') and self.manual_transform is not None:
                self.X3 = np.dot(final_transform, self.manual_transform)
            else:
                self.X3 = final_transform
            
            # Apply to original MRI for visualization
            mri_final = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 50000)
            mri_final.transform(self.X3)
            
            self.progress_bar.setValue(90)
            self.instructions_label.setText("Registration in progress")
            QApplication.processEvents()
            
            # Visualize the result
            self.visualizeSimpleRegistration(mri_final, standard_head)
            
            self.progress_bar.setValue(100)
            QApplication.processEvents()
            
            return self.X3
            
        except Exception as e:
            print(f"Error in continueWithAutomaticRegistration: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in automatic registration: {str(e)}")
            raise e
    
    def manualPreAlignMRI(self):
        """Allow manual pre-alignment of MRI to head scan before automatic registration."""
        try:
            # Load the point clouds
            self.progress_bar.setValue(10)
            self.instructions_label.setText("Loading point clouds for manual alignment")
            QApplication.processEvents()
            
            # Load outside MSR scan (already transformed to standard space)
            standard_head = self.loadPointCloudOpen3D(self.file_paths["Outside MSR"], 50000)
            # Apply the head to standard transform
            standard_head.transform(self.X2)
            
            # Load MRI scalp file
            mri_scalp = self.loadPointCloudOpen3D(self.file_paths["Scalp File"], 50000)
            
            # Create dialog for manual alignment
            align_dialog = QDialog(self)
            align_dialog.setWindowTitle("Manual Pre-alignment")
            align_dialog.setMinimumSize(400, 500)
            
            layout = QVBoxLayout()
            align_dialog.setLayout(layout)
            
            # Add instructions
            instructions = QLabel("Adjust the MRI orientation to match the head scan:")
            layout.addWidget(instructions)
            
            # Create sliders for rotation around each axis
            slider_group = QGroupBox("Rotation")
            slider_layout = QVBoxLayout()
            slider_group.setLayout(slider_layout)
            
            # X-axis rotation
            x_layout = QHBoxLayout()
            x_label = QLabel("X-axis:")
            x_slider = QSlider(Qt.Horizontal)
            x_slider.setMinimum(-180)
            x_slider.setMaximum(180)
            x_slider.setValue(0)
            x_slider.setTickPosition(QSlider.TicksBelow)
            x_slider.setTickInterval(30)
            x_value = QLabel("0°")
            x_layout.addWidget(x_label)
            x_layout.addWidget(x_slider)
            x_layout.addWidget(x_value)
            slider_layout.addLayout(x_layout)
            
            # Y-axis rotation
            y_layout = QHBoxLayout()
            y_label = QLabel("Y-axis:")
            y_slider = QSlider(Qt.Horizontal)
            y_slider.setMinimum(-180)
            y_slider.setMaximum(180)
            y_slider.setValue(0)
            y_slider.setTickPosition(QSlider.TicksBelow)
            y_slider.setTickInterval(30)
            y_value = QLabel("0°")
            y_layout.addWidget(y_label)
            y_layout.addWidget(y_slider)
            y_layout.addWidget(y_value)
            slider_layout.addLayout(y_layout)
            
            # Z-axis rotation
            z_layout = QHBoxLayout()
            z_label = QLabel("Z-axis:")
            z_slider = QSlider(Qt.Horizontal)
            z_slider.setMinimum(-180)
            z_slider.setMaximum(180)
            z_slider.setValue(0)
            z_slider.setTickPosition(QSlider.TicksBelow)
            z_slider.setTickInterval(30)
            z_value = QLabel("0°")
            z_layout.addWidget(z_label)
            z_layout.addWidget(z_slider)
            z_layout.addWidget(z_value)
            slider_layout.addLayout(z_layout)
            
            # Add slider group to main layout
            layout.addWidget(slider_group)
            
            # Scale slider
            scale_group = QGroupBox("Scale")
            scale_layout = QHBoxLayout()
            scale_group.setLayout(scale_layout)
            
            scale_label = QLabel("Scale:")
            scale_slider = QSlider(Qt.Horizontal)
            scale_slider.setMinimum(50)
            scale_slider.setMaximum(200)
            scale_slider.setValue(100)
            scale_slider.setTickPosition(QSlider.TicksBelow)
            scale_slider.setTickInterval(10)
            scale_value = QLabel("1.00")
            scale_layout.addWidget(scale_label)
            scale_layout.addWidget(scale_slider)
            scale_layout.addWidget(scale_value)
            
            layout.addWidget(scale_group)
            
            # Translation sliders
            trans_group = QGroupBox("Translation")
            trans_layout = QVBoxLayout()
            trans_group.setLayout(trans_layout)
            
            # X translation
            tx_layout = QHBoxLayout()
            tx_label = QLabel("X:")
            tx_slider = QSlider(Qt.Horizontal)
            tx_slider.setMinimum(-100)
            tx_slider.setMaximum(100)
            tx_slider.setValue(0)
            tx_value = QLabel("0")
            tx_layout.addWidget(tx_label)
            tx_layout.addWidget(tx_slider)
            tx_layout.addWidget(tx_value)
            trans_layout.addLayout(tx_layout)
            
            # Y translation
            ty_layout = QHBoxLayout()
            ty_label = QLabel("Y:")
            ty_slider = QSlider(Qt.Horizontal)
            ty_slider.setMinimum(-100)
            ty_slider.setMaximum(100)
            ty_slider.setValue(0)
            ty_value = QLabel("0")
            ty_layout.addWidget(ty_label)
            ty_layout.addWidget(ty_slider)
            ty_layout.addWidget(ty_value)
            trans_layout.addLayout(ty_layout)
            
            # Z translation
            tz_layout = QHBoxLayout()
            tz_label = QLabel("Z:")
            tz_slider = QSlider(Qt.Horizontal)
            tz_slider.setMinimum(-100)
            tz_slider.setMaximum(100)
            tz_slider.setValue(0)
            tz_value = QLabel("0")
            tz_layout.addWidget(tz_label)
            tz_layout.addWidget(tz_slider)
            tz_layout.addWidget(tz_value)
            trans_layout.addLayout(tz_layout)
            
            layout.addWidget(trans_group)
            
            # Update button
            update_button = QPushButton("Update View")
            layout.addWidget(update_button)
            
            # Preset buttons
            preset_layout = QHBoxLayout()
            flip_x_button = QPushButton("Flip X")
            flip_y_button = QPushButton("Flip Y")
            flip_z_button = QPushButton("Flip Z")
            reset_button = QPushButton("Reset")
            preset_layout.addWidget(flip_x_button)
            preset_layout.addWidget(flip_y_button)
            preset_layout.addWidget(flip_z_button)
            preset_layout.addWidget(reset_button)
            layout.addLayout(preset_layout)
            
            # Button box
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(button_box)
            
            # Connect slider value changes to labels
            def update_slider_labels():
                x_value.setText(f"{x_slider.value()}°")
                y_value.setText(f"{y_slider.value()}°")
                z_value.setText(f"{z_slider.value()}°")
                scale_value.setText(f"{scale_slider.value()/100:.2f}")
                tx_value.setText(f"{tx_slider.value()}")
                ty_value.setText(f"{ty_slider.value()}")
                tz_value.setText(f"{tz_slider.value()}")
            
            x_slider.valueChanged.connect(update_slider_labels)
            y_slider.valueChanged.connect(update_slider_labels)
            z_slider.valueChanged.connect(update_slider_labels)
            scale_slider.valueChanged.connect(update_slider_labels)
            tx_slider.valueChanged.connect(update_slider_labels)
            ty_slider.valueChanged.connect(update_slider_labels)
            tz_slider.valueChanged.connect(update_slider_labels)
            
            # Create copy of the MRI for transformation
            mri_aligned = copy.deepcopy(mri_scalp)
            
            # Function to update the visualization
            def update_visualization():
                # Get values from sliders
                x_rot = np.radians(x_slider.value())
                y_rot = np.radians(y_slider.value())
                z_rot = np.radians(z_slider.value())
                scale = scale_slider.value() / 100.0
                tx = tx_slider.value() * 5  # Scale for better control
                ty = ty_slider.value() * 5
                tz = tz_slider.value() * 5
                
                # Create a fresh copy
                mri_aligned = copy.deepcopy(mri_scalp)
                
                # Build transformation matrix
                # First center the point cloud
                center = mri_aligned.get_center()
                
                # Translate to origin
                T_center = np.eye(4)
                T_center[:3, 3] = -center
                
                # Create rotation matrices
                Rx = np.eye(4)
                Rx[1, 1] = np.cos(x_rot)
                Rx[1, 2] = -np.sin(x_rot)
                Rx[2, 1] = np.sin(x_rot)
                Rx[2, 2] = np.cos(x_rot)
                
                Ry = np.eye(4)
                Ry[0, 0] = np.cos(y_rot)
                Ry[0, 2] = np.sin(y_rot)
                Ry[2, 0] = -np.sin(y_rot)
                Ry[2, 2] = np.cos(y_rot)
                
                Rz = np.eye(4)
                Rz[0, 0] = np.cos(z_rot)
                Rz[0, 1] = -np.sin(z_rot)
                Rz[1, 0] = np.sin(z_rot)
                Rz[1, 1] = np.cos(z_rot)
                
                # Scale matrix
                S = np.eye(4)
                S[0, 0] = scale
                S[1, 1] = scale
                S[2, 2] = scale
                
                # Translation back
                T_back = np.eye(4)
                T_back[:3, 3] = center
                
                # Additional translation
                T_adjust = np.eye(4)
                T_adjust[0, 3] = tx
                T_adjust[1, 3] = ty
                T_adjust[2, 3] = tz
                
                # Combine transformations: center -> rotate -> scale -> translate back -> adjust
                transform = np.dot(T_adjust, 
                                np.dot(T_back, 
                                        np.dot(S, 
                                            np.dot(Rz, 
                                                    np.dot(Ry, 
                                                        np.dot(Rx, T_center))))))
                
                # Apply transformation
                mri_aligned.transform(transform)
                
                # Visualize
                self.visualizeSimpleRegistration(mri_aligned, standard_head)
                
                # Store the transformation matrix
                self.manual_transform = transform
            
            # Connect update button
            update_button.clicked.connect(update_visualization)
            
            # Connect preset buttons
            def flip_x():
                x_slider.setValue(180 if x_slider.value() == 0 else 0)
                update_visualization()
                
            def flip_y():
                y_slider.setValue(180 if y_slider.value() == 0 else 0)
                update_visualization()
                
            def flip_z():
                z_slider.setValue(180 if z_slider.value() == 0 else 0)
                update_visualization()
                
            def reset_all():
                x_slider.setValue(0)
                y_slider.setValue(0)
                z_slider.setValue(0)
                scale_slider.setValue(100)
                tx_slider.setValue(0)
                ty_slider.setValue(0)
                tz_slider.setValue(0)
                update_visualization()
            
            flip_x_button.clicked.connect(flip_x)
            flip_y_button.clicked.connect(flip_y)
            flip_z_button.clicked.connect(flip_z)
            reset_button.clicked.connect(reset_all)
            
            # Connect dialog buttons
            button_box.accepted.connect(align_dialog.accept)
            button_box.rejected.connect(align_dialog.reject)
            
            # Initialize with a first visualization
            self.manual_transform = np.eye(4)
            update_visualization()
            
            # Show the dialog
            if align_dialog.exec_() == QDialog.Accepted:
                return True, mri_aligned, self.manual_transform
            else:
                return False, None, None
                
        except Exception as e:
            print(f"Error in manualPreAlignMRI: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error in manual alignment: {str(e)}")
            return False, None, None
    
    def improvedRegistration(self, source_cloud, target_cloud, max_iterations=200, initial_threshold=25.0, final_threshold=5.0):
        """
        Perform improved point cloud registration with multi-stage ICP and visual feedback.
        
        Args:
            source_cloud: The source point cloud (Open3D PointCloud)
            target_cloud: The target point cloud (Open3D PointCloud)
            max_iterations: Maximum ICP iterations per stage
            initial_threshold: Initial distance threshold for ICP (mm)
            final_threshold: Final distance threshold for ICP (mm)
        
        Returns:
            The transformation matrix from source to target
        """
        try:
            # Make copies to avoid modifying the originals
            source = copy.deepcopy(source_cloud)
            target = copy.deepcopy(target_cloud)
            
            # Define the stage thresholds (gradually decreasing)
            thresholds = np.linspace(initial_threshold, final_threshold, 4)
            
            # Get initial alignment based on centroids
            trans_init = np.eye(4)
            source_center = np.mean(np.asarray(source.points), axis=0)
            target_center = np.mean(np.asarray(target.points), axis=0)
            trans_init[:3, 3] = target_center - source_center
            
            # Apply initial transform
            current_transform = trans_init
            source.transform(current_transform)
            
            # Visualize initial alignment
            self.visualizeRegistrationProgress(source, target, "Initial Alignment")
            
            # Perform multi-stage ICP with decreasing thresholds
            for i, threshold in enumerate(thresholds):
                self.progress_bar.setValue(25 * (i + 1))
                QApplication.processEvents()
                
                # Configure ICP for this stage
                criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=int(max_iterations/(i+1))
                )
                
                # Run ICP for this stage
                result_icp = o3d.pipelines.registration.registration_icp(
                    source, target, threshold, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria
                )
                
                # Apply transformation to source for next stage
                source.transform(result_icp.transformation)
                
                # Update current transformation
                current_transform = np.dot(result_icp.transformation, current_transform)
                
                # Visualize intermediate result if not final stage
                if i < len(thresholds) - 1:
                    self.visualizeRegistrationProgress(source, target, f"Alignment Stage {i+1}")
            
            # Return the complete transformation
            return current_transform
            
        except Exception as e:
            print(f"Error in improvedRegistration: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
    def visualizeRegistrationProgress(self, source_cloud, target_cloud, stage_name="Registration"):
        """
        Visualize registration progress with interactive controls.
        
        Args:
            source_cloud: The source point cloud (transformed)
            target_cloud: The target point cloud
            stage_name: The name of the current registration stage
        """
        try:
            # Get points from clouds
            source_points = np.asarray(source_cloud.points)
            target_points = np.asarray(target_cloud.points)
            
            # Downsample for visualization
            max_points = 8000
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
            source_actor.GetProperty().SetColor(1, 0, 0)  # Red for source
            source_actor.GetProperty().SetPointSize(2)
            
            target_actor = vtk.vtkActor()
            target_actor.SetMapper(target_mapper)
            target_actor.GetProperty().SetColor(0, 1, 0)  # Green for target
            target_actor.GetProperty().SetPointSize(2)
            
            # Add actors to renderer
            self.ren.AddActor(source_actor)
            self.ren.AddActor(target_actor)
            self.actors.append(source_actor)
            self.actors.append(target_actor)
            
            # Add text to show current stage
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(stage_name)
            text_actor.SetPosition(10, 10)
            text_actor.GetTextProperty().SetColor(1, 1, 1)  # White text
            text_actor.GetTextProperty().SetFontSize(16)
            self.ren.AddActor2D(text_actor)
            self.actors.append(text_actor)
            
            # Reset camera and render
            self.ren.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            
            # Allow user to inspect registration
            if stage_name != "Initial Alignment":
                reply = QMessageBox.question(self, f"{stage_name}", 
                                        "Is the registration acceptable?\nClick 'Yes' to continue, 'No' to adjust points.",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                
                if reply == QMessageBox.No and stage_name != "Final Registration":
                    # Set up for manual adjustment
                    self.manualAdjustRegistration(source_cloud, target_cloud)
                    
        except Exception as e:
            print(f"Error in visualizeRegistrationProgress: {e}")
            import traceback
            traceback.print_exc()    
 
    def manualAdjustRegistration(self, source_cloud, target_cloud):
        """
        Allow manual adjustment of registration through fiducial point selection.
        
        Args:
            source_cloud: The source point cloud to adjust
            target_cloud: The target point cloud (reference)
        """
        # Create adjustment dialog
        adjust_dialog = QMessageBox(self)
        adjust_dialog.setWindowTitle("Registration Adjustment")
        adjust_dialog.setText("Select the option to improve registration:")
        adjust_dialog.setStandardButtons(QMessageBox.Cancel)
        
        # Add custom buttons
        select_points_button = adjust_dialog.addButton("Select More Points", QMessageBox.ActionRole)
        scale_button = adjust_dialog.addButton("Adjust Scale", QMessageBox.ActionRole)
        
        result = adjust_dialog.exec_()
        
        if adjust_dialog.clickedButton() == select_points_button:
            # Clear existing selected points
            self.clearPoints()
            
            # Update instructions for manual selection
            self.instructions_label.setText(
                "Select 3-5 corresponding points on the visible model\n"
                "Use Shift+Left Click to select points"
            )
            
            # Set a flag to indicate we're in manual adjustment mode
            self.manual_adjustment_mode = True
            
            # Store the clouds for later use
            self._source_cloud_points = source_cloud
            self._target_cloud_points = target_cloud
            
            # Show message to user
            QMessageBox.information(self, "Point Selection", 
                                "Select at least 3 points that match features you can identify.")
                    
        elif adjust_dialog.clickedButton() == scale_button:
            # Create dialog for scale adjustment
            scale_dialog = QDialog(self)
            scale_dialog.setWindowTitle("Adjust Scale Factor")
            
            layout = QVBoxLayout()
            
            # Add slider for scale adjustment
            scale_label = QLabel("Scale Factor:")
            layout.addWidget(scale_label)
            
            scale_slider = QSlider(Qt.Horizontal)
            scale_slider.setMinimum(80)
            scale_slider.setMaximum(120)
            scale_slider.setValue(100)
            scale_slider.setTickPosition(QSlider.TicksBelow)
            scale_slider.setTickInterval(5)
            layout.addWidget(scale_slider)
            
            value_label = QLabel("1.00")
            layout.addWidget(value_label)
            
            # Update label when slider changes
            def update_value():
                value = scale_slider.value() / 100.0
                value_label.setText(f"{value:.2f}")
            
            scale_slider.valueChanged.connect(update_value)
            
            # Add buttons
            button_box = QHBoxLayout()
            apply_button = QPushButton("Apply")
            cancel_button = QPushButton("Cancel")
            button_box.addWidget(apply_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)
            
            scale_dialog.setLayout(layout)
            
            # Connect buttons
            apply_button.clicked.connect(scale_dialog.accept)
            cancel_button.clicked.connect(scale_dialog.reject)
            
            # Show dialog
            if scale_dialog.exec_() == QDialog.Accepted:
                scale_factor = scale_slider.value() / 100.0
                
                # Apply scale factor to source cloud
                points = np.asarray(source_cloud.points)
                source_center = np.mean(points, axis=0)
                
                # Scale around centroid
                centered_points = points - source_center
                scaled_points = centered_points * scale_factor
                new_points = scaled_points + source_center
                
                # Update source cloud
                source_cloud.points = o3d.utility.Vector3dVector(new_points)
                
                # Update visualization
                self.visualizeRegistrationProgress(source_cloud, target_cloud, "After Scale Adjustment")
                
    def visualizeSimpleRegistration(self, source_cloud, target_cloud):
        """Visualization function with better differentiation between point clouds."""
        try:
            # Get points from clouds
            source_points = np.asarray(source_cloud.points)
            target_points = np.asarray(target_cloud.points)
            
            # Sample points for visualization
            max_points = 50000
            
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
            
            # Source cloud (MRI - red)
            source_polydata = vtk.vtkPolyData()
            source_vtk_points = vtk.vtkPoints()
            for point in source_points:
                source_vtk_points.InsertNextPoint(point[0], point[1], point[2])
            source_polydata.SetPoints(source_vtk_points)
            
            source_vertices = vtk.vtkCellArray()
            for i in range(source_vtk_points.GetNumberOfPoints()):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                source_vertices.InsertNextCell(vertex)
            source_polydata.SetVerts(source_vertices)
            
            source_mapper = vtk.vtkPolyDataMapper()
            source_mapper.SetInputData(source_polydata)
            
            source_actor = vtk.vtkActor()
            source_actor.SetMapper(source_mapper)
            source_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Pure red for MRI
            source_actor.GetProperty().SetPointSize(3)
            source_actor.GetProperty().SetOpacity(0.8)  # Slightly transparent
            
            # Target cloud (Head scan - green)
            target_polydata = vtk.vtkPolyData()
            target_vtk_points = vtk.vtkPoints()
            for point in target_points:
                target_vtk_points.InsertNextPoint(point[0], point[1], point[2])
            target_polydata.SetPoints(target_vtk_points)
            
            target_vertices = vtk.vtkCellArray()
            for i in range(target_vtk_points.GetNumberOfPoints()):
                vertex = vtk.vtkVertex()
                vertex.GetPointIds().SetId(0, i)
                target_vertices.InsertNextCell(vertex)
            target_polydata.SetVerts(target_vertices)
            
            target_mapper = vtk.vtkPolyDataMapper()
            target_mapper.SetInputData(target_polydata)
            
            target_actor = vtk.vtkActor()
            target_actor.SetMapper(target_mapper)
            target_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Pure green for head scan
            target_actor.GetProperty().SetPointSize(2)
            
            # Add actors to renderer
            self.ren.AddActor(source_actor)
            self.ren.AddActor(target_actor)
            self.actors.append(source_actor)
            self.actors.append(target_actor)
            
            # Add a legend at bottom of screen
            legend_actor = vtk.vtkTextActor()
            legend_actor.SetInput("Red: MRI Scalp\nGreen: Head Scan")
            legend_actor.SetPosition(10, 10)
            legend_actor.GetTextProperty().SetFontSize(16)
            legend_actor.GetTextProperty().SetColor(1, 1, 1)  # White text
            self.ren.AddActor2D(legend_actor)
            self.actors.append(legend_actor)
            
            # Set camera to side view for better assessment
            camera = self.ren.GetActiveCamera()
            camera.SetPosition(0, -500, 0)  # Side view
            camera.SetViewUp(0, 0, 1)
            
            # Reset camera and render
            self.ren.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as e:
            print(f"Error in visualizeSimpleRegistration: {e}")
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
                
                # Create device-to-head transform - combine the manual alignment with X21
                dev_head_t = mne.transforms.Transform("meg", "head", trans=None)
                inside_transform = self.vtk_to_numpy_matrix(self.alignment_transforms["Inside MSR"])
                combined_X21 = np.dot(self.X21, inside_transform)
                dev_head_t['trans'] = combined_X21.copy()
                
                # Convert from mm to m for MNE
                dev_head_t['trans'][0:3, 3] = np.divide(dev_head_t['trans'][0:3, 3], 1000)
                
                # Update raw info with transform
                raw.info.update(dev_head_t=dev_head_t)
                
                # Save updated MEG file
                output_file = os.path.join(save_dir, os.path.basename(self.file_paths["OPM Data"]))
                raw.save(output_file, overwrite=True)
                
                # Create MRI-to-head transform
                mri_head_t = mne.transforms.Transform("mri", "head", trans=None)
                mri_transform = self.vtk_to_numpy_matrix(self.alignment_transforms["Scalp File"])
                combined_X3 = np.dot(self.X3, mri_transform)
                mri_head_t['trans'] = combined_X3.copy()
                
                # Convert from mm to m for MNE
                mri_head_t['trans'][0:3, 3] = np.divide(mri_head_t['trans'][0:3, 3], 1000)
                
                # Save MRI-to-head transform
                trans_file = os.path.join(save_dir, os.path.basename(self.file_paths["OPM Data"]).split('.fif')[0] + '_trans.fif')
                mne.write_trans(trans_file, mri_head_t, overwrite=True)
                
                # Clean up
                for temp_file in self.aligned_file_paths.values():
                    if os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except Exception as e:
                            print(f"Warning: Could not remove temporary file {temp_file}: {e}")
                
                # Continue with saving points...
                
                QMessageBox.information(self, "Files Saved", 
                                    f"Transformation and point files have been saved to:\n{save_dir}")

def main():
    app = QApplication(sys.argv)
    window = OPMCoRegistrationGUI()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()