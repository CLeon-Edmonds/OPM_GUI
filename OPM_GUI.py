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

sensor_length = 6e-3

class WorkerThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(object)
    
    def __init__(self, task, args=None):
        super().__init__()
        self.task = task
        self.args = args if args is not None else []
        self.result = None
        
    def run(self):
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
            #Actual MRI to head registration
            try:
                # Here ywe would call the head_to_mri function but it's simulted now. 
                for i in range(101):
                    self.progress_signal.emit(i)
                    self.msleep(50)
                self.finished_signal.emit(None)
            except Exception as e:
                self.finished_signal.emit(str(e))
        elif self.task == "final_registration":
            #This would perform the actual head point check and file writing
            try:
                # Simulate final steps, I wil do that later. 
                for i in range(101):
                    self.progress_signal.emit(i)
                    self.msleep(50)
                self.finished_signal.emit(None)
            except Exception as e:
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
        self.current_stage = "inside_msr"  #Tracks the current stage of the workflow, to be deleted after it is all working and optimized probably. Just ignore this commentf or now
        self.known_sensor_positions = np.array([
            [102.325, 0.221, 16.345],
            [92.079, 66.226, -27.207],
            [67.431, 113.778, -7.799],
            [-0.117, 138.956, -5.576],
            [-67.431, 113.778, -7.799],
            [-92.079, 66.226, -27.207],
            [-102.325, 0.221, 16.345]
        ])
        self.fiducial_labels = ["Right Pre-auricular", "Left Pre-auricular", "Nasion"]
        self.point_labels = []
        self.actors = [] #Store all actors. 
        
        # MAtrix transformation 
        self.X1 = None  #Inside MSR to helmet transform
        self.X2 = None  #Standard transform
        self.X21 = None  #Combined transform
        self.X3 = None  #MRI to head transform
        
        self.initUI()

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
        self.ren.SetBackground(0.1, 0.1, 0.1)  # Maybe we will change the background eventually 
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

        #Button controls
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
        
        self.continue_button = QPushButton("Continue", self)
        self.continue_button.setFont(font)
        self.continue_button.setStyleSheet("background-color: #007BFF; color: white; font-weight: bold;")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continueWorkflow)
        button_grid_layout.addWidget(self.continue_button, 1, 1)

        right_layout.addLayout(button_grid_layout)

        #Point coordinates display
        self.coordinates_layout = QVBoxLayout()
        self.coordinates_label = QLabel("Selected Point Coordinates:", self)
        self.coordinates_label.setFont(QFont("Arial", 14))
        self.coordinates_layout.addWidget(self.coordinates_label)
        
        #Add point coordinate labels
        self.point_coordinates_labels = []
        for i in range(7):  #Should stop at 7 but do we want to keep?
            label = QLabel("", self)
            label.setFont(font)
            self.point_coordinates_labels.append(label)
            self.coordinates_layout.addWidget(label)

        right_layout.addLayout(self.coordinates_layout)
        
        #Save button for final step
        self.save_button = QPushButton("Save Results", self)
        self.save_button.setFont(font)
        self.save_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.saveResults)
        right_layout.addWidget(self.save_button)
        self.save_button.hide()  #Hide until needed
        
        right_layout.addStretch()

        # Set up worker thread
        self.worker = None

        # Initialise instructions
        self.updateInstructions()

        self.show()
        
        # Initialise VTK interactor
        self.iren.Initialize()

    def updateInstructions(self):
        if self.current_stage == "inside_msr":
            self.instructions_label.setText(
                "Step 1: "
                "Select the 7 helmet label points from left to right\n"
                "Use Shift+Left Click to select points"
            )
            self.status_label.setText("Inside MSR Point Selection Status")
            
            #7 points shuld show up here.
            self.updatePointStatusPanel(7)
            
        elif self.current_stage == "outside_msr":
            self.instructions_label.setText(
                "Step 2: Select the 3 fiducial points in this order:\n"
                "1. Right pre-auricular\n"
                "2. Left pre-auricular\n"
                "3. Nasion\n"
                "Use Shift+Left Click to select points"
            )
            self.status_label.setText("Fiducial Point Selection Status")
            
            #Fiducial popints to show up here
            self.updatePointStatusPanel(3, fiducials=True)
            
        elif self.current_stage == "mri_scalp":
            self.instructions_label.setText(
                "Step 3: Performing MRI scalp to head registration\n"
                "Preview will be shown when complete"
            )
        elif self.current_stage == "finished":
            self.instructions_label.setText(
                "Co-registration complete!\n"
                "Click 'Save Results' to save the transformation files"
            )
            self.save_button.show()
            self.save_button.setEnabled(True)

    def updatePointStatusPanel(self, num_points, fiducials=False):
        for i in range(len(self.distance_labels)):
            self.distance_labels[i].setParent(None)
            self.distance_boxes[i].setParent(None)
        
        self.distance_labels.clear()
        self.distance_boxes.clear()
        
        #Might change fonts later
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
            self.file_paths[file_type] = file_path
            self.labels[["Inside MSR", "Outside MSR", "Scalp File", "OPM Data"].index(file_type)].setText(
                os.path.basename(file_path))
            
            if self.file_paths.get("Inside MSR"):
                self.load_button.setEnabled(True)

    def onShiftLeftClick(self, obj, event):
        if self.iren.GetShiftKey():
            click_pos = self.iren.GetEventPosition()
            picker = vtk.vtkPointPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, self.ren)
            picked_position = picker.GetPickPosition()
            
            if picked_position != (0, 0, 0): 
                self.addPoint(picked_position)
                
                #Update confirm button
                if self.current_stage == "inside_msr" and len(self.selected_points) == 7:
                    self.confirm_button.setEnabled(True)
                elif self.current_stage == "outside_msr" and len(self.selected_points) == 3:
                    self.confirm_button.setEnabled(True)

    def addPoint(self, position):
        point = vtk.vtkSphereSource()
        point.SetCenter(position)
        point.SetRadius(3.0)  #We can change this if the buttons are too small
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(point.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
    
        if self.current_stage == "inside_msr":
            actor.GetProperty().SetColor(1, 0, 0) #This is red
        else:
            actor.GetProperty().SetColor(0, 1, 0)  #This is green
        
        self.ren.AddActor(actor)
        self.actors.append(actor)
        self.point_actors.append(actor)
        self.selected_points.append(position)
        
        self.updatePointCoordinates()
        
        # Update status
        idx = len(self.selected_points) - 1
        if self.current_stage == "inside_msr" and idx < 7:
            self.distance_labels[idx].setText(f"Point {idx+1}: Selected")
        elif self.current_stage == "outside_msr" and idx < 3:
            self.distance_labels[idx].setText(f"{self.fiducial_labels[idx]}: Selected")
        
        self.vtk_widget.GetRenderWindow().Render()

    def updatePointCoordinates(self):
        #Clear previous labels
        for label in self.point_coordinates_labels:
            label.setText("")
        
        #Update with current points
        max_points = min(len(self.selected_points), len(self.point_coordinates_labels))
        for i in range(max_points):
            pos = self.selected_points[i]
            if self.current_stage == "inside_msr":
                self.point_coordinates_labels[i].setText(f"Point {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
            else:
                label = self.fiducial_labels[i] if i < len(self.fiducial_labels) else f"Point {i+1}"
                self.point_coordinates_labels[i].setText(f"{label}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

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
        
        #Clear previous actors
        for actor in self.actors:
            self.ren.RemoveActor(actor)
        self.actors = []
        
        #Load new point cloud
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_path)
        reader.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        self.ren.AddActor(actor)
        self.actors.append(actor)
        
        #Reset camera to fit the model, at a different point we will see if we can get away without this. 
        self.ren.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def clearPoints(self):
        for actor in self.point_actors:
            self.ren.RemoveActor(actor)
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
        
        #Clear coordinates display
        for label in self.point_coordinates_labels:
            label.setText("")
        
        self.confirm_button.setEnabled(False)
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
            
            #Update coordinates
            self.updatePointCoordinates()
            
            #Disable confirm if we no longer have enough points
            if ((self.current_stage == "inside_msr" and len(self.selected_points) < 7) or 
                (self.current_stage == "outside_msr" and len(self.selected_points) < 3)):
                self.confirm_button.setEnabled(False)
            
            self.vtk_widget.GetRenderWindow().Render()

    def confirmPoints(self):
        if self.current_stage == "inside_msr":
            self.confirmInsideMSR()
        elif self.current_stage == "outside_msr":
            self.confirmOutsideMSR()

    def confirmInsideMSR(self):
        #Calculate and display distances from known points, maybe we will need to update this. 
        for i, point in enumerate(self.selected_points):
            if i < len(self.known_sensor_positions):
                distance = np.linalg.norm(np.array(point) - self.known_sensor_positions[i])
                self.distance_labels[i].setText(f"Point {i+1}: {distance:.2f} mm")
                self.distance_boxes[i].setText(f"{distance:.2f} mm")
                
                #Color based on distance
                if distance > 3:
                    self.distance_boxes[i].setStyleSheet("background-color: #FF5733; color: white;")
                else:
                    self.distance_boxes[i].setStyleSheet("background-color: #4CAF50; color: white;")
        
        # Store the transform matrix (in a real implementation, will do this later, ignore for now)
        # self.X1 = head_to_helmet(self.file_paths["Inside MSR"], list(range(1, 8)))
        
        #Enable continue button
        self.continue_button.setEnabled(True)

    def confirmOutsideMSR(self):
        for i, point in enumerate(self.selected_points):
            if i < len(self.fiducial_labels):
                self.distance_labels[i].setText(f"{self.fiducial_labels[i]}: Confirmed")
                self.distance_boxes[i].setText("Confirmed")
                self.distance_boxes[i].setStyleSheet("background-color: #4CAF50; color: white;")
        
        self.continue_button.setEnabled(True)

    def continueWorkflow(self):
        if self.current_stage == "inside_msr":
            self.moveToOutsideMSR()
        elif self.current_stage == "outside_msr":
            self.moveToMRIScalp()

    def moveToOutsideMSR(self):
        # Prepare GUI for Outside MSR stage
        self.current_stage = "outside_msr"
        self.updateInstructions()

        self.clearPoints()
        self.loadPointCloud("Outside MSR")
        
        self.continue_button.setEnabled(False)

    def moveToMRIScalp(self):
        # This would perform all the remaining corregistration steps, we will look a this later
        self.current_stage = "mri_scalp"
        self.updateInstructions()
        
        #Start worker thread for processing
        self.progress_bar.setValue(0)
        self.worker = WorkerThread("final_registration")
        self.worker.progress_signal.connect(self.updateProgress)
        self.worker.finished_signal.connect(self.finalizeCoregistration)
        self.worker.start()

    def finalizeCoregistration(self, error=None):
        if error:
            QMessageBox.critical(self, "Error", f"An error occurred during co-registration: {error}")
            return
        
        QMessageBox.information(self, "Co-registration Complete", 
                              "The co-registration process has been completed successfully.")
        
        #Update GUI for final stage
        self.current_stage = "finished"
        self.updateInstructions()
        
        #Disable buttons that are no longer needed, might make them dissapear later, we will see. 
        self.clear_button.setEnabled(False)
        self.reverse_button.setEnabled(False)
        self.confirm_button.setEnabled(False)
        self.continue_button.setEnabled(False)

    def saveResults(self):
        #Open file dialog for saving results
        options = QFileDialog.Options()
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Results", "", options=options)
        
        if save_dir:
            try:
                # Simulate saving files, for now
                output_file = os.path.join(save_dir, "meg_trans.fif")
                trans_file = os.path.join(save_dir, "mri_trans.fif")
                
                #This would actually call write_output function, ignore for now
                #write_output(self.file_paths["OPM Data"], self.X21, self.X3)
                
                QMessageBox.information(self, "Files Saved", 
                                       f"Transformation files have been saved to:\n{save_dir}")
                
                #Ask if user wants to exit?
                reply = QMessageBox.question(self, "Exit Application", 
                                            "Would you like to exit the application?",
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                
                if reply == QMessageBox.Yes:
                    self.close()
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while saving: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OPMCoRegistrationGUI()
    sys.exit(app.exec_())