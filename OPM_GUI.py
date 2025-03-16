
import os
import mne
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QProgressBar, QFrame
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QTimer
import vtk
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import sys


class OPMCoRegistrationGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.selected_points = []
        self.point_actors = []
        self.file_paths = {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("OPM-MEG Co-Registration")
        self.setGeometry(100, 100, 1200, 600)
        self.setStyleSheet("background-color: #2e2e2e; color: white;")
        
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Left section (25%)
        left_frame = QFrame(self)
        left_frame.setFixedWidth(int(self.width() * 0.25))
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
            left_layout.addLayout(group_layout)
            
            button = QPushButton("Select", self)
            button.setFont(font)
            button.setStyleSheet("background-color: #555555; color: white; border-radius: 5px; padding: 5px;")
            button.clicked.connect(lambda checked, t=file_type: self.selectFile(t))
            
            file_label = QLabel("No file selected", self)
            file_label.setFont(font)
            
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
        left_layout.addWidget(self.progress_bar)
        
        # Middle section (50%)
        middle_frame = QFrame(self)
        middle_frame.setFixedWidth(int(self.width() * 0.50))
        middle_layout = QVBoxLayout()
        middle_frame.setLayout(middle_layout)
        main_layout.addWidget(middle_frame)
        
        self.instructions_label = QLabel("", self)
        self.instructions_label.setFont(QFont("Arial", 14))
        self.instructions_label.setAlignment(Qt.AlignCenter)
        middle_layout.addWidget(self.instructions_label)
        
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        middle_layout.addWidget(self.vtk_widget)
        
        self.ren = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.ren)
        
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.iren.AddObserver("LeftButtonPressEvent", self.onLeftClick)
        
        # Right section (25%)
        right_frame = QFrame(self)
        right_frame.setFixedWidth(int(self.width() * 0.25))
        right_layout = QVBoxLayout()
        right_frame.setLayout(right_layout)
        main_layout.addWidget(right_frame)
        
        # Add Clear and Continue buttons
        button_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Points", self)
        self.clear_button.setFont(font)
        self.clear_button.setStyleSheet("background-color: #FF5733; color: white; font-weight: bold;")
        self.clear_button.clicked.connect(self.clearPoints)
        button_layout.addWidget(self.clear_button)
        
        self.continue_button = QPushButton("Continue", self)
        self.continue_button.setFont(font)
        self.continue_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.continue_button.setEnabled(False)
        button_layout.addWidget(self.continue_button)
        
        right_layout.addLayout(button_layout)
        
        # Add point coordinate boxes
        self.point_labels = []
        for i in range(7):
            point_label = QLabel(f"Point {i+1}: ", self)
            point_label.setFont(font)
            right_layout.addWidget(point_label)
            self.point_labels.append(point_label)
        
        right_layout.addStretch()
        
        self.show()
    
    def selectFile(self, file_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, f"Select {file_type} File", "", self.extensions[file_type], options=options)
        if file_path:
            self.file_paths[file_type] = file_path
            self.labels[["Inside MSR", "Outside MSR", "Scalp File", "OPM Data"].index(file_type)].setText(file_path)
        if self.file_paths.get("Inside MSR"):
            self.load_button.setEnabled(True)
    
    def onLeftClick(self, obj, event):
        if self.iren.GetShiftKey():
            click_pos = self.iren.GetEventPosition()
            picker = vtk.vtkPointPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, self.ren)
            picked_position = picker.GetPickPosition()
            if picked_position:
                self.addPoint(picked_position)
    
    def addPoint(self, position):
        point = vtk.vtkSphereSource()
        point.SetCenter(position)
        point.SetRadius(2.0)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(point.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)
        
        self.ren.AddActor(actor)
        self.point_actors.append(actor)
        self.selected_points.append(position)
        
        self.updatePointLabels()
        self.vtk_widget.GetRenderWindow().Render()
    
    def updatePointLabels(self):
        for i, point in enumerate(self.selected_points):
            self.point_labels[i].setText(f"Point {i+1}: {point}")
    
    def loadModel(self):
        self.load_button.setText("Loading...")
        self.progress_bar.setValue(0)
        QTimer.singleShot(100, self.updateProgress)
    
    def updateProgress(self):
        for i in range(1, 101):
            QTimer.singleShot(i * 50, lambda: self.progress_bar.setValue(i))
        QTimer.singleShot(5000, self.finishLoading)
    
    def finishLoading(self):
        self.load_button.setText("Load")
        self.progress_bar.setValue(100)
        self.continue_button.setEnabled(True)
        self.loadPointCloud("Inside MSR")
    
    def loadPointCloud(self, file_type):
        file_path = self.file_paths[file_type]
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_path)
        reader.Update()
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        self.ren.AddActor(actor)
        self.vtk_widget.GetRenderWindow().Render()
    
    def clearPoints(self):
        for actor in self.point_actors:
            self.ren.RemoveActor(actor)
        self.point_actors.clear()
        self.selected_points.clear()
        self.updatePointLabels()
        self.vtk_widget.GetRenderWindow().Render()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OPMCoRegistrationGUI()
    window.show()
    sys.exit(app.exec_())
