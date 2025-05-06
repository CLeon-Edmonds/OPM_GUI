#!/usr/bin/env python

import sys
import traceback

print("Starting test_gui.py")
print(f"Python version: {sys.version}")

try:
    import vtk
    print(f"VTK version: {vtk.vtkVersion().GetVTKVersion()}")
except ImportError as e:
    print(f"Error importing VTK: {e}")

try:
    from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
    from PyQt5.QtCore import Qt
    print("PyQt5 imported successfully")
except ImportError as e:
    print(f"Error importing PyQt5: {e}")
    sys.exit(1)

try:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    print("QVTKRenderWindowInteractor imported successfully")
except ImportError as e:
    print(f"Error importing QVTKRenderWindowInteractor: {e}")
except Exception as e:
    print(f"Error initializing QVTKRenderWindowInteractor: {e}")
    traceback.print_exc()

class SimpleVTKTest(QWidget):
    def __init__(self):
        super().__init__()
        print("Initializing SimpleVTKTest...")
        self.initUI()
        
    def initUI(self):
        try:
            print("Setting up window...")
            self.setWindowTitle("VTK Test")
            self.setGeometry(100, 100, 800, 600)
            
            layout = QVBoxLayout()
            self.setLayout(layout)
            
            print("Creating test button...")
            button = QPushButton("Test Button", self)
            button.clicked.connect(self.buttonClicked)
            layout.addWidget(button)
            
            print("Creating VTK widget...")
            self.vtk_widget = QVTKRenderWindowInteractor(self)
            layout.addWidget(self.vtk_widget)
            
            print("Creating VTK renderer...")
            self.renderer = vtk.vtkRenderer()
            self.renderer.SetBackground(0.1, 0.2, 0.4)
            self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
            
            print("Creating VTK interactor...")
            self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
            self.iren.Initialize()
            
            print("UI initialized")
        except Exception as e:
            print(f"Error in initUI: {e}")
            traceback.print_exc()
            
    def buttonClicked(self):
        print("Button clicked")
        
if __name__ == "__main__":
    try:
        print("Creating QApplication...")
        app = QApplication(sys.argv)
        
        print("Creating SimpleVTKTest...")
        widget = SimpleVTKTest()
        
        print("Showing window...")
        widget.show()
        
        print("Starting event loop...")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()