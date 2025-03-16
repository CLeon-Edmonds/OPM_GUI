import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import open3d as o3d
import numpy as np
import mne
import os
import copy

sensor_length = 6e-3

class CoregistrationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OPM-MEG Coregistration")

        self.file_paths = {
            "Inside MSR .ply": None,
            "Outside MSR .ply": None,
            "Scalp .stl": None,
            "OMP data .fif": None
        }

        self.create_widgets()

    def create_widgets(self):
        # Create buttons for loading files
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.labels = {}
        self.buttons = {}
        for file_type in self.file_paths.keys():
            label = tk.Label(self.button_frame, text=file_type, font=("Arial", 12))
            label.pack(pady=5)
            self.labels[file_type] = label

            button = tk.Button(self.button_frame, text=f"Load {file_type}", command=lambda ft=file_type: self.load_file(ft))
            button.pack(pady=5)
            self.buttons[file_type] = button

        # Create canvas for point cloud visualization
        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.canvas = tk.Canvas(self.canvas_frame, width=600, height=600)
        self.canvas.pack()

        # Create error feedback section
        self.error_frame = tk.Frame(self.root)
        self.error_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        self.error_label = tk.Label(self.error_frame, text="Error Feedback", font=("Arial", 16))
        self.error_label.pack(pady=5)

        # Create Load button
        self.load_button = tk.Button(self.root, text="Load", command=self.start_loading)
        self.load_button.pack(side=tk.BOTTOM, pady=10)

        # Create progress bar
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(side=tk.BOTTOM, pady=10)

    def load_file(self, file_type):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_paths[file_type] = file_path
            self.labels[file_type].config(text=os.path.basename(file_path))
            print(f"{file_type} file loaded: {file_path}")

    def start_loading(self):
        self.progress['value'] = 0
        self.root.update_idletasks()
        self.load_point_clouds()

    def load_point_clouds(self):
        for i, (file_type, file_path) in enumerate(self.file_paths.items()):
            if file_path:
                self.display_point_cloud(file_path)
                self.progress['value'] += 25
                self.root.update_idletasks()
        messagebox.showinfo("Info", "All files loaded successfully!")

    def display_point_cloud(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        o3d.visualization.draw_geometries([pcd])

def main():
    root = tk.Tk()
    app = CoregistrationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()