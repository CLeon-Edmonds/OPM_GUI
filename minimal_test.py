import vtk

reader = vtk.vtkPLYReader()
reader.SetFileName("/Volumes/CHRISDRIVE1/GUI_test/Ania/Ania_Inside.ply")
try:
    reader.Update()
    print("PLY file loaded successfully")
except Exception as e:
    print(f"Error loading PLY file: {e}")