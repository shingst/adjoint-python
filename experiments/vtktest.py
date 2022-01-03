import vtkmodules.all as vtk
import numpy as np
from vtkmodules.util import numpy_support


reader = vtk.vtkUnstructuredGridReader()
filename="/home/sven/exa/adjoint/forward/output/pointsource-1.vtk"
reader.SetFileName(filename)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
data:vtk.vtkUnstructuredGrid = reader.GetOutput()
cells=data.GetCells()
ncells=cells.GetNumberOfCells()

n_points=data.GetNumberOfPoints()
x=np.ones(3)*1.2
for i in range(10):
	data.GetPoint(i,x)
	print(x)

pts:vtk.vtkPoints=data.GetPoints()
pointdata=data.GetPointData()
vtkQ=pointdata.GetArray('Q')
Q=numpy_support.vtk_to_numpy(vtkQ)


for i in range(5):
	mx=np.argmax(Q[:,i])
	mn=np.argmin(Q[:,i])
	data.GetPoint(mn, x)
	print(f"max Q[:,{i}] location {x}")
	data.GetPoint(mx, x)
	print(f"min Q[:,{i}] location {x}")
print("tmp")

