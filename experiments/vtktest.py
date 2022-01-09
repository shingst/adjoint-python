import vtkmodules.all as vtk
import numpy as np
from vtkmodules.util import numpy_support

def readUnstructuredGrid(filename) -> vtk.vtkUnstructuredGrid:
	reader=vtk.vtkUnstructuredGridReader()
	reader.SetFileName(filename)
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	return reader.GetOutput()

filename="/home/sven/exa/adjoint/forward/output/pointsource-2.vtk"
filename2="/home/sven/exa/adjoint/forward/output/pointsource-20.vtk"

data=readUnstructuredGrid(filename)
data2=readUnstructuredGrid(filename2)

cells=data.GetCells()
ncells=cells.GetNumberOfCells()

n_points=data.GetNumberOfPoints()

pts:vtk.vtkPoints=data.GetPoints()

# pointsset:vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
# pointsset.SetPoints(pts)
x=np.ones(3)*1.2
y=np.ones(3)*1.2
for i in range(data.GetNumberOfPoints()):
	data.GetPoint(i,x)
	data2.GetPoint(i,y)
	assert (x==y).all()

# pts:vtk.vtkPoints=data.GetPoints()
pointdata=data.GetPointData()
vtkQ=pointdata.GetArray('Q')
Q=numpy_support.vtk_to_numpy(vtkQ)


for i in range(5):
	mx=np.argmax(Q[:,i])
	mn=np.argmin(Q[:,i])
	data.GetPoint(mn, x)
	print(f"max Q[:,{i}] location {x},{mx}")
	data.GetPoint(mx, x)
	print(f"min Q[:,{i}] location {x},{mn}")
print("tmp")



