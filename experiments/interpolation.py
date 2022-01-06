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
	
forward_file="/home/sven/exa/adjoint/forward/output/pointsource-20.vtk"
adjoint_file="/home/sven/exa/adjoint/adjoint/outputA/adaptive-20.vtk"

data=readUnstructuredGrid(forward_file)
adj=readUnstructuredGrid(adjoint_file)

#TODO check bounds and timestamps





pts:vtk.vtkPoints=data.GetPoints()

pointsset:vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
pointsset.SetPoints(pts)


# writer:vtk.vtkUnstructuredGridWriter =vtk.vtkUnstructuredGridWriter()
# writer.SetFileName("ps.vtk")
# writer.SetInputData(pointsset)
# writer.Write()

gaussian_kernel = vtk.vtkGaussianKernel()
gaussian_kernel.SetSharpness(4)
gaussian_kernel.SetRadius(1)

interp=vtk.vtkPointInterpolator()
interp.SetInputData(pointsset)
interp.SetSourceData(adj)
interp.SetKernel(gaussian_kernel)
interp.Update()
adj_interpolated:vtk.vtkUnstructuredGrid=interp.GetOutput()#TODO use pipelines

# result.SetCells(data.GetCellTypesArray(),data.GetCells())

x=np.ones(3)*1.2
y=np.ones(3)*1.2
for i in range(data.GetNumberOfPoints()):
	data.GetPoint(i,x)
	adj_interpolated.GetPoint(i,y)
	assert (x==y).all()

pointdata=data.GetPointData()
vtkQ=pointdata.GetArray('Q')
fQ=numpy_support.vtk_to_numpy(vtkQ)

pointdata=adj_interpolated.GetPointData()
vtkQ=pointdata.GetArray('Q')
aQ=numpy_support.vtk_to_numpy(vtkQ)

magnitude=np.abs(np.sum(fQ*aQ,axis=1)) #scalar product for each point)

#! this reuses the previous pointset
pointsset.GetPointData().AddArray(numpy_support.numpy_to_vtk(magnitude)) #TODO set name
pointsset.SetCells(data.GetCellTypesArray(),data.GetCells()) 

a=0
writer:vtk.vtkUnstructuredGridWriter =vtk.vtkUnstructuredGridWriter()
writer.SetFileName("magnitude.vtk")
writer.SetInputData(pointsset)
# writer.SetInputConnection(0,interp.GetOutputPort(0))
writer.Write()
