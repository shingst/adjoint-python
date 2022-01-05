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
	
forward_file="/home/sven/exa/adjoint/forward/output/pointsource-2.vtk"
adjoint_file="/home/sven/exa/adjoint/adjoint/outputA/adaptive2a-10.vtk"

data=readUnstructuredGrid(forward_file)
adj=readUnstructuredGrid(adjoint_file)

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
result:vtk.vtkUnstructuredGrid=interp.GetOutput()#TODO use pipelines

# result.SetCells(data.GetCellTypesArray(),data.GetCells())








# writer:vtk.vtkUnstructuredGridWriter =vtk.vtkUnstructuredGridWriter()
# writer.SetFileName("test2a.vtk")
# writer.SetInputData(result)
# # writer.SetInputConnection(0,interp.GetOutputPort(0))
# writer.Write()
