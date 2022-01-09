import vtkmodules.all as vtk
import numpy as np
from vtkmodules.util import numpy_support
import os
from pathlib import Path

def readUnstructuredGrid(filename) -> vtk.vtkUnstructuredGrid:
	reader=vtk.vtkUnstructuredGridReader()
	reader.SetFileName(filename)
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	return reader.GetOutput()

def plot_numpy_array(arr, ugrid):
	m_max=arr.max()
	m_min=arr.min()

	magdata: vtk.vtkArray=numpy_support.numpy_to_vtk(arr)
	magdata.SetName("Magnitude")
	#! this reuses the previous pointset
	# pointsset.GetPointData().AddArray(magdata) #TODO set name
	ugrid.GetPointData().SetActiveScalars('Magnitude')
	ugrid.GetPointData().SetScalars(magdata)
	# ugrid.SetCells(data.GetCellTypesArray(), data.GetCells())

	mapper: vtk.vtkDataSetMapper=vtk.vtkDataSetMapper()
	mapper.SetInputData(ugrid)
	mapper.SetScalarRange(m_min, m_max)

	actor: vtk.vtkActor=vtk.vtkActor()
	actor.SetMapper(mapper)

	window=vtk.vtkRenderWindow()
	# Sets the pixel width, length of the window.
	window.SetSize(2000, 2000)

	interactor=vtk.vtkRenderWindowInteractor()
	interactor.SetRenderWindow(window)

	renderer=vtk.vtkRenderer()
	window.AddRenderer(renderer)

	renderer.AddActor(actor)
	# Setting the background to blue.
	renderer.SetBackground(0.1, 0.1, 0.4)

	window.Render()
	interactor.Start()


def interpolate(fwdPoints:vtk.vtkUnstructuredGrid,adj:vtk.vtkUnstructuredGrid)-> vtk.vtkUnstructuredGrid:
	gaussian_kernel=vtk.vtkGaussianKernel()
	gaussian_kernel.SetSharpness(4)
	gaussian_kernel.SetRadius(1)

	interp=vtk.vtkPointInterpolator()
	interp.SetInputData(fwdPoints)
	interp.SetSourceData(adj)
	interp.SetKernel(gaussian_kernel)
	interp.Update()
	return interp.GetOutput()  #TODO use pipelines


def handle_two_files(forward_file,adjoint_file):
	data=readUnstructuredGrid(forward_file)
	adj=readUnstructuredGrid(adjoint_file)

	#TODO check bounds and timestamps

	pts: vtk.vtkPoints=data.GetPoints()

	onlypoints: vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
	onlypoints.SetPoints(pts)

	adj_interpolated: vtk.vtkUnstructuredGrid=interpolate(onlypoints, adj)  #TODO use pipelines

	pointdata: vtk.vtkPointData=data.GetPointData()
	vtkQ=pointdata.GetArray('Q')
	fQ=numpy_support.vtk_to_numpy(vtkQ)

	pointdata: vtk.vtkPointData=adj_interpolated.GetPointData()
	vtkQ=pointdata.GetArray('Q')
	aQ=numpy_support.vtk_to_numpy(vtkQ)

	magnitude=np.abs(np.sum(fQ*aQ, axis=1))  #scalar product for each point)
	impact=np.log10(magnitude+1e-9)
	# impact=magnitude

	onlypoints.SetCells(data.GetCellTypesArray(), data.GetCells())#needed for visualization
	return impact,onlypoints

# fws=list((Path("/home/sven/exa/adjoint/forward/output/")).glob("pointsource-*.vtk"))
# fws=os.listdir("/home/sven/exa/adjoint/forward/output/")
numfiles=35
#TODO better file reading
forward_file="/home/sven/exa/adjoint/forward/output/pointsource-10.vtk"
adjoint_file="/home/sven/exa/adjoint/adjoint/outputA/adaptive2a-20.vtk"

if True:
	for i in range(1,numfiles):
		forward_file=f"/home/sven/exa/adjoint/forward/output/pointsource-{i}.vtk"
		adjoint_file_file=f"/home/sven/exa/adjoint/adjoint/outputA/constsource-{numfiles-i}.vtk"
		impact, onlypoints=handle_two_files(forward_file, adjoint_file)
		i_normalized=impact/impact.max()
		if i==1:
			refine=np.zeros(impact.size)
		refine=np.logical_or(refine, i_normalized>0.9)
		

# result.SetCells(data.GetCellTypesArray(),data.GetCells())

# x=np.ones(3)*1.2
# y=np.ones(3)*1.2
# for i in range(data.GetNumberOfPoints()):
# 	data.GetPoint(i,x)
# 	adj_interpolated.GetPoint(i,y)
# 	assert (x==y).all()






# i_normalized=impact/impact.max()
# refine=i_normalized>0.7
# plot_numpy_array(refine.astype(int),onlypoints)#! warning alters onlypoints


# impact,onlypoints=handle_two_files(forward_file,adjoint_file)
# i_normalized=impact/impact.max()
# refine=i_normalized>0.7
plot_numpy_array(refine.astype(int),onlypoints)#! warning alters onlypoints


a=0
# writer:vtk.vtkUnstructuredGridWriter =vtk.vtkUnstructuredGridWriter()
# writer.SetFileName("magnitude.vtk")
# writer.SetInputData(onlypoints)
# # writer.SetInputConnection(0,interp.GetOutputPort(0))
# writer.Write()
