import vtkmodules.all as vtk
import numpy as np
from vtkmodules.util import numpy_support
import os
from pathlib import Path
from string import Template

def readUnstructuredGrid(filename) -> vtk.vtkUnstructuredGrid:
	reader=vtk.vtkUnstructuredGridReader()
	reader.SetFileName(filename)
	reader.ReadAllVectorsOn()
	reader.ReadAllScalarsOn()
	reader.Update()
	return reader.GetOutput()

def writeUnstructuredGrid(filename,ugrid):
	writer:vtk.vtkUnstructuredGridWriter =vtk.vtkUnstructuredGridWriter()
	writer.SetFileName(filename)
	writer.SetInputData(ugrid)
	# writer.SetInputConnection(0,interp.GetOutputPort(0))
	writer.Write()

def write_numpy_array(arr, ugrid_to_copy,filename):
	m_max=arr.max()
	m_min=arr.min()

	ugrid: vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
	ugrid.DeepCopy(ugrid_to_copy)

	magdata: vtk.vtkArray=numpy_support.numpy_to_vtk(arr)
	magdata.SetName("Magnitude")
	# pointsset.GetPointData().AddArray(magdata) #TODO set name
	ugrid.GetPointData().SetActiveScalars('Magnitude')
	ugrid.GetPointData().SetScalars(magdata)
	writeUnstructuredGrid(filename,ugrid)

def plot_numpy_array(arr, ugrid_to_copy):
	m_max=arr.max()
	m_min=arr.min()
	
	ugrid:vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
	ugrid.DeepCopy(ugrid_to_copy)

	magdata: vtk.vtkArray=numpy_support.numpy_to_vtk(arr)
	magdata.SetName("Magnitude")
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

def adjoint_over_time(forward_template:Template,adjoint_template:Template,start, finish):
	"""
	:param forward_template: 
	:param adjoint_template: 
	:param start: 
	:param finish: larger than the last index
	:return: 
	"""
	data=readUnstructuredGrid(forward_template.substitute({'file':'1'}))
	pts: vtk.vtkPoints=data.GetPoints()

	onlypoints: vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
	onlypoints.SetPoints(pts)
	num_pts=pts.GetNumberOfPoints()
	
	adjoints=np.zeros((finish-start,num_pts,5))
	for i in range(finish-start):
		adj:vtk.vtkUnstructuredGrid=readUnstructuredGrid(adjoint_template.substitute({'file': i+start}))
		adj_interpolated: vtk.vtkUnstructuredGrid=interpolate(onlypoints, adj)  #TODO use pipelines
		pointdata: vtk.vtkPointData=adj_interpolated.GetPointData()
		vtkQ=pointdata.GetArray('Q')
		aQ=numpy_support.vtk_to_numpy(vtkQ)
		adjoints[i,:,:]=aQ
	return adjoints










# fws=list((Path("/home/sven/exa/adjoint/forward/output/")).glob("pointsource-*.vtk"))
# fws=os.listdir("/home/sven/exa/adjoint/forward/output/")
numfiles=89
#TODO better file reading
# forward_file="/home/sven/exa/adjoint/forward/output/pointsource-10.vtk"
# adjoint_file="/home/sven/exa/adjoint/adjoint/outputA/adaptive2a-20.vtk"

if False:
	for i in range(1,numfiles):
		forward_file=f"/home/sven/exa/adjoint/forward/output/pointsource-{i}.vtk"
		adjoint_file=f"/home/sven/exa/adjoint/adjoint/outputA/constsource-{numfiles-i}.vtk"
		impact, onlypoints=handle_two_files(forward_file, adjoint_file)
		i_normalized=impact/impact.max()
		if i==1:
			refine=np.zeros(impact.size)
		# refine=np.logical_or(refine, i_normalized>0.9)
		refine+=(i_normalized>0.9).astype(int)
		
		
forward_file=Template("/home/sven/exa/adjoint/forward/output/pointsource-$file.vtk")
adjoint_file=Template("/home/sven/exa/adjoint/adjoint/outputA/constsource-$file.vtk")

adjoints=adjoint_over_time(forward_file,adjoint_file,1,90)
for i in range(1,89):
	data=readUnstructuredGrid(forward_file.substitute({'file': i}))
	pointdata: vtk.vtkPointData=data.GetPointData()
	vtkQ=pointdata.GetArray('Q')
	fQ=numpy_support.vtk_to_numpy(vtkQ)

	onlypoints: vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
	onlypoints.SetPoints(data.GetPoints())
	for j in range(89-i):
		magnitude=np.abs(np.sum(fQ*adjoints[j,:,:], axis=1))  #scalar product for each point)
		impact=np.log10(magnitude+1e-9)
		i_normalized=impact/impact.max()
		if j==0:
			refine=np.zeros(impact.size)
		# refine+=(i_normalized>0.9).astype(int)
		refine=np.logical_or(refine, i_normalized>0.9)
		aa=1
	onlypoints.SetCells(data.GetCellTypesArray(), data.GetCells())
	write_numpy_array(refine.astype(int),onlypoints,f"outputE/version1-{i}.vtk")
	if i==3:break
	# plot_numpy_array(refine.astype(int), onlypoints)
		
		

		

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

