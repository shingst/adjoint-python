import vtkmodules.all as vtk
import numpy as np
from vtkmodules.util import numpy_support
import os
from pathlib import Path
from string import Template
import sys

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
	# interactor.Start()


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


def points_to_cartetsian(ptsvtk:vtk.vtkPoints, refine:np.ndarray, xsize, ysize) -> np.ndarray:
	"""
	nearest neighbor interpolation
	:param ptsvtk: #! gets changed
	:param refine: 
	:param xsize: 
	:param ysize: 
	:return: 
	"""
	ret=np.zeros((xsize,ysize)).astype(int)
	x0,xmax,y0,ymax,_,_=ptsvtk.GetBounds()
	# xstride=(xmax-x0)/xsize
	# ystride=(ymax-y0)/ysize
	# xstart=x0+xstride/2
	# ystart=y0+ystride/2
	pts=numpy_support.vtk_to_numpy(ptsvtk.GetData()).copy()
	pts[:,0]-=x0 #sets lower bound to zero
	pts[:,0]*=xsize/(xmax-x0)# sets bounds to [0,xsize]
	pts[:, 1]-=y0  #sets lower bound to zero 
	pts[:, 1]*=ysize/(ymax-y0)  # sets bounds to [0,xsize]
	idx=np.floor(pts).astype(int)
	idx[pts[:,0]==xsize,0]-=1
	idx[pts[:,1]==ysize,1]-=1
	for i in range(idx.shape[0]):
		ret[idx[i,0],idx[i,1]]=np.maximum(ret[idx[i,0],idx[i,1]],refine[i])
	return  ret

def refine_steps(refine:np.ndarray,inorm:np.ndarray):
	np.maximum(np.floor(inorm*10)/10,refine,refine)
	
	
def three_to_one_balancing(ptsvtk:vtk.vtkPoints,refine:np.ndarray,levels,ref_lvls)->np.ndarray:
	pts=(3**(levels)-2)*3**(ref_lvls-1)
	x,y=pts.astype(int)
	toint=refine*10-10+ref_lvls
	refineclasses=np.maximum(toint, 0).astype(int)
	grid=points_to_cartetsian(ptsvtk, refineclasses, x, y)
	
	for lvl in range(1,ref_lvls): #inverse level
		for i in range((3**lvl-1)//2,x,3**lvl):
			for j in range((3**lvl-1)//2,y,3**lvl):
				stride=3**(lvl-1)
				for m in range(max((3**lvl-2)//2,i-2*stride),min(x,i+2*stride),stride):
					for n in range(max((3**lvl-2)//2,j-2*stride),min(y,j+2*stride),stride):
						if grid[m,n]>ref_lvls-lvl:
							grid[i,j]=max(grid[i,j],ref_lvls-lvl)
				m_start=-2
				m_end=3
				if i-2*stride<0:
					m_start=-1
				if i+2*stride>=x:
					m_end=2
				n_start=-2
				n_end=3
				if j-2*stride<0:
					n_start=-1
				if j+2*stride>=y:
					n_end=2
				counter=0
				for m in range(m_start,m_end):
					for n in range(n_start,n_end):
						cell_val=grid[i+m*stride,j+n*stride]
						if cell_val>ref_lvls-lvl:
							grid[i,j]=max(grid[i,j],ref_lvls-lvl)
							break
						if cell_val==ref_lvls-lvl and -2<m<2 and -2<n<2: #only for values in the same cell
							counter+=1
				if counter >=0: # TODO adapt if interpolation does useful stuff
					grid[i, j]=max(grid[i, j], ref_lvls-lvl)
							
						
						
	return grid
	
	
	
	
	
if __name__=='__main__':
	sys.path.append("/home/sven/uni/mt/ExaHyPE-Engine/Toolkit/exahype")
	# import as module
	from toolkit import Controller

	sys.argv=[sys.argv[0], '/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward.exahype'] #! improve
	config=Controller().spec
	
	#TODO handle paths
	forward_file=Template("/home/sven/exa/adjoint/forward/output/pointsource-$file.vtk")
	adjoint_file=Template("/home/sven/exa/adjoint/adjoint/outputA/constsource-$file.vtk")
	
	#TODO get data from the correct file (first run or prod run)
	end_time=config['computational_domain']['end_time']
	domain=np.array(config['computational_domain']['width'])
	output_interval=config['solvers'][0]['plotters'][0]['repeat'] #! read from coarse forward 
	max_cell_size=config['solvers'][0]['maximum_mesh_size']
	max_depth=config['solvers'][0]['maximum_mesh_depth']
	
	num_files=(end_time+1e-9)//output_interval
	levels=np.ceil(-np.log(max_cell_size/domain)/np.log(3)) #includes level 0 so maybe np.floor would be better ! <-comment is false
	
	
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
			if j==0 and i==1:
				refine=np.zeros(impact.size)
			# refine+=(i_normalized>0.9).astype(int)
			# refine=np.logical_or(refine, i_normalized>0.9)
			refine_steps(refine,i_normalized)
			aa=1
	onlypoints.SetCells(data.GetCellTypesArray(), data.GetCells())
		# write_numpy_array(refine,onlypoints,f"outputE/version2-{i}.vtk")
	plot_numpy_array(refine, onlypoints)
	ref=three_to_one_balancing(data.GetPoints(), refine,levels,max_depth)
	np.save("outputE/balancing4.npy", ref) #TODO maybe use smaller int
	a=0

	
	# np.savetxt("test2.txt", ref, '%f', header=str(ref.shape))
		

		

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
# plot_numpy_array(refine.astype(int),onlypoints)#! warning alters onlypoints


a=0

