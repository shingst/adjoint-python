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
	gaussian_kernel.SetRadius(0.5)#TODO cell size

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

def points_to_cartetsian_int(ptsvtk:vtk.vtkUnstructuredGrid, refine:np.ndarray, xsize, ysize,domain) -> np.ndarray:
	# sgrid: vtk.vtkStructuredPoints=vtk.vtkStructuredPoints()
	# sgrid.SetDimensions(xsize, ysize, 1)
	# sgrid.SetScalarType()
	
	sgrid:vtk.vtkStructuredGrid=vtk.vtkStructuredGrid()
	sgrid.SetDimensions(xsize,ysize,1)
	pts=vtk.vtkPoints()
	pts.SetNumberOfPoints(xsize*ysize)
	xst,yst=domain/[xsize,ysize]
	for j in range(ysize):
		for i in range(xsize):
			pts.SetPoint(j*xsize+i,[i*xst,j*yst,0])
			
	sgrid.SetPoints(pts)
	

	refdata: vtk.vtkArray=numpy_support.numpy_to_vtk(refine)
	refdata.SetName('ref')
	ptsvtk.GetPointData().SetActiveScalars('ref')
	ptsvtk.GetPointData().SetScalars(refdata)
	
	voronoi=vtk.vtkVoronoiKernel() #maybe shepar kernel or something else Linear is bad

	interp=vtk.vtkPointInterpolator()
	interp.SetInputData(sgrid)#TODO change to double/int
	interp.SetSourceData(ptsvtk)
	interp.SetKernel(voronoi)
	interp.Update()
	res: vtk.vtkStructuredPoints=interp.GetOutput()
	scalars=res.GetPointData().GetScalars()
	ret=numpy_support.vtk_to_numpy(scalars)
	return ret.reshape((ysize,xsize)) #TODO
	

def refine_steps(refine:np.ndarray,inorm:np.ndarray):
	np.maximum(np.floor(inorm*10)/10,refine,refine)
	
def refine_steps2(refine:np.ndarray,inorm:np.ndarray):
	np.maximum(inorm,refine,refine)
	
	
def three_to_one_balancing(ptsvtk:vtk.vtkUnstructuredGrid,refine:np.ndarray,num_pts,ref_lvls,domain)->np.ndarray:
	# pts=(3**(levels)-2)*3**(ref_lvls-1)
	x,y=(num_pts*3**(ref_lvls-1)).astype(int)
	basic_grid=(points_to_cartetsian_int(ptsvtk, refine, x, y,domain)).T #transposed due to numpy assuming a matrix instead of coordinates
	# np.save("second-interpolated.npy",basic_grid)
	# toint=np.floor(basic_grid*10-10+ref_lvls)
	# grid=np.maximum(toint, 0).astype(int).T 
	grid=(basic_grid>np.quantile(basic_grid, 2/3))*1
	for i in range(2, ref_lvls+1):
		grid+=(basic_grid>np.quantile(basic_grid,1- 1/(3**i)))*1
	
	for lvl in range(1,ref_lvls): #inverse level
		for i in range((3**lvl-1)//2,x,3**lvl):
			for j in range((3**lvl-1)//2,y,3**lvl):
				stride=3**(lvl-1)
				# for m in range(max((3**lvl-2)//2,i-2*stride),min(x,i+2*stride),stride):
				# 	for n in range(max((3**lvl-2)//2,j-2*stride),min(y,j+2*stride),stride):
				# 		if grid[m,n]>ref_lvls-lvl:
				# 			grid[i,j]=max(grid[i,j],ref_lvls-lvl)
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
				counter=np.zeros(ref_lvls-lvl+1)
				for m in range(m_start,m_end):
					for n in range(n_start,n_end):
						cell_val=grid[i+m*stride,j+n*stride]
						if cell_val>ref_lvls-lvl:
							grid[i,j]=max(grid[i,j],ref_lvls-lvl)
							break
						if -2<m<2 and -2<n<2: #only for values in the same cell
							counter[cell_val]+=1
				# TODO adapt if interpolation does useful stuff
				nzero=np.nonzero(counter)[0]
				if(nzero.shape[0]>0):
					grid[i, j]=max(grid[i, j], nzero[-1])# nonzero returns a tuple with an array in it
							
						
						
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
	
	num_files=int((end_time+1e-9)//output_interval) #todo toint 
	max_level=np.max(np.ceil(-np.log(max_cell_size/domain)/np.log(3)) )#includes level 0 so maybe np.floor would be better ! <-comment is false
	max_pts=(3**(max_level)-2)
	cell_size=np.max(domain/max_pts)
	
	level_points=np.ceil(domain/cell_size)
	assert (np.max(level_points)==max_pts)
	
	adjoints=adjoint_over_time(forward_file,adjoint_file,1,num_files)#TODO start at 0?
	print("interpolated all adjoints to the forward grid")
	for i in range(1,num_files-1):
		data=readUnstructuredGrid(forward_file.substitute({'file': i}))
		pointdata: vtk.vtkPointData=data.GetPointData()
		vtkQ=pointdata.GetArray('Q')
		fQ=numpy_support.vtk_to_numpy(vtkQ)
	
		onlypoints: vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
		onlypoints.SetPoints(data.GetPoints())
		for j in range(num_files-1-i):
			magnitude=np.abs(np.sum(fQ*adjoints[j,:,:], axis=1))  #scalar product for each point)
			impact=np.log10(magnitude+1e-9)
			if j==0 and i==1:
				refine=np.zeros(impact.size)
			if impact.max()>0:
				i_normalized=impact/impact.max()
				refine_steps2(refine, i_normalized)
				# refine+=(i_normalized>0.9).astype(int)
				# refine=np.logical_or(refine, i_normalized>0.9)
	onlypoints.SetCells(data.GetCellTypesArray(), data.GetCells())
		# write_numpy_array(refine,onlypoints,f"outputE/version2-{i}.vtk")
	print("created refinement grid")
	plot_numpy_array(refine, onlypoints)
	ref=three_to_one_balancing(onlypoints, refine,level_points,max_depth,domain)
	print("finished 3 to 1 balancing")
	np.save("outputE/secondwide.npy", ref) #TODO maybe use smaller int
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

