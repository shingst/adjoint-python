import vtkmodules.all as vtk
import numpy as np
from vtkmodules.util import numpy_support
import os
from pathlib import Path
from string import Template
import sys
import hashlib

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
	fwhash = hashlib.md5(Path(forward_template.substitute({'file':'1'})).read_bytes()).hexdigest()
	adjhash = hashlib.md5(Path(adjoint_template.substitute({'file':'1'})).read_bytes()).hexdigest()
	cachfolder=os.path.expanduser('~/.cache/earthadj/')
	
	try:
		ret=np.load(cachfolder+fwhash+adjhash+".npy")
		print("loaded interpolation to forward grid from cache")
		return ret
	except OSError :
		print("could not load cached interpolation to forward grid")
	
	data=readUnstructuredGrid(forward_template.substitute({'file':'1'}))
	pts: vtk.vtkPoints=data.GetPoints()
	# ptsforward=numpy_support.vtk_to_numpy(pts.GetData())
	onlypoints: vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
	onlypoints.SetPoints(pts)
	num_pts=pts.GetNumberOfPoints()
	
	adjoints=np.zeros((finish-start,num_pts,5))
	percentcounter=0.1
	for i in range(finish-start):
		adj:vtk.vtkUnstructuredGrid=readUnstructuredGrid(adjoint_template.substitute({'file': i+start}))
		# ptsadjoint=numpy_support.vtk_to_numpy(adj.GetPoints().GetData())
		# if (ptsforward==ptsadjoint).all():
		# 	print("adjoint points and forward points are equal")
		# 	adj_interpolated=adj
		# else:
		adj_interpolated: vtk.vtkUnstructuredGrid=interpolate(onlypoints, adj)  #TODO use pipelines
		pointdata: vtk.vtkPointData=adj_interpolated.GetPointData()
		vtkQ=pointdata.GetArray('Q')
		aQ=numpy_support.vtk_to_numpy(vtkQ)
		adjoints[i,:,:]=aQ
		if i/(finish-start)>percentcounter:
			print(f"interpolation to forward grid  {100*percentcounter}% finished")
			percentcounter+=0.1

	if not os.path.exists(cachfolder):
		os.makedirs(cachfolder)
	ret=np.save(cachfolder+fwhash+adjhash+".npy",adjoints)
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

def points_to_cartetsian_int(ptsvtk:vtk.vtkUnstructuredGrid, refine:np.ndarray, xsize, ysize,domain,offset) -> np.ndarray:
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
			pts.SetPoint(j*xsize+i,[i*xst+offset[0],j*yst+offset[1],0])
			
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
	
	
def three_to_one_balancing(ptsvtk:vtk.vtkUnstructuredGrid,refine:np.ndarray,num_pts,ref_lvls,domain,offset,quantiles)->np.ndarray:
	# pts=(3**(levels)-2)*3**(ref_lvls-1)
	x,y=(num_pts*3**(ref_lvls-1)).astype(int)
	basic_grid=(points_to_cartetsian_int(ptsvtk, refine, x, y,domain,offset)).T #transposed due to numpy assuming a matrix instead of coordinates
	# np.save("second-interpolated.npy",basic_grid)
	# toint=np.floor(basic_grid*10-10+ref_lvls)
	# grid=np.maximum(toint, 0).astype(int).T 
	grid=(basic_grid>np.quantile(basic_grid, 1-quantiles[0]))*1
	for i in range(1, ref_lvls):
		grid+=(basic_grid>np.quantile(basic_grid,1- quantiles[i]))*1
	
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
				# nzero=np.nonzero(counter)[0]
				# if(nzero.shape[0]>0):
				# 	grid[i, j]=max(grid[i, j], nzero[-1])# nonzero returns a tuple with an array in it
							
						
						
	return grid
	
def countrefs(ref:np.ndarray):
	xx,yy=ref.shape	
	l1=np.count_nonzero(ref[1::3, 1::3])
	l2=np.count_nonzero(ref>=2)
	print(f"{100*l1/((xx-1)//3*(yy-1)//3):.1f}% refined once {100*l2/(xx*yy):.1f}% refined twice")
	
	
	
if __name__=='__main__':
	sys.path.append("/home/sven/uni/mt/ExaHyPE-Engine/Toolkit/exahype")
	# import as module
	from toolkit import Controller
	from tools import tools

	sys.argv=[sys.argv[0], '/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward.exahype'] #! improve
	configfw=Controller().spec
	tools.tools=[] # needed or if will be filled twice and the parser crashes
	sys.argv=[sys.argv[0], '/home/sven/uni/mt/ExaHyPE-Engine/adjoint/fwrefined.exahype']  #! improve
	configref=Controller().spec
	
	#TODO handle paths
	
	# forward_file=Template("/home/sven/exa/adjoint/forward/output/pointsource-$file.vtk")
	# adjoint_file=Template("/home/sven/exa/adjoint/adjoint/outputA/constsource-$file.vtk")
	# forward_file=Template("/home/sven/exa/adjoint/forward/output/secondwide-$file.vtk")
	# adjoint_file=Template("/home/sven/exa/adjoint/adjoint/outputA/secondvelo-$file.vtk")
	# forward_file=Template("/home/sven/exa/adjoint/forward/sismo/wp1c-$file.vtk")
	# adjoint_file=Template("/home/sven/exa/adjoint/adjoint/outputA/sismoWP1-$file.vtk")
	# forward_file=Template("/home/sven/exa/adjoint/forward/output/helsinkimo-$file.vtk")
	# adjoint_file=Template("/home/sven/exa/adjoint/adjoint/outputA/helsinkimo-$file.vtk")
	forward_file=Template("/home/sven/exa/adjoint/forward/output/bel/coarse-$file.vtk")
	adjoint_file=Template("/home/sven/exa/adjoint/adjoint/outputA/bel10vel-$file.vtk")
	
	end_time=configfw['computational_domain']['end_time']
	domain=np.array(configref['computational_domain']['width'])
	offset=np.array(configref['computational_domain']['offset'])
	assert np.allclose(domain,np.array(configfw['computational_domain']['width']))#domain between refined and coarse grid should be the same
	output_interval=configfw['solvers'][0]['plotters'][0]['repeat'] 
	max_cell_size=configref['solvers'][0]['maximum_mesh_size']
	max_depth=configref['solvers'][0]['maximum_mesh_depth']
	output_file=configref['solvers'][0]['adg']
	
	num_files=int((end_time+1e-9)//output_interval) #todo toint 
	max_level=np.max(np.ceil(-np.log(max_cell_size/domain)/np.log(3)) )#includes level 0 so maybe np.floor would be better ! <-comment is false
	max_pts=(3**(max_level)-2)
	cell_size=np.max(domain/max_pts)
	
	level_points=np.ceil(np.round(domain/cell_size,7))
	assert (np.max(level_points)==max_pts)
	
	adjoints=adjoint_over_time(forward_file,adjoint_file,1,num_files)#TODO start at 0?
	print("interpolated all adjoints to the forward grid")
	percentcounter=0.0
	for i in range(1,num_files-1):
		data=readUnstructuredGrid(forward_file.substitute({'file': i}))
		pointdata: vtk.vtkPointData=data.GetPointData()
		vtkQ=pointdata.GetArray('Q')
		fQ=numpy_support.vtk_to_numpy(vtkQ)
	
		onlypoints: vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
		onlypoints.SetPoints(data.GetPoints())
		for j in range(num_files-1-i):
			magnitude=np.abs(np.sum(fQ[:,3:]*adjoints[j,:,3:], axis=1))  #scalar product for each point)
			# magnitude=np.abs(np.sum(fQ*adjoints[j, :, :], axis=1))
			if magnitude.max()>0:
				impact=magnitude/magnitude.max()
			# impact=np.log10(magnitude+1e-9)+1.5#rescales the offset
			# impact=magnitude
			if j==0:
				dbgtmp=np.zeros(impact.size)
			if j==0 and i==1:
				refine=np.zeros(impact.size)
			# if impact.max()>0:
			# 	i_normalized=impact/impact.max()
			# 	refine_steps2(refine, i_normalized)
			
			refine_steps2(refine,impact)
			refine_steps2(dbgtmp,impact)
			# refine_steps2(refine,magnitude)
		if (i+1)/(num_files-2)>percentcounter:
			print(f"scalar product of {100*percentcounter}% forward grids finished")
			percentcounter+=0.1
		onlypoints.SetCells(data.GetCellTypesArray(), data.GetCells())
		plot_numpy_array(dbgtmp, onlypoints)

	onlypoints.SetCells(data.GetCellTypesArray(), data.GetCells())
		# write_numpy_array(refine,onlypoints,f"outputE/version2-{i}.vtk")
	print("created refinement grid")
	plot_numpy_array(refine, onlypoints)
	quantiles=[1/3,1/9,1/27,1/81,1/243,1/729,1/2187]
	ref=three_to_one_balancing(onlypoints, refine,level_points,max_depth,domain,offset,quantiles)
	print("finished 3 to 1 balancing")
	countrefs(ref)
	np.save(output_file, ref) #TODO maybe use smaller int
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

