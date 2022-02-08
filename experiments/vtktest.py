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
# data2=readUnstructuredGrid(filename2)

cells=data.GetCells()
ncells=cells.GetNumberOfCells()

n_points=data.GetNumberOfPoints()

pts:vtk.vtkPoints=data.GetPoints()

# pointsset:vtk.vtkUnstructuredGrid=vtk.vtkUnstructuredGrid()
# pointsset.SetPoints(pts)
x=np.ones(3)*1.2
y=np.ones(3)*1.2
# for i in range(data.GetNumberOfPoints()):
# 	data.GetPoint(i,x)
# 	data2.GetPoint(i,y)
# 	assert (x==y).all()

# pts:vtk.vtkPoints=data.GetPoints()
pointdata=data.GetPointData()
vtkQ=pointdata.GetArray('Q')
Q=numpy_support.vtk_to_numpy(vtkQ)

impact=np.log10(Q+1e-9)
i_normalized=impact/impact.max()




def points_to_cartetsian(ptsvtk:vtk.vtkPoints, refine:np.ndarray, xsize, ysize)->np.ndarray:
	ret=np.zeros((xsize,ysize))
	x0,xmax,y0,ymax,_,_=ptsvtk.GetBounds()
	# xstride=(xmax-x0)/xsize
	# ystride=(ymax-y0)/ysize
	# xstart=x0+xstride/2
	# ystart=y0+ystride/2
	pts=numpy_support.vtk_to_numpy(ptsvtk.GetData())
	pts[:,0]-=x0 #sets lower bound to zero
	pts[:,0]*=xsize/(xmax-x0)# sets bounds to [0,xsize]
	idx=np.floor(pts).astype(int)
	idx[pts==xsize]-=1
	for i in range(idx.shape[0]):
		ret[idx[i,0],idx[i,1]]=np.maximum(ret[idx[i,0],idx[i,1]],refine[i,0])
	
	return ret
	
	
# points_to_cartetsian(pts,Q,10,10)
# sgrid:vtk.vtkStructuredPoints=vtk.vtkStructuredPoints()
# sgrid.SetDimensions(10,10,1)
# 
# sgrid.GetPointData()
# 
# gaussian_kernel=vtk.vtkGaussianKernel()
# gaussian_kernel.SetSharpness(4)
# gaussian_kernel.SetRadius(1)
# 
# interp=vtk.vtkPointInterpolator()
# interp.SetInputData(sgrid)
# interp.SetSourceData(data)
# # interp.SetKernel(gaussian_kernel)
# interp.Update()
# res:vtk.vtkStructuredPoints= interp.GetOutput() 

pts=vtk.vtkPoints()
pts.SetNumberOfPoints(5*3)






# for i in range(5):
# 	mx=np.argmax(Q[:,i])
# 	mn=np.argmin(Q[:,i])
# 	data.GetPoint(mn, x)
# 	print(f"max Q[:,{i}] location {x},{mx}")
# 	data.GetPoint(mx, x)
# 	print(f"min Q[:,{i}] location {x},{mn}")
print("tmp")



