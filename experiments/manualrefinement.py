import numpy as np
import shapely.affinity
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams.update({'font.size': 28})

# a=np.load("/home/sven/exa/adjoint/forward/output/bel/cutoff10thirds.npy")
xn=237
yn=120
xsize=30
ysize=15

# xn=159
# yn=237
# xsize=20
# ysize=30



b=np.zeros((yn,xn),dtype=int)

# inner=Polygon([(8,0),(11.5,0),(5,6),(4,6),(4,4.5)])
# outer=Polygon([(3.5,0),(12,0),(11,6),(3.5,6.5)])

#Manual 12 wel
# inner=Polygon([(4.5,3.3),(12.5,3.3),(12.5,6.45),(4.5,6.45)])
# outer=Polygon([(3.5,0.0),(13.5,0.0),(13.5,7.79),(3.5,7.79)])

#Manual 11 wel
line=Polygon([(5.0,5.0),(11.0,2.0),(5.0,5.0)])
inner=line.buffer(1.92)
# outer=line.buffer(2.5) #11C
outer=line.buffer(3.3) #11D

# line=Polygon([(5.0,5.0),(11.0,2.0),(5.0+30/7,0.0)])
# inner=line.buffer(1.57)
# # outer=line.buffer(2.5) #11C #TODO not done
# outer=line.buffer(3.2) #11D

inner=shapely.affinity.translate(inner,5.0) #The algorithm does not work with offsets
outer=shapely.affinity.translate(outer,5.0) #The algorithm does not work with offsets

#Manual 13 simple
# inner=Point(8,14).buffer(np.sqrt(600.0/(18.0*np.pi)))
# outer=Point(8,14).buffer(np.sqrt(600.0/(6.0*np.pi)))

#Manual 18 simple
# line=Polygon([(7.0,15.0),(13.0,18.0),(7.0,15.0)])
# inner=line.buffer(1.76)
# outer=line.buffer(3.90)




for iy, ix in np.ndindex(b.shape):
	x=(xsize*ix)/xn +1e-15
	y=ysize*iy/yn  +1e-15
	if outer.contains(Point(x,y)):
		if inner.contains(Point(x,y)):
			b[iy,ix]=2
		else:
			b[iy,ix]=1
	
def countrefs(ref:np.ndarray):
	xx,yy=ref.shape	
	l1=np.count_nonzero(ref[1::3,1::3])
	l2=np.count_nonzero(ref>=2)
	print(f"{100*l1/((xx-1)//3*(yy-1)//3):.1f}% refined once {100*l2/(xx*yy):.1f}% refined twice")
	
def plotref(ref:np.ndarray,domain,offset):# TODO color in whole level1 cells instead of subcells
	xx=np.linspace(offset[0],domain[0]+offset[0],ref.shape[0])
	yy=np.linspace(offset[1],domain[1]+offset[1],ref.shape[1])
	# plt.pcolor(xx,yy,ref.T,shading='auto',cmap="summer_r")
	plt.figure(figsize=(15,12))
	# plt.pcolor(xx,yy,ref.T,shading='auto',cmap="Blues",vmax=3,vmin=-1)
	plt.pcolor(xx,yy,ref.T,shading='auto',cmap="YlGn",vmax=3,vmin=-1)
	plt.plot(5,5,'ro',markersize=12)
	plt.plot(11, 2, 'bo',markersize=12)
	plt.xlabel('x in km')
	plt.ylabel('y in km')
	plt.tight_layout()
	# plt.savefig("test")
	plt.show()
	
	
	
# plotref(b.T,[xsize,ysize],[0.0,0.0]) #The algorithm does not work with offsets # -5,0 offset for wide/pwaves
plotref(b.T,[xsize,ysize],[-5.0,0.0]) #The algorithm does not work with offsets # -5,0 offset for wide/pwaves
countrefs(b)
# a=0
# bg=np.load("outputE/inner13a.npy")
# pcts=np.zeros(100)
# for i in range(100):
# 	pcts[i]=np.percentile(bg,i)
# 	
# plt.plot(np.arange(100),pcts)
# plt.show()
# 
# mx=np.percentile(bg,99.5)
# mn=0#mx*1e-10
# a=np.histogram(bg,bins=50,range=(mn,mx),density=True)
# 
# plt.plot(a[1][1:-1],a[0][1:])
# plt.xlabel("inner product")
# plt.ylabel("frequency")
# plt.semilogx()
# plt.show()
# 
# # plt.hist(bg.flatten(),bins=10)
# # plt.show()
np.save("/home/sven/exa/adjoint/forward/output/wel/manualmirrir11D.npy",b.T)


