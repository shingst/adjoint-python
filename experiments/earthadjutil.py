import numpy as np
import matplotlib
import matplotlib.pyplot as plt
def countrefs(ref:np.ndarray):
	xx,yy=ref.shape	
	l1=np.count_nonzero(ref[1::3, 1::3])
	l2=np.count_nonzero(ref>=2)
	print(f"{100*l1/((xx-1)//3*(yy-1)//3):.1f}% refined once {100*l2/(xx*yy):.1f}% refined twice")

def plotrefsave(ref: np.ndarray, domain, offset,name,p1x,p1y,p2x,p2y):  # TODO color in whole level1 cells instead of subcells
	xx=np.linspace(offset[0], domain[0]+offset[0], ref.shape[0])
	yy=np.linspace(offset[1], domain[1]+offset[1], ref.shape[1])
	# plt.pcolor(xx,yy,ref.T,shading='auto',cmap="summer_r")
	plt.figure(figsize=(15, 12))
	# plt.pcolor(xx,yy,ref.T,shading='auto',cmap="Blues",vmax=3,vmin=-1)
	plt.pcolor(xx, yy, ref.T, shading='auto', cmap="YlGn", vmax=3, vmin=-1)
	plt.plot(p1x, p1y, 'ro', markersize=12)
	plt.plot(p2x, p2y, 'bo', markersize=12)
	plt.xlabel('x in km')
	plt.ylabel('y in km')
	plt.tight_layout()
	plt.savefig(name)