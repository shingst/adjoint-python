import numpy as np
def countrefs(ref:np.ndarray):
	xx,yy=ref.shape	
	l1=np.count_nonzero(ref[1::3, 1::3])
	l2=np.count_nonzero(ref>=2)
	print(f"{100*l1/((xx-1)//3*(yy-1)//3):.1f}% refined once {100*l2/(xx*yy):.1f}% refined twice")