import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt

# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondnoheart.probe")
# coarse=pr.values
# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondno-mo.probe")
# staticadj=pr.values
# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondfull.probe")
# fine=pr.values

pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/f10thirds10.probe")
coarse=pr.values
if coarse.shape[0]==151:
	coarse=np.delete(coarse, 100, 0)  #is plotted twice

pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/f10quarter10.probe")
staticadj=pr.values
staticadj=np.delete(staticadj,100,0)#is plotted twice
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/full10-rank-26.probe")
fine=pr.values

def RMS(approx,fgrid,col):
	a=approx[:, col]-fgrid[:,col]
	return np.sqrt((a@a)/(fgrid[:, col]@fgrid[:, col]))


Qnames=['σ_11','σ_22','σ_12','u','v']

def timeerror(probe):
	terr=probe[1:,0]-probe[:-1,0]
	avg=np.average(np.abs(terr))




for i in range(5):
	print(f"{Qnames[i]}:RMS of first :{RMS(coarse,fine,i+2)}  RMS second:{RMS(staticadj,fine,i+2)}")


nh=np.round(coarse[:,2:7]-fine[:,2:7],8)
adj=np.round(staticadj[:,2:7]-fine[:,2:7],8)

roundfine=np.round(fine[:,2:7],10)

print("l2 norm",np.linalg.norm(adj),"\t\t\t",np.linalg.norm(nh))

nhrel=np.abs(nh/roundfine)
adjrel=np.abs(adj/roundfine)

for i in range(5):
	plt.plot(coarse[:, 0], nhrel[:,i], label='coarse')
	plt.plot(coarse[:, 0], adjrel[:,i], label='adjoined refined')
	plt.semilogy()
	plt.legend()
	plt.title(f"3relative errors of {Qnames[i]}")
	plt.show()
	
	
nhrel[np.isnan(nhrel)]=0.0
adjrel[np.isnan(adjrel)]=0.0

nhrel[np.isinf(nhrel)]=0.0
adjrel[np.isinf(adjrel)]=0.0

nhsqt=0.0
adjsqt=0.0
for i in range(5):
	nhsq=np.sqrt(nhrel[22:,i]@nhrel[22:,i]/nhrel.shape[0])
	adjsq=np.sqrt(nhrel[22:,i]@adjrel[22:,i]/adjrel.shape[0])
	print(f"mean squared error of {Qnames[i]} refined:{adjsq} without the heart shape:{nhsq}")
	nhsqt+=nhsq
	adjsqt+=adjsq
	
print(f"total mean squared error refined:{adjsqt} without the heart shape:{nhsqt}")