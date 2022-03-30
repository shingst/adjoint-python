import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt

# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondnoheart.probe")
# coarse=pr.values
# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondno-mo.probe")
# staticadj=pr.values
# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondfull.probe")
# fine=pr.values

pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/hel/coarse5.probe")
coarse=pr.values
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/hel/a5thirds5.probe")
staticadj=pr.values
staticadj=np.delete(staticadj,100,0)#is plotted twice
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/hel/full5-rank-9.probe")
fine=pr.values

Qnames=['σ_11','σ_22','σ_12','u','v']

nh=np.round(coarse[:,2:7]-fine[:,2:7],8)
adj=np.round(staticadj[:,2:7]-fine[:,2:7],8)

roundfine=np.round(fine[:,2:7],10)

print("l2 norm",np.linalg.norm(adj),"\t\t\t",np.linalg.norm(nh))

nhrel=np.abs(nh/roundfine)
adjrel=np.abs(adj/roundfine)

for i in range(5):
	plt.plot(coarse[:, 0], nhrel[:,i], label='no heart')
	plt.plot(coarse[:, 0], adjrel[:,i], label='adjoined refined')
	plt.semilogy()
	plt.legend()
	plt.title(f"5relative errors of {Qnames[i]}")
	plt.show()
	
	
nhrel[np.isnan(nhrel)]=0.0
adjrel[np.isnan(adjrel)]=0.0

nhrel[np.isinf(nhrel)]=0.0
adjrel[np.isinf(adjrel)]=0.0

nhsqt=0.0
adjsqt=0.0
for i in range(5):
	nhsq=nhrel[22:,i]@nhrel[22:,i]/nhrel.shape[0]
	adjsq=nhrel[22:,i]@adjrel[22:,i]/adjrel.shape[0]
	print(f"mean squared error of {Qnames[i]} refined:{adjsq} without the heart shape:{nhsq}")
	nhsqt+=nhsq
	adjsqt+=adjsq
	
print(f"total mean squared error refined:{adjsqt} without the heart shape:{nhsqt}")