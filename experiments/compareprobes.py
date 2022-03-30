import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt


# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/second.probe")
# # pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/coarse.probe")
# coarse=pr.values
# pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/staticrefined.probe")
# staticref=pr.values
# pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/fine.probe")
# fine=pr.values

# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/sismo/rec1c.probe")
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/hel/coarse8.probe")
coarse=pr.values
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/hel/a8thirds8.probe")
staticref=pr.values
staticref=np.delete(staticref,100,0)#is plotted twice
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/hel/full8-rank-9.probe")
fine=pr.values


# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondnoheart.probe")
# coarse=pr.values
# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondno-mo.probe")
# staticref=pr.values
# pr=pd.read_csv("/home/sven/x/steamdisk/thesisdata/secondcomp/secondfull.probe")
# fine=pr.values


# length=pr.shape[0]
# 
# coarsenorm=np.linalg.norm((coarse-fine)/(fine +1e-14),axis=1)
# statnorm=np.linalg.norm((staticref-fine)/(fine +1e-14),axis=1)
# coarsenorm[coarsenorm<1e-15]=1e-15
# statnorm[statnorm<1e-15]=1e-15


# plt.plot(coarse[:, 0], coarsenorm, label='σ_11 noheart')
# plt.plot(coarse[:, 0], statnorm, label='σ_11 static')



col=3
# plt.plot(coarse[:, 0], (coarse[:, col]-staticref[:150,col])/(staticref[:150,col]+1e-9), label='σ_11 noheart')
plt.plot(staticref[:, 0], staticref[:, col], label='σ_11 static')
plt.plot(fine[:, 0], fine[:, col], label='σ_11 fine')
plt.plot(coarse[:, 0], coarse[:, col], label='σ_11 coarse')
# plt.plot(coarse[:, 0], coarse[:, 3], label='σ_22')
# plt.plot(coarse[:, 0], coarse[:, 4], label='σ_12')
# plt.plot(coarse[:, 0], coarse[:, 5], label='u')
# plt.plot(coarse[:, 0], coarse[:, 6], label='v')
# plt.semilogy()
plt.legend()
plt.show()
