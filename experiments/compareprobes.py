import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import matplotlib
import sys

# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/second.probe")
# # pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/coarse.probe")
# coarse=pr.values
# pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/staticrefined.probe")
# staticref=pr.values
# pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/fine.probe")
# fine=pr.values
# file1cs="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/inner12a12.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/manual12a12.probe"
# file1cs= "/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner13a13.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner13B13.probe"
file1cs= "/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner18a18.probe"
file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/manual18a18.probe"

# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/sismo/rec1c.probe")
pr=pd.read_csv(file1cs)
coarse=pr.values
pr=pd.read_csv(file2adj)
staticref=pr.values
# staticref=np.delete(staticref,100,0)#is plotted twice
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/full10.probe")
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/full12.probe")
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/fine18.probe")
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
# int1=sci.interp1d(coarse[:,0],coarse[:,2])
# cc=int1()
# 
# def resampleprobe(probe,interval=0.02,start=0.0,end=3.0):
# 	time_points=np.arange(0, 3, 0.02)
# 	ret=np.zeros_like(probe)
# 	ret[:,0]=time_points
# 	ret[:,1]=time_points
# 	for i in range(5):
# 		int1=sci.interp1d(probe[:,0],probe[:,i+2])
# 		ret[:,i+2]=int1(time_points)
# 	return ret
# 
# cc=resampleprobe(coarse)


# coarse=fine
col=2
if len(sys.argv)==2:
	col=int(sys.argv[1])

# # plt.plot(coarse[:, 0], (coarse[:, col]-staticref[:150,col])/(staticref[:150,col]+1e-9), label='σ_11 cmp')
# matplotlib.use('Gtk3Cairo')
plt.plot(staticref[:, 0], staticref[:, col], label='σ_11 manual')
plt.plot(fine[:, 0], fine[:, col], label='σ_11 fine')
plt.plot(coarse[:, 0], coarse[:, col], label='σ_11 Adjrefined')
# plt.plot(coarse[:, 0], coarse[:, 3], label='σ_22')
# plt.plot(coarse[:, 0], coarse[:, 4], label='σ_12')
# # plt.plot(coarse[:, 0], coarse[:, 5], label='u')
# # plt.plot(coarse[:, 0], coarse[:, 6], label='v')
# # plt.semilogy()
plt.legend()
plt.show()
