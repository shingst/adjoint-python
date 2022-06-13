import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import matplotlib
import sys
import matplotlib
#Creates plots

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# matplotlib.rcParams.update({'font.size': 34})
matplotlib.rcParams.update({'font.size': 28})


# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/second.probe")
# # pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/alphaadj.probe")
# alphaadj=pr.values
# pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/staticrefined.probe")
# staticref=pr.values
# pr=pd.read_csv("/home/sven/uni/mt/data/firstexamplerefinementcompare/fine.probe")
# fine=pr.values
# file1cs="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/inner12a12.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/manual12a12.probe"
file1cs="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/wel/inner11D11.probe"
file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/wel/manual11D11.probe"
# file1cs= "/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner13a13.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner13B13.probe"
# file1cs= "/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner18a18.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/manual18a18.probe"

# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/sismo/rec1c.probe")
pr=pd.read_csv(file1cs)
alphaadj=pr.values
pr=pd.read_csv(file2adj)
staticref=pr.values
# staticref=np.delete(staticref,100,0)#is plotted twice
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/full10.probe")
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/wel/full11.probe")
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/full12.probe")
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/fine18.probe")
fine=pr.values


# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/sigma12h12.probe")
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/wel/inner11stressD11.probe")
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner18stress18.probe")
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/fine18.probe")
betaadj=pr.values


# length=pr.shape[0]
# 
# alphaadjnorm=np.linalg.norm((alphaadj-fine)/(fine +1e-14),axis=1)
# statnorm=np.linalg.norm((staticref-fine)/(fine +1e-14),axis=1)
# alphaadjnorm[alphaadjnorm<1e-15]=1e-15
# statnorm[statnorm<1e-15]=1e-15



# alphaadj=fine
col=5
if len(sys.argv)==2:
	col=int(sys.argv[1])
	
ylab=["σ_11 in MPa","σ_11 in MPa","σ_11 in MPa","u in m/s","v in m/s"]

# # plt.plot(alphaadj[:, 0], (alphaadj[:, col]-staticref[:150,col])/(staticref[:150,col]+1e-9), label='σ_11 cmp')
# matplotlib.use('Gtk3Cairo')
# plt.figure(figsize=(15/0.7, 9))
plt.figure(figsize=(15, 12))

plt.plot(fine[:, 0], fine[:, col], label='fine grid', linewidth=2,color="black")
plt.plot(staticref[:, 0], staticref[:, col], label='manual refinement', linewidth=2)
# plt.plot(alphaadj[:, 0], alphaadj[:, col], label='adjoint refinement', linewidth=2)
plt.plot(alphaadj[:, 0], alphaadj[:, col], label='α-adjoint refinement', linewidth=2)
plt.plot(betaadj[:, 0], betaadj[:, col], label='β-adjoint refinement', linewidth=2)

#"β"  α

# plt.plot(alphaadj[:, 0], alphaadj[:, 3], label='σ_22')
# plt.plot(alphaadj[:, 0], alphaadj[:, 4], label='σ_12')
# # plt.plot(alphaadj[:, 0], alphaadj[:, 5], label='u')
# # plt.plot(alphaadj[:, 0], alphaadj[:, 6], label='v')



plt.xlim([0.8,3])
# plt.xlim([1.2,1.6])
# plt.ylim([-0.6,4.2])
plt.xlabel('t in s')
# plt.ylabel('Moment-rate in Nm')
plt.ylabel(ylab[col-2])
# plt.semilogy()
plt.legend()

plt.tight_layout()
# plt.savefig("/home/sven/thesis/images/plothel11Du")
plt.show()
