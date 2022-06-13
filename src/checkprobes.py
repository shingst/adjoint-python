import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import matplotlib
#RMS computation + many different data points + plots

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams.update({'font.size': 28})




# file1cs="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/inner12a12.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/manual12a12.probe"
file1cs="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/wel/inner11stressD11.probe"
file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/wel/manual11D11.probe"
# file1cs="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/amr8t8.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/manual10thirds10.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/f8thirds8.probe"
# file1cs="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/tunenoamr10t10.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/tunenoamr10v10.probe"
# file1cs= "/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner18a18.probe"
# file2adj="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/manual18a18.probe"


pr=pd.read_csv(file1cs)
coarse=pr.values
if coarse.shape[0]==151:
	coarse=np.delete(coarse, 100, 0)  #is plotted twice

pr=pd.read_csv(file2adj)
staticadj=pr.values
if staticadj.shape[0]==151:
	staticadj=np.delete(staticadj, 100, 0)  #is plotted twice
	
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/inner18stress18.probe")
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/wel/inner11stressD11.probe")
beta=pr.values
if beta.shape[0]==151:
	beta=np.delete(beta, 100, 0)  #is plotted twice

# staticadj=np.delete(staticadj,100,0)#is plotted twice
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/full10.probe")
pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/wel/full11.probe")
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/pwaves/full12.probe")
# pr=pd.read_csv("/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/probes/simple/fine18.probe")
fine=pr.values
if fine.shape[0]==151:
	fine=np.delete(fine, 100, 0)  #is plotted twice


def RMS(approx,fgrid,col):
	a=approx[:, col]-fgrid[:,col]
	return np.sqrt((a@a)/(fgrid[:, col]@fgrid[:, col]))

def RMSall(approx, fgrid):
	a=(approx[:, 2:7]-fgrid[:, 2:7])
	return np.sqrt(np.sum(a*a)/np.sum(fgrid[:, 2:7]*(fgrid[:, 2:7])))


Qnames=['σ_11','σ_22','σ_12','u','v']

def timeerror(probe):
	terr=probe[1:,0]-probe[:-1,0]
	avg=np.average(np.abs(terr))

def resampleprobe(probe,interval=0.02,start=0.0,end=2.98):
	time_points=np.arange(start, end+interval, interval)
	ret=np.zeros_like(probe)
	ret[:,0]=time_points
	ret[:,1]=time_points
	for i in range(5):
		int1=sci.interp1d(probe[:,0],probe[:,i+2])
		ret[:,i+2]=int1(time_points)
	return ret

coarse=resampleprobe(coarse)
fine=resampleprobe(fine)
staticadj=resampleprobe(staticadj)
beta=resampleprobe(beta)

rmsfirst=np.zeros(6)
rmssecond=np.zeros(6)
for i in range(5):
	rmsfirst[i]=RMS(coarse,fine,i+2)
	rmssecond[i]=RMS(staticadj,fine,i+2)
	print(f"{Qnames[i]}:RMS of first :{rmsfirst[i]}  RMS second:{rmssecond[i]}")

rmsfirst[5]=RMSall(coarse,fine)
rmssecond[5]=RMSall(staticadj,fine)

print(f"overall RMS of first :{rmsfirst[5]}  RMS second:{rmssecond[5]}")
print(f"& {rmsfirst[0]:.3} & {rmsfirst[1]:.3} & {rmsfirst[2]:.3} & {rmsfirst[3]:.3} & {rmsfirst[4]:.3} & {rmsfirst[5]:.3} ")
print(f"& {rmssecond[0]:.3} & {rmssecond[1]:.3} & {rmssecond[2]:.3} & {rmssecond[3]:.3} & {rmssecond[4]:.3} & {rmssecond[5]:.3} ")

# cse=np.round(coarse[:,2:7]-fine[:,2:7],8)
# adj=np.round(staticadj[:,2:7]-fine[:,2:7],8)
cse=np.round(coarse[:,2:7]-fine[:,2:7],8)
adj=np.round(staticadj[:,2:7]-fine[:,2:7],8)


roundfine=np.round(fine[:,2:7],10)

print("l2 norm first",np.linalg.norm(cse),"\tsecond\t",np.linalg.norm(adj))

coarserel=np.abs(cse/(roundfine+1e-3))
adjrel=np.abs(adj/(roundfine+1e-3))

# for i in range(5):
# 	plt.figure(figsize=(15, 12))
# 	plt.plot(coarse[:, 0], coarserel[:,i], label='first')
# 	plt.plot(coarse[:, 0], adjrel[:,i], label='second')
# 	plt.semilogy()
# 	plt.legend()
# 	plt.title(f"relative errors of {Qnames[i]}")
# 	plt.tight_layout()
# 	plt.show()
	
for i in range(5): #TODO (3,4)
	plt.figure(figsize=(15, 12))
	
	plt.plot(coarse[:, 0], np.abs(staticadj[:,i+2]-fine[:,i+2]), label='manual refinement')#second
	plt.plot(coarse[:, 0], np.abs(coarse[:, i+2]-fine[:, i+2]), label='α-adjoint refinement')#first
	plt.plot(coarse[:, 0], np.abs(beta[:, i+2]-fine[:, i+2]), label='β-adjoint refinement')#first
	# plt.ylim(1e-5,20)
	# plt.semilogy()
	plt.legend()
	plt.xlabel('t in s')
	plt.xlim([0.8,3])
	if i!=3:#as u is saved 
		plt.title(f"absoulute errors of {Qnames[i]}")
	else:
		plt.ylabel("absolute error of in u in m/s")
	plt.tight_layout()
	# plt.savefig("/home/sven/thesis/images/errorhel11Du")
	plt.show()

#! uncomment evrything below	
plt.figure(figsize=(15, 12))
plt.plot(coarse[:, 0], np.sum(np.abs(coarse[:,2:7]-fine[:,2:7]),axis=1), label='first')
plt.plot(coarse[:, 0], np.sum(np.abs(staticadj[:,2:7]-fine[:,2:7]),axis=1), label='second')
# plt.ylim(1e-5,20)
# plt.semilogy()
plt.legend()
plt.title("absoulute l1 errors")
plt.tight_layout()
plt.show()

coarserel[np.isnan(coarserel)]=0.0
adjrel[np.isnan(adjrel)]=0.0

coarserel[np.isinf(coarserel)]=0.0
adjrel[np.isinf(adjrel)]=0.0

coarsesqt=0.0
adjsqt=0.0
for i in range(5):
	coarsesq=np.sqrt(coarserel[22:,i]@coarserel[22:,i]/coarserel.shape[0])
	adjsq=np.sqrt(coarserel[22:,i]@adjrel[22:,i]/adjrel.shape[0])
	print(f"mean squared error of {Qnames[i]} refined:{adjsq} coarse:{coarsesq}")
	coarsesqt+=coarsesq
	adjsqt+=adjsq

print(f"total mean squared error refined:{adjsqt} coarse:{coarsesqt}")