import pandas as pd
import numpy as  np


filename="/home/sven/uni/mt/ExaHyPE-Engine/adjoint/forward/output/bel/coarse8.probe"
pr=pd.read_csv(filename)
arr=pr.values

nums=arr.shape[0]
sps=50
outf=open("/tmp/abc.txt","w")

outf.write(f"TIMESERIES XX_TEST__BHZ_R, {nums} samples, {sps} sps, 2020-01-01T00:00:00.000000, TSPAIR, FLOAT, M/S\n")
for i in range(nums):
	if arr[i,5]<0:
		outf.write(f"2020-01-01T00:00:{arr[i,0]:09.6f} {arr[i,5]:9.6f}\n")
	else:
		outf.write(f"2020-01-01T00:00:{arr[i, 0]:09.6f} {arr[i, 5]:9.6f}\n")
outf.close()
