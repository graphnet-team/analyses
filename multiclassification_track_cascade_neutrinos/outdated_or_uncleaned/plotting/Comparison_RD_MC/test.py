import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

a = np.array([0,1,1,1,1,2,3,2,3,2,3,4,4,4,5,5,5,5,5,5,5,5,5,5,5])
weight = np.ones_like(a)+np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1])
b = np.array([0,1,5,5,1,2,3,2,3,2,3,4,4,4,5,5,5,5,5,5,5,5,5,5,5])
bins = np.linspace(0,10,10)

fig, axs = plt.subplots(4,1,sharex=False,figsize=(8, 32))
counts,_,_,_ = axs[0].hist2d(a,b,bins)
counts_log,_,_,_ = axs[0].hist2d(a,b,bins,norm=colors.LogNorm())
counts_weights,_,_,_ = axs[0].hist2d(a,b,bins,norm=colors.LogNorm(),weights=weight)


print(counts)
#print(counts_log)
print(counts_weights)
print(counts-counts_weights)