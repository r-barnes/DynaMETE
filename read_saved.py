#!/usr/bin/env python3
import gzip
import matplotlib as mpl        #Used for controlling color
import matplotlib.colors        #Used for controlling color as well
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def GetSavePoint(f, savepoint_num, width):
  rawdata = f.read(8*savepoint_num*width)
  savepoint = np.frombuffer(rawdata, dtype=np.float64)
  savepoint = savepoint.reshape(savepoint_num, width)
  return savepoint

def GetVector(f, width):
  rawdata = f.read(8*width)
  return np.frombuffer(rawdata, dtype=np.float64)

class SavedFile:
  def __init__(self, filename):
    dtype = np.dtype([
      ("timesteps",  np.uint32),
      ("savepoints", np.uint32),
      ("Fwidth",     np.uint32),
      ("Gwidth",     np.uint32),
      ("Hwidth",     np.uint32)
    ])

    fin = open(filename, "rb")
    fin.seek(0, os.SEEK_SET)

    rawdata = fin.read(5*4)
    self.timesteps, self.savepoints, self.Fwidth, self.Gwidth, self.Hwidth  = np.frombuffer(rawdata, dtype=dtype)[0]

    self.f = GetSavePoint(fin, self.savepoints, self.Fwidth)
    self.G = GetSavePoint(fin, self.savepoints, self.Gwidth)
    self.H = GetSavePoint(fin, self.savepoints, self.Hwidth)

    self.sum_H = GetVector(fin, self.timesteps)
    self.sum_G = GetVector(fin, self.timesteps)
    self.sum_F = GetVector(fin, self.timesteps)
    self.avg_E = GetVector(fin, self.timesteps)
    self.avg_N = GetVector(fin, self.timesteps)
    self.avg_S = GetVector(fin, self.timesteps)

if len(sys.argv)!=2:
  print("Syntax: {0} <Save File>".format(sys.argv[0]))
  sys.exit(-1)

sf = SavedFile(sys.argv[1])

def MatShow(times, mat, cutoff=None, log=True, show_zeros=True):
  temp   = mat.copy()
  temp   = temp.transpose() #Fits better since time is longer than bins
  temp   = np.flipud(temp)
  fig    = plt.figure()
  ax     = fig.add_subplot(111)
  extent = [times[0],times[-1],0,temp.shape[0]]
  clabel = "Probability"
  if show_zeros:
    natural_zeros = temp==0
    natural_zeros = np.ma.masked_where(natural_zeros==False, natural_zeros)
  if cutoff:
    cutoffm = np.logical_and(temp!=0,temp<cutoff)
    cutoffm = np.ma.masked_where(cutoffm==False, cutoffm)
  if log:
    temp   = np.log(temp)/np.log(10)
    clabel = "log(Probability)"
  matplotted = ax.matshow(temp, aspect='auto', extent=extent)
  if show_zeros:
    natural_zeros_cmap = mpl.colors.ListedColormap(['black','black'])
    ax.matshow(natural_zeros, aspect='auto', extent=extent, cmap=natural_zeros_cmap, vmin=0, vmax=1)
  if cutoff:
    cutoff_cmap = mpl.colors.ListedColormap(['black','red'])
    ax.matshow(cutoffm, aspect='auto', extent=extent, cmap=cutoff_cmap, vmin=0, vmax=1)    
  ax.set_xticks(times)
  plt.locator_params(axis='x', nbins=10)
  plt.xticks(rotation=-30)
  plt.xlabel("Time")
  plt.ylabel("Bin Number")
  plt.colorbar(matplotted, label="log(Probability)")
  plt.show()
  return natural_zeros

raise Exception("Interactive time")

MatShow(list(range(sf.f.shape[0])), sf.f, log=False, cutoff=1e-3)
MatShow(list(range(sf.G.shape[0])), sf.G, log=True, cutoff=1e-3)
MatShow(list(range(sf.H.shape[0])), sf.H, log=True, cutoff=1e-3)


fig = plt.figure()
ax = fig.add_subplot(111)
temp = sf.G.copy()
temp_masked = np.ma.masked_where(temp==0, temp, copy=True)
temp_masked[temp_masked==0] = np.nan
#temp[temp<1e-3]= 0
ax.matshow(temp.transpose(), aspect='auto');
ax.matshow(temp_masked.transpose(), aspect='auto', cmap='Greys');
plt.show()

plt.matshow(np.log(temp.transpose())/np.log(10), aspect='auto'); plt.show()

plt.matshow(np.log(sf.H.transpose())/np.log(10), aspect='auto'); plt.show()

plt.plot(sf.G[19000,:]); plt.show()


fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
ax[0].plot(sf.sum_H, label="sum_H")
ax[0].set_ylim([-0.1,1.1])
ax[0].legend()
ax[1].plot(sf.sum_G, label="sum_G")
ax[1].legend()
ax[2].plot(sf.sum_F, label="sum_F")
ax[2].legend()
fig.suptitle("Normalization Check (should be 1)")
plt.show()

fig, ax = plt.subplots(1,3, sharex=True)
ax[0].plot(sf.avg_E, label="avg_E")
ax[0].legend()
ax[1].plot(sf.avg_N, label="avg_N")
ax[1].legend()
ax[2].plot(sf.avg_S, label="avg_S")
ax[2].legend()
fig.suptitle("Average of state variables")
plt.show()


fig, ax = plt.subplots(1,4, sharex=True)
ax[0].plot(sf.avg_E, label="avg_E")
ax[0].legend()
ax[1].plot(sf.avg_N, label="avg_N")
ax[1].legend()
ax[2].plot(sf.avg_S, label="avg_S")
ax[2].legend()
ax[3].plot(sf.avg_N/sf.avg_S, label="avg_N/avg_S")
ax[3].legend()
fig.suptitle("Average of state variables")
plt.show()
















# sfe = SavedFile("/z/saved_euler")
# sfr = SavedFile("/z/saved_runge")


# f = np.abs(sfe.f - sfr.f)
# G = np.abs(sfe.G - sfr.G)
# H = np.abs(sfe.H - sfr.H)


# plt.matshow(np.log(f.transpose())/np.log(10), aspect='auto'); plt.show()
# plt.matshow(np.log(G.transpose())/np.log(10), aspect='auto'); plt.show()
# plt.matshow(np.log(H.transpose())/np.log(10), aspect='auto'); plt.show()


# temp = H.copy()
# temp[temp<1e-6]= 0
# plt.matshow(np.log(temp.transpose())/np.log(10), aspect='auto'); plt.colorbar(); plt.show()

# plt.matshow(np.log(sf.H.transpose())/np.log(10), aspect='auto'); plt.show()
