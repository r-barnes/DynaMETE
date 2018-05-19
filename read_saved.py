#!/usr/bin/env python3
import gzip
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

raise Exception("Interactive time")

plt.matshow(np.log(sf.f.transpose())/np.log(10), aspect='auto'); plt.colorbar(); plt.show()
plt.matshow(np.log(sf.G.transpose())/np.log(10), aspect='auto'); plt.colorbar(); plt.show()
plt.matshow(np.log(sf.H.transpose())/np.log(10), aspect='auto'); plt.colorbar(); plt.show()

temp = sf.G.copy()
temp[temp<1e-3]= 0
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
