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

dtype = np.dtype([
  ("timesteps",  np.uint32),
  ("savepoints", np.uint32),
  ("Fwidth",     np.uint32),
  ("Gwidth",     np.uint32),
  ("Hwidth",     np.uint32)
])


if len(sys.argv)!=2:
  print("Syntax: {0} <Save File>".format(sys.argv[1]))
  sys.exit(-1)



#fin = gzip.GzipFile("/z/saved")
fin = open(sys.argv[1], "rb")
fin.seek(0, os.SEEK_SET)

rawdata = fin.read(5*4)
timesteps, savepoints, Fwidth, Gwidth, Hwidth  = np.frombuffer(rawdata, dtype=dtype)[0]

f = GetSavePoint(fin, savepoints, Fwidth)
G = GetSavePoint(fin, savepoints, Gwidth)
H = GetSavePoint(fin, savepoints, Hwidth)

sum_H = GetVector(fin, timesteps)
sum_G = GetVector(fin, timesteps)
sum_F = GetVector(fin, timesteps)
avg_E = GetVector(fin, timesteps)
avg_N = GetVector(fin, timesteps)
avg_S = GetVector(fin, timesteps)

raise Exception("Interactive time")

plt.matshow(np.log(f.transpose())/np.log(10), aspect='auto'); plt.show()
plt.matshow(np.log(G.transpose())/np.log(10), aspect='auto'); plt.show()
plt.matshow(np.log(H.transpose())/np.log(10), aspect='auto'); plt.show()

temp = G.copy()
temp[temp<1e-3]= 0
plt.matshow(np.log(temp.transpose())/np.log(10), aspect='auto'); plt.show()

plt.matshow(np.log(H.transpose())/np.log(10), aspect='auto'); plt.show()

plt.plot(G[19000,:]); plt.show()


fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
ax[0].plot(sum_H, label="sum_H")
ax[0].set_ylim([-0.1,1.1])
ax[0].legend()
ax[1].plot(sum_G, label="sum_G")
ax[1].legend()
ax[2].plot(sum_F, label="sum_F")
ax[2].legend()
fig.suptitle("Normalization Check (should be 1)")
plt.show()

fig, ax = plt.subplots(1,3, sharex=True)
ax[0].plot(avg_E, label="avg_E")
ax[0].legend()
ax[1].plot(avg_N, label="avg_N")
ax[1].legend()
ax[2].plot(avg_S, label="avg_S")
ax[2].legend()
fig.suptitle("Average of state variables")
plt.show()


fig, ax = plt.subplots(1,4, sharex=True)
ax[0].plot(avg_E, label="avg_E")
ax[0].legend()
ax[1].plot(avg_N, label="avg_N")
ax[1].legend()
ax[2].plot(avg_S, label="avg_S")
ax[2].legend()
ax[3].plot(avg_N/avg_S, label="avg_N/avg_S")
ax[3].legend()
fig.suptitle("Average of state variables")
plt.show()
