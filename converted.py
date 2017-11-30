#!/usr/bin/env python3

######################
#PACKAGE CONFIGURATION
######################

import numpy as np

#precision = Number of decimal places
#suppress=True => No scientific notation
#linewidth = Number of characters in line before wrapping
np.set_printoptions(precision=4, suppress=True, linewidth=160)



######################
#CONSTANTS
######################
b0    = 0.0025
d0    = 0.001
lam0  = 0
m     = 0.002
Emax  = 300000
EEmax = 1 #EEmax = 1:E/Emax
w0    = 0.01
Smeta = 60
w1    = 0.0003
d1    = (b0-d0)/Emax
meta  = 100 #1000/(E/N) of meta

MAX_TIMESTEP    = 10 #50001

MAX_INDIVIDUALS = 10 #251
MAX_SPECIES     = 10 # 65
MAX_METABOLIC   = 10 #324



######################
#GLOBALS
######################

f = np.zeros(shape=(MAX_TIMESTEP, MAX_SPECIES))     #Total number of species
G = np.zeros(shape=(MAX_TIMESTEP, MAX_INDIVIDUALS)) #Total number of individuals
H = np.zeros(shape=(MAX_TIMESTEP, MAX_METABOLIC))   #Total metabolic rate



######################
#INITIAL CONDITIONS
######################

#NOTE: The equations below assume that Nvalue[0] = 0 and Evalue[0] = 0
#TODO: Try to eliminate Nint and Eint
Nint   = 10
Eint   = 1000
Nvalue = np.array([Nint*x for x in range(MAX_INDIVIDUALS)])
Evalue = np.array([Eint*x for x in range(MAX_METABOLIC  )])

f[0,5] = 1 #100% probability of having 10 species at t=0
G[0,5] = 1 #100% probability of having 10 individuals at t=0
H[0,5] = 1 #100% probability of having 10 units of metabolic rate (Watts) at t=0



######################
#TIME LOOP
######################

for t in range(1,MAX_TIMESTEP):
  ###########################
  #SET UP INTERMEDIATE VALUES
  ###########################

  #NOTE: This blows up if Nvalue[0]=0, so we only look at Nvalue[1:]
  logN        = np.log(Nvalue[1:])
  expected_N1 = np.sum(Nvalue[1:]**(1/3)               *G[t-1,1:]) #avg_N1 = <N^(1/3)>
  expected_N2 = np.sum(1/logN                          *G[t-1,1:]) #avg_N2 = <1/ln(N)>
  expected_N3 = np.sum(1/logN**(1/3)                   *G[t-1,1:]) #avg_N3 = <1/ln(N)^(1/3)>
  expected_N4 = np.sum(logN**(1/3)                     *G[t-1,1:]) #avg_N4 = <ln(N)^(1/3)>
  expected_N5 = np.sum(Nvalue[1:]**(1/3) / logN**(2/3) *G[t-1,1:]) #avg_N5 = <N^(1/3)/ln(N)^(2/3)

  #NOTE: This blows up if Evalue[0]=0, so we only look at Evalue[1:]
  expected_E1 = np.sum(Evalue[1:]**(-1/3) * H[t-1,1:])   #avg_E1 = <E^(-1/3)>
  expected_E2 = np.sum(Evalue[1:]**( 2/3) * H[t-1,1:])   #avg_E2 = <E^(2/3)>

  ###################
  #Calculate H matrix
  ###################

  #Offsets from each column to its left and right neighbours
  ci = list(range(2,MAX_METABOLIC-1)) #Center index
  li = np.roll(ci,shift= 1)           #Left index
  ri = np.roll(ci,shift=-1)           #Right index

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately following.
  H[t,ci] = (H[t-1,ci] - m*H[t-1,ci]/meta + m*H[t-1,li]/meta + 
      (d0*Evalue[ri]**(2/3) + d1*Evalue[ri]**(5/3))*expected_N5*H[t-1,ri]/Eint
    - (d0*Evalue[ci]**(2/3) + d1*Evalue[ci]**(5/3))*expected_N5*H[t-1,ci]/Eint
    -  w0*Evalue[ci]**(2/3)*expected_N5*H[t-1,ci]/Eint 
    +  w1*Evalue[ci]                   *H[t-1,ci]/Eint
    +  w0*Evalue[li]**(2/3)*expected_N5*H[t-1,li]/Eint 
    -  w1*Evalue[li]                   *H[t-1,li]/Eint
  )

  #######################
  #H matrix special cases
  #######################

  #Outer columns are special cases: First column
  H[t,0] = (H[t-1,0] - m*H[t-1,0]/meta 
    + (d0*Evalue[1]**(2/3) + d1*Evalue[1]**(5/3))*expected_N5*H[t-1,1]/Eint
  )
  
  #Special case: Second column
  H[t,1] = (H[t-1,1] - m*H[t-1,1]/meta + m*H[t-1,0]/meta 
    + (d0*Evalue[2]**(2/3) + d1*Evalue[2]**(5/3))*expected_N5*H[t-1,2]/Eint
    - (d0*Evalue[1]**(2/3) + d1*Evalue[1]**(5/3))*expected_N5*H[t-1,1]/Eint
    -  w0*Evalue[1]**(2/3)*expected_N5*H[t-1,1]/Eint 
    +  w1*Evalue[1]                   *H[t-1,1]/Eint
  )

  #Special case: last column
  #NOTE: This is unconverted code using a right-most column that is all zeros. Not sure why that column exists, so we hard-code 0 in the actual code. Ask John about this. TODO
  #H[t+1,323] = H[t,323] + m*H[t,322]/meta + (d0*H[1,324]^(2/3) + d1*H[1,324]^(5/3)) * avg_N5*H[t,324]/1000 - (d0*H[1,323]^(2/3) + d1*H[1,323]^(5/3)) * avg_N5*H[t,323]/1000 + w0*H[1,322]^(2/3)*avg_N5*H[t,322]/1000 - w1*H[1,322]*H[t,322]/1000
  
  #Special case: last column
  H[t,-1] = (H[t-1,-1] + m*H[t-1,-2]/meta 
    - (d0*Evalue[-1]**(2/3) + d1*Evalue[-1]**(5/3)) * expected_N5*H[t-1,-1]/Eint 
    +  w0*Evalue[-2]**(2/3)*expected_N5*H[t-1,-2]/Eint 
    -  w1*Evalue[-2]                   *H[t-1,-2]/Eint
  )

