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
b0    = 0.0025       #Birth rate per capita
d0    = 0.001        #Death rate per capita
lam0  = 0            #Speciation rate
m     = 0.002        #Migration rate
Emax  = 300000       #Soft maximum for energy (TODO)
EEmax = 1            #EEmax = 1:E/Emax
                     #Ontogenic growth dMetabolicRate/dt=w0*e^(2/3)-w1*e (TODO)
w0    = 0.01         #Ontogenic growth (growth of individual over its lifespan): See above
w1    = 0.0003       #Ontogenic growth (growth of individual over its lifespan): See above
Smeta = 60           #Species richness of the meta community
d1    = (b0-d0)/Emax #Density dependent contribution to death rate
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
Sint   = 1
Nvalue = np.array([Nint*x for x in range(MAX_INDIVIDUALS)])
Evalue = np.array([Eint*x for x in range(MAX_METABOLIC  )])
Svalue = np.array([Sint*x for x in range(MAX_SPECIES    )])

f[0,5] = 1 #100% probability of having 10 species at t=0
G[0,5] = 1 #100% probability of having 10 individuals at t=0
H[0,5] = 1 #100% probability of having 10 units of metabolic rate (Watts) at t=0

sum_H = np.array([0 for x in range(MAX_TIMESTEP)]) #Use this later to check normalization
sum_G = np.array([0 for x in range(MAX_TIMESTEP)]) #Use this later to check normalization
sum_F = np.array([0 for x in range(MAX_TIMESTEP)]) #Use this later to check normalization

E_avg = np.array([0 for x in range(MAX_TIMESTEP)]) #Use this later to calculate average
N_avg = np.array([0 for x in range(MAX_TIMESTEP)]) #Use this later to calculate average
S_avg = np.array([0 for x in range(MAX_TIMESTEP)]) #Use this later to calculate average


######################
#TIME LOOP
######################

for t in range(1,MAX_TIMESTEP):
  ###########################
  #SET UP INTERMEDIATE VALUES
  ###########################

  #NOTE: This blows up if Nvalue[0]=0, so we only look at Nvalue[1:]
  logN        = np.log(Nvalue[1:])
  expected_N1 = np.sum(Nvalue[1:]**(1/3)               *G[t-1,1:]) #expected_N1 = <N^(1/3)>
  expected_N2 = np.sum(1/logN                          *G[t-1,1:]) #expected_N2 = <1/ln(N)>
  expected_N3 = np.sum(1/logN**(1/3)                   *G[t-1,1:]) #expected_N3 = <1/ln(N)^(1/3)>
  expected_N4 = np.sum(logN**(1/3)                     *G[t-1,1:]) #expected_N4 = <ln(N)^(1/3)>
  expected_N5 = np.sum(Nvalue[1:]**(1/3) / logN**(2/3) *G[t-1,1:]) #expected_N5 = <N^(1/3)/ln(N)^(2/3)

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
  H[t,ci] = (H[t-1,ci] - m*H[t-1,ci]/meta + m*H[t-1,li]/meta
    +  w0*Evalue[li]**(2/3)                        *expected_N5*H[t-1,li]/Eint
    -  w1*Evalue[li]                                           *H[t-1,li]/Eint
    - (d0*Evalue[ci]**(2/3) + d1*Evalue[ci]**(5/3))*expected_N5*H[t-1,ci]/Eint
    -  w0*Evalue[ci]**(2/3)                        *expected_N5*H[t-1,ci]/Eint
    +  w1*Evalue[ci]                                           *H[t-1,ci]/Eint
    + (d0*Evalue[ri]**(2/3) + d1*Evalue[ri]**(5/3))*expected_N5*H[t-1,ri]/Eint
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
    - (d0*Evalue[1]**(2/3) + d1*Evalue[1]**(5/3))*expected_N5*H[t-1,1]/Eint
    -  w0*Evalue[1]**(2/3)                       *expected_N5*H[t-1,1]/Eint
    +  w1*Evalue[1]                                          *H[t-1,1]/Eint
    + (d0*Evalue[2]**(2/3) + d1*Evalue[2]**(5/3))*expected_N5*H[t-1,2]/Eint
  )

  #Special case: last column
  H[t,-1] = (H[t-1,-1] + m*H[t-1,-2]/meta
    +  w0*Evalue[-2]**(2/3)                         * expected_N5*H[t-1,-2]/Eint
    -  w1*Evalue[-2]                                             *H[t-1,-2]/Eint
    - (d0*Evalue[-1]**(2/3) + d1*Evalue[-1]**(5/3)) * expected_N5*H[t-1,-1]/Eint
  )

  ###################
  #Calculate G matrix
  ###################

  #Offsets from each column to its left and right neighbours
  ci = list(range(2,MAX_INDIVIDUALS-1)) #Center index
  li = np.roll(ci,shift= 1)           #Left index
  ri = np.roll(ci,shift=-1)           #Right index

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately following.
  G[t,ci] = (G[t-1,ci] - m*(G[t-1,ci] - G[t-1,li])/Nint
  + G[t-1,li]*(     b0*expected_E1                 ) * Nvalue[li]**(4/3)*np.log(Nvalue[li])**(1/3)/Nint
  - G[t-1,ci]*((b0+d0)*expected_E1 + d1*expected_E2) * Nvalue[ci]**(4/3)*np.log(Nvalue[ci])**(1/3)/Nint
  + G[t-1,ri]*(     d0*expected_E1 + d1*expected_E2) * Nvalue[ri]**(4/3)*np.log(Nvalue[ri])**(1/3)/Nint
  )

  #######################
  #G matrix special cases
  #######################

  #Outer columns are special cases: First column
  G[t,0] = (G[t-1,0] - m*G[t-1,0]/Nint
  + G[t-1, 1]*(     d0*expected_E1 + d1*expected_E2) * Nvalue[ 1]**(4/3)*np.log(Nvalue[ 1])**(1/3)/Nint
  )

  #Special case: Second column
  G[t,1] = (G[t-1,1]- m*(G[t-1,1]-G[t-1,0])/Nint
  - G[t-1, 1]*((b0+d0)*expected_E1 + d1*expected_E2) * Nvalue[ 1]**(4/3)*np.log(Nvalue[ 1])**(1/3)/Nint
  + G[t-1, 2]*(     d0*expected_E1 + d1*expected_E2) * Nvalue[ 2]**(4/3)*np.log(Nvalue[ 2])**(1/3)/Nint
  )

  #Special case: last column
  G[t,-1] = (G[t-1,-1] + m*G[t-1,-2]/Nint
  + G[t-1,-2]*(b0*expected_E1)                  *np.log(Nvalue[-2])**(1/3)*Nvalue[-2]**(4/3)/Nint
  - G[t-1,-1]*(d0*expected_E1 + d1*expected_E2) *np.log(Nvalue[-1])**(1/3)*Nvalue[-1]**(4/3)/Nint
  )

  ###################
  #Calculate F matrix
  ###################

  #Offsets from each column to its left and right neighbours
  ci = list(range(2,MAX_SPECIES-1))  #Center index
  li = np.roll(ci,shift= 1)          #Left index
  ri = np.roll(ci,shift=-1)          #Right index

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately followingself.
  f[t,ci] = (lam0*Svalue[li]*f[t-1,li] + f[t-1,li]*m*(1-Svalue[li]/Smeta)
    + f[t-1,ci] - lam0*f[1,ci]*f[t-1,ci]- f[t-1,ci]*m*(1-Svalue[ci]/Smeta)
    - f[t-1,ci]*Svalue[ci]**(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
    + f[t-1,ri]*Svalue[ri]**(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
  )

  #######################
  #F matrix special cases
  #######################

  #Outer columns are special cases: First column
  f[t,0] = (f[t-1,0] - f[t-1,0]*m*(1-Svalue[0 ]/Smeta)
    + f[t-1, 1]*Svalue[ 1]**(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
  )

  #Special case: last column
  f[t,-1] = (f[t-1,-1] + f[t-1,-2]*m*(1-Svalue[-2]/Smeta)
    - f[t-1,-1]*Svalue[-1]**(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
    + lam0*Svalue[-2]*f[t-1,-2]
  )

  ####################
  #Check Normalization
  ####################

  #see wether the probabilities sum up to 1
  
  prob_E = np.sum(H[t, 1:])
  sum_H[t] = prob_E

  prob_N = np.sum(G[t, 1:])
  sum_G[t] = prob_N

  prob_S = np.sum(f[t, 1:])
  sum_F[t] = prob_S

  #Calculate <E>, <N>, <S>
  avg_E = 0
  avg_N = 0
  avg_S = 0

  #Offsets from each column to its left and right neighbours
  ci = list(range(2,MAX_METABOLIC-1)) #Center index
  li = np.roll(ci,shift= 1)           #Left index
  ri = np.roll(ci,shift=-1)           #Right index

  avg_E = avg_E + Evalue[ci] * H[t, ci]

  #Offsets from each column to its left and right neighbours
  ci = list(range(2,MAX_INDIVIDUALS-1)) #Center index
  li = np.roll(ci,shift= 1)           #Left index
  ri = np.roll(ci,shift=-1)           #Right index

  avg_N = avg_N + Nvalue[ci] * G[t, ci]

  #Offsets from each column to its left and right neighbours
  ci = list(range(2,MAX_SPECIES-1)) #Center index
  li = np.roll(ci,shift= 1)           #Left index
  ri = np.roll(ci,shift=-1)           #Right index

  avg_S = avg_S + Svalue[ci] * f[t, ci]


  #Code from R
  for (t in 2:nrow(H)){
    prob_E = 0
    for (E in 1:(ncol(H))){
      prob_E = prob_E + H[t,E]
    }
    sum_H[t-1] = prob_E
  }

  sum_f <- vector(,length = 50001)
  for (t in 2:nrow(f)){
    prob_S = 0
    for (S in 1:(ncol(f))){
      prob_S = prob_S + f[t,S]
    }
    sum_f[t-1] = prob_S
  }

  sum_G <- vector(,length = 50001)
  for (t in 2:nrow(G)){
    prob_N = 0
    for (N in 1:(ncol(G))){
      prob_N = prob_N + G[t,N]
    }
    sum_G[t-1] = prob_N
  }

  #Calculate <S>, <N> and <E>
  S_avg <- vector(,length = 50000)
  N_avg <- vector(,length = 50000)
  E_avg <- vector(,length = 50000)

  for (t in 2:50001){
    avg_S = 0
    avg_N = 0
    avg_E = 0
    for (N in 1:ncol(G)){
      avg_N = avg_N + G[1,N]*G[t,N]
      N_avg[t-1] = avg_N
    }
    
    for (S in 1:ncol(f)){
      avg_S = avg_S + f[1,S]*G[t,S]
      S_avg[t-1] = avg_S
    }
    
    for(E in 1:ncol(H)){
      avg_E = avg_E + H[1,E]*H[t,E]
      E_avg[t-1] = avg_E
    }
  } 








