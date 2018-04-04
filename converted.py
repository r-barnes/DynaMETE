#!/usr/bin/env python3

######################
#PACKAGE CONFIGURATION
######################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#from line_profiler import LineProfiler

#precision = Number of decimal places
#suppress=True => No scientific notation
#linewidth = Number of characters in line before wrapping
np.set_printoptions(precision=4, suppress=True, linewidth=160)

np.seterr(all='warn')
#np.seterr(all='raise')


######################
#CONSTANTS
######################
b0              = 0.0025       #Birth rate per capita
d0              = 0.001        #Death rate per capita
lam0            = 0            #Speciation rate
m               = 0.001        #Migration rate
Emax            = 300000       #Soft maximum for energy (TODO)
w0              = 0.01         #Ontogenic growth (growth of individual over its lifespan): See above
w1              = 0.0003       #Ontogenic growth (growth of individual over its lifespan): See above
Smeta           = 60           #Species richness of the meta community
d1              = (b0-d0)/Emax #Density dependent contribution to death rate
meta            = 100          #100/(E/N) of meta
Nint            = 10           #Number of individuals per individual bin
Eint            = 100          #Number of energy units per energy bin
Sint            = 1            #Number of species per species bin
MAX_TIMESTEP    = 50001        #Number of timesteps to take
MAX_INDIVIDUALS = 251          #Number of bins for individuals
MAX_SPECIES     = 65           #Number of bins for species
MAX_METABOLIC   = 3240         #Number of bins for energy



#########################
#PRINT CONFIGURATION INFO
#########################
print("m b0              = {0:f}".format(b0))
print("m d0              = {0:f}".format(d0))
print("m lam0            = {0:f}".format(lam0))
print("m m               = {0:f}".format(m))
print("m Emax            = {0:f}".format(Emax))
print("m w0              = {0:f}".format(w0))
print("m w1              = {0:f}".format(w1))
print("m Smeta           = {0:f}".format(Smeta))
print("m d1              = {0:f}".format(d1))
print("m meta            = {0:f}".format(meta))
print("m Nint            = {0:f}".format(Nint))
print("m Eint            = {0:f}".format(Eint))
print("m Sint            = {0:f}".format(Sint))
print("m MAX_TIMESTEP    = {0:f}".format(MAX_TIMESTEP))
print("m MAX_INDIVIDUALS = {0:f}".format(MAX_INDIVIDUALS))
print("m MAX_SPECIES     = {0:f}".format(MAX_SPECIES))
print("m MAX_METABOLIC   = {0:f}".format(MAX_METABOLIC))



######################
#GLOBALS
######################

f = np.zeros(shape=(MAX_TIMESTEP, MAX_SPECIES))     #Total number of species
G = np.zeros(shape=(MAX_TIMESTEP, MAX_INDIVIDUALS)) #Total number of individuals
H = np.zeros(shape=(MAX_TIMESTEP, MAX_METABOLIC))   #Total metabolic rate



######################
#INITIAL CONDITIONS
######################

#NOTE: The equations below assume that Gvalue[0] = 0 and Hvalue[0] = 0
Gvalue = np.array([Nint*x for x in range(MAX_INDIVIDUALS)])
Hvalue = np.array([Eint*x for x in range(MAX_METABOLIC  )])
Fvalue = np.array([Sint*x for x in range(MAX_SPECIES    )])

f[0,5]   = 1 #100% probability of having 5 species at t=0
G[0,5]   = 1 #100% probability of having 50 individuals at t=0
H[0,100] = 1 #100% probability of having 10000 units of metabolic rate (Watts) at t=0

sum_H = np.zeros(MAX_TIMESTEP) #Use this later to check normalization
sum_G = np.zeros(MAX_TIMESTEP) #Use this later to check normalization
sum_F = np.zeros(MAX_TIMESTEP) #Use this later to check normalization

avg_E = np.zeros(MAX_TIMESTEP) #Use this later to calculate average
avg_N = np.zeros(MAX_TIMESTEP) #Use this later to calculate average
avg_S = np.zeros(MAX_TIMESTEP) #Use this later to calculate average



######################
#Initial normalization

#see wether the probabilities sum up to 1
sum_H[0] = np.sum(H[0, 1:]) #the sum of the probabilities of Energy at time 0
sum_G[0] = np.sum(G[0, 1:]) #the sum of the probabilities of Individuals at time 0
sum_F[0] = np.sum(f[0, 1:]) #the sum of the probabilities of Species at time 0

#Calculate <E>, <N>, <S>
avg_E[0] = np.sum(Hvalue[2:] * H[0, 2:])
avg_N[0] = np.sum(Gvalue[2:] * G[0, 2:])
avg_S[0] = np.sum(Fvalue[2:] * f[0, 2:])



######################
#TIME LOOP
######################

for t in range(1,MAX_TIMESTEP):
  if t%100==0:
    print('p t = {0}'.format(t))


  ###########################
  #SET UP INTERMEDIATE VALUES
  ###########################

  #NOTE: This blows up if Gvalue[0]=0, so we only look at Gvalue[1:]
  n_s = avg_N[t-1] / avg_S[t-1] #constant avg_N divided by avg_S
  logN = np.log(n_s * np.log(n_s * np.log(n_s * np.log(n_s * np.log(n_s))))) #formula to calculate logN
  #logN        = np.log(Gvalue[1:])
  expected_N2 = np.sum(1/logN                          *G[t-1,1:]) #expected_N2 = <1/ln(N)>
  expected_N3 = np.sum(1/logN**(1/3)                   *G[t-1,1:]) #expected_N3 = <1/ln(N)^(1/3)>
  expected_N4 = np.sum(logN**(1/3)                     *G[t-1,1:]) #expected_N4 = <ln(N)^(1/3)>
  expected_N5 = np.sum(Gvalue[1:]**(1/3) / logN**(2/3) *G[t-1,1:]) #expected_N5 = <N^(1/3)/ln(N)^(2/3)

  #NOTE: This blows up if Hvalue[0]=0, so we only look at Hvalue[1:]
  expected_E1 = np.sum(Hvalue[1:]**(-1/3) * H[t-1,1:])   #avg_E1 = <E^(-1/3)>
  expected_E2 = np.sum(Hvalue[1:]**( 2/3) * H[t-1,1:])   #avg_E2 = <E^(2/3)>

  ###################
  #Calculate H matrix
  ###################

  l = lambda x: x[0:-2]
  c = lambda x: x[1:-1]
  r = lambda x: x[2:  ]

  pl = lambda x: l(x[t-1,:])
  pc = lambda x: c(x[t-1,:])
  pr = lambda x: r(x[t-1,:])

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately following.
  H[t,1:-1] = (
          pc(H)
    -   m*pc(H)/meta
    +   m*pl(H)/meta
    +  w0*l(Hvalue)**(2/3)                        *expected_N5*pl(H)/Eint
    -  w1*l(Hvalue)                                           *pl(H)/Eint
    - (d0*c(Hvalue)**(2/3) + d1*c(Hvalue)**(5/3)) *expected_N5*pc(H)/Eint
    -  w0*c(Hvalue)**(2/3)                        *expected_N5*pc(H)/Eint
    +  w1*c(Hvalue)                                           *pc(H)/Eint
    + (d0*r(Hvalue)**(2/3) + d1*r(Hvalue)**(5/3)) *expected_N5*pr(H)/Eint
  )

  #######################
  #H matrix special cases
  #######################

  #Outer columns are special cases: First column
  H[t,0] = (
        H[t-1,0]
    - m*H[t-1,0]/meta
    + (d0*Hvalue[1]**(2/3) + d1*Hvalue[1]**(5/3))*expected_N5*H[t-1,1]/Eint
  )

  #Special case: Second column
  H[t,1] = (H[t-1,1] - m*H[t-1,1]/meta + m*H[t-1,0]/meta
    - (d0*Hvalue[1]**(2/3) + d1*Hvalue[1]**(5/3))*expected_N5*H[t-1,1]/Eint
    -  w0*Hvalue[1]**(2/3)                       *expected_N5*H[t-1,1]/Eint
    +  w1*Hvalue[1]                                          *H[t-1,1]/Eint
    + (d0*Hvalue[2]**(2/3) + d1*Hvalue[2]**(5/3))*expected_N5*H[t-1,2]/Eint
  )

  #Special case: last column
  H[t,-1] = (
        H[t-1,-1]
    + m*H[t-1,-2]/meta
    +  w0*Hvalue[-2]**(2/3)                         * expected_N5*H[t-1,-2]/Eint
    -  w1*Hvalue[-2]                                             *H[t-1,-2]/Eint
    - (d0*Hvalue[-1]**(2/3) + d1*Hvalue[-1]**(5/3)) * expected_N5*H[t-1,-1]/Eint
  )

  ###################
  #Calculate G matrix
  ###################

  #Offsets from each column to its left and right neighbours
  #1:-1 = list(range(2,MAX_INDIVIDUALS-1)) #Center index
  #0:-2 = np.roll(1:-1,shift= 1)           #Left index
  #2: = np.roll(1:-1,shift=-1)           #Right index

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately following.
  G[t,2:-1] = (
         G[t-1,2:-1]
    - m*(G[t-1,2:-1] - G[t-1,1:-2])/Nint
    +    G[t-1,1:-2]*(     b0*expected_E1                 ) * Gvalue[1:-2]**(4/3)*np.log(Gvalue[1:-2])**(1/3)/Nint
    -    G[t-1,2:-1]*((b0+d0)*expected_E1 + d1*expected_E2) * Gvalue[2:-1]**(4/3)*np.log(Gvalue[2:-1])**(1/3)/Nint
    +    G[t-1,3:  ]*(     d0*expected_E1 + d1*expected_E2) * Gvalue[3:  ]**(4/3)*np.log(Gvalue[3:  ])**(1/3)/Nint
  )

  #######################
  #G matrix special cases
  #######################

  #Outer columns are special cases: First column
  G[t,0] = (
        G[t-1,0]
    - m*G[t-1,0]/Nint
    +   G[t-1,1]*(     d0*expected_E1 + d1*expected_E2) * Gvalue[ 1]**(4/3)*np.log(Gvalue[ 1])**(1/3)/Nint
  )

  #Special case: Second column
  G[t,1] = (
         G[t-1,1]
    - m*(G[t-1,1]-G[t-1,0])/Nint
    -    G[t-1, 1]*((b0+d0)*expected_E1 + d1*expected_E2) * Gvalue[ 1]**(4/3)*np.log(Gvalue[ 1])**(1/3)/Nint
    +    G[t-1, 2]*(     d0*expected_E1 + d1*expected_E2) * Gvalue[ 2]**(4/3)*np.log(Gvalue[ 2])**(1/3)/Nint
  )

  #Special case: last column
  G[t,-1] = (
        G[t-1,-1] +
      m*G[t-1,-2]/Nint
    +   G[t-1,-2]*(b0*expected_E1)                  *np.log(Gvalue[-2])**(1/3)*Gvalue[-2]**(4/3)/Nint
    -   G[t-1,-1]*(d0*expected_E1 + d1*expected_E2) *np.log(Gvalue[-1])**(1/3)*Gvalue[-1]**(4/3)/Nint
  )

  ###################
  #Calculate F matrix
  ###################

  #Offsets from each column to its left and right neighbours

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately followingself.
  f[t,1:-1] = (
      f[t-1,1:-1]
    + f[t-1,0:-2]*lam0*Fvalue[0:-2]
    + f[t-1,0:-2]*m*(1-Fvalue[0:-2]/Smeta)
    - f[t-1,1:-1]*lam0*Fvalue[1:-1]
    - f[t-1,1:-1]*m*(1-Fvalue[1:-1]/Smeta)
    - f[t-1,1:-1]*Fvalue[1:-1]**(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
    + f[t-1,2:  ]*Fvalue[2:  ]**(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
  )

  #######################
  #F matrix special cases
  #######################

  #Outer columns are special cases: First column
  f[t,0] = (
      f[t-1,0]
    - f[t-1,0]*m*(1-Fvalue[0 ]/Smeta)
    + f[t-1, 1]*Fvalue[ 1]**(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
  )

  #Special case: last column
  f[t,-1] = (
    f[t-1,-1]
    + f[t-1,-2]*m*(1-Fvalue[-2]/Smeta)
    - f[t-1,-1]*Fvalue[-1]**(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
    + f[t-1,-2]*lam0*Fvalue[-2]
  )

  ####################
  #Check Normalization
  ####################

  #see wether the probabilities sum up to 1
  sum_H[t] = np.sum(H[t, 1:]) #the sum of the probabilities of Energy at time t
  sum_G[t] = np.sum(G[t, 1:]) #the sum of the probabilities of Individuals at time t
  sum_F[t] = np.sum(f[t, 1:]) #the sum of the probabilities of Species at time t

  #Calculate <E>, <N>, <S>
  avg_E[t] = np.sum(Hvalue[2:] * H[t, 2:])
  avg_N[t] = np.sum(Gvalue[2:] * G[t, 2:])
  avg_S[t] = np.sum(Fvalue[2:] * f[t, 2:])

  f[t][f[t]<0] = 0
  G[t][G[t]<0] = 0
  H[t][H[t]<0] = 0


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

fig, ax = plt.subplots(1,3, sharex=False)
ax[0].plot(Fvalue,f[-1,:], label="Fvalue")
ax[0].legend()
ax[1].plot(Gvalue,G[-1,:], label="Gvalue")
ax[1].legend()
ax[2].plot(Hvalue,H[-1,:], label="Hvalue")
ax[2].legend()
fig.suptitle("State variable distribution at final time step")
plt.legend()
plt.show()

# lp = LineProfiler()
# lp_wrapper = lp(main)
# lp_wrapper()
# lp.print_stats()

#main()
