#!/usr/bin/env julia

######################
#PACKAGE CONFIGURATION
######################


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
meta  = 100          #1000/(E/N) of meta

MAX_TIMESTEP    = 5000 #50001

MAX_INDIVIDUALS = 100 #251
MAX_SPECIES     = 65 # 65
MAX_METABOLIC   = 100 #324



######################
#GLOBALS
######################

#Arrays are arranged such that they are MAX_TIMESTEP high and (e.g.) MAX_SPECIES
#wide

# f = zeros((MAX_TIMESTEP, MAX_SPECIES))     #Total number of species
# G = zeros((MAX_TIMESTEP, MAX_INDIVIDUALS)) #Total number of individuals
# H = zeros((MAX_TIMESTEP, MAX_METABOLIC))   #Total metabolic rate

fprev = zeros(MAX_SPECIES)
Gprev = zeros(MAX_INDIVIDUALS)
Hprev = zeros(MAX_METABOLIC)

f = zeros(MAX_SPECIES)
G = zeros(MAX_INDIVIDUALS)
H = zeros(MAX_METABOLIC)


######################
#INITIAL CONDITIONS
######################

#NOTE: The equations below assume that Nvalue[0] = 0 and Evalue[0] = 0
#TODO: Try to eliminate Nint and Eint
Nint   = 10.
Eint   = 1000.
Sint   = 1.
Nvalue = Nint*collect(0:MAX_INDIVIDUALS-1)
Evalue = Eint*collect(0:MAX_METABOLIC  -1)
Svalue = Sint*collect(0:MAX_SPECIES    -1)

fprev[6] = 1 #100% probability of having 10 species at t=1
Gprev[6] = 1 #100% probability of having 10 individuals at t=1
Hprev[6] = 1 #100% probability of having 10 units of metabolic rate (Watts) at t=1

sum_H = zeros(MAX_TIMESTEP) #Use this later to check normalization
sum_G = zeros(MAX_TIMESTEP) #Use this later to check normalization
sum_F = zeros(MAX_TIMESTEP) #Use this later to check normalization

avg_E = zeros(MAX_TIMESTEP) #Use this later to calculate average
avg_N = zeros(MAX_TIMESTEP) #Use this later to calculate average
avg_S = zeros(MAX_TIMESTEP) #Use this later to calculate average


######################
#TIME LOOP
######################

t=2
# for t = 2:MAX_TIMESTEP
  ###########################
  #SET UP INTERMEDIATE VALUES
  ###########################

  #NOTE: This blows up if Nvalue[0]=0, so we only look at Nvalue[1:]
  logN        = log.(Nvalue[2:end])
  expected_N1 = sum(Nvalue[2:end].^(1/3)                .*Gprev[2:end]) #expected_N1 = <N^(1/3)>
  expected_N2 = sum(1./logN                             .*Gprev[2:end]) #expected_N2 = <1/ln(N)>
  expected_N3 = sum(1./logN.^(1/3)                      .*Gprev[2:end]) #expected_N3 = <1/ln(N)^(1/3)>
  expected_N4 = sum(logN.^(1/3)                         .*Gprev[2:end]) #expected_N4 = <ln(N)^(1/3)>
  expected_N5 = sum(Nvalue[2:end].^(1/3) ./ logN.^(2/3) .*Gprev[2:end]) #expected_N5 = <N^(1/3)/ln(N)^(2/3)

  #NOTE: This blows up if Evalue[0]=0, so we only look at Evalue[1:]
  expected_E1 = sum(Evalue[2:end].^(-1/3) .* Hprev[2:end])   #avg_E1 = <E^(-1/3)>
  expected_E2 = sum(Evalue[2:end].^( 2/3) .* Hprev[2:end])   #avg_E2 = <E^(2/3)>

  ###################
  #Calculate H matrix
  ###################

  #Offsets from each column to its left and right neighbours
  #2:end-1 = list(range(2,MAX_METABOLIC-1)) #Center index
  #1:end-2 = np.roll(2:end-1,shift= 1)           #Left index
  #2: = np.roll(2:end-1,shift=-1)           #Right index

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately following.
  H[2:end-1] = (
        Hprev[2:end-1] 
    - m*Hprev[2:end-1]/meta 
    + m*Hprev[1:end-2]/meta
    +  w0*Evalue[1:end-2].^(2/3)                             *expected_N5.*Hprev[1:end-2]/Eint
    -  w1*Evalue[1:end-2]                                                .*Hprev[1:end-2]/Eint
    - (d0*Evalue[2:end-1].^(2/3) + d1*Evalue[2:end-1].^(5/3))*expected_N5.*Hprev[2:end-1]/Eint
    -  w0*Evalue[2:end-1].^(2/3)                             *expected_N5.*Hprev[2:end-1]/Eint
    +  w1*Evalue[2:end-1]                                                .*Hprev[2:end-1]/Eint
    + (d0*Evalue[3:end  ].^(2/3) + d1*Evalue[3:end].^(5/3))  *expected_N5.*Hprev[3:end  ]/Eint
  )

  #######################
  #H matrix special cases
  #######################

  #Outer columns are special cases: First column
  H[1] = (Hprev[1] - m*Hprev[1]/meta
    + (d0*Evalue[2].^(2/3) + d1*Evalue[2].^(5/3))*expected_N5*Hprev[2]/Eint
  )

  #Special case: Second column
  H[2] = (Hprev[2] - m*Hprev[2]/meta + m*Hprev[1]/meta
    - (d0*Evalue[2].^(2/3) + d1*Evalue[2].^(5/3))*expected_N5*Hprev[2]/Eint
    -  w0*Evalue[2].^(2/3)                       *expected_N5*Hprev[2]/Eint
    +  w1*Evalue[2]                                          *Hprev[2]/Eint
    + (d0*Evalue[3].^(2/3) + d1*Evalue[3].^(5/3))*expected_N5*Hprev[3]/Eint
  )

  #Special case: last column
  H[end] = (
          Hprev[end] 
    +   m*Hprev[end-1]/meta
    +  w0*Evalue[end-1].^(2/3)                          * expected_N5*Hprev[end-1]/Eint
    -  w1*Evalue[end-1]                                              *Hprev[end-1]/Eint
    - (d0*Evalue[end  ].^(2/3) + d1*Evalue[end].^(5/3)) * expected_N5*Hprev[end  ]/Eint
  )

  ###################
  #Calculate G matrix
  ###################

  #Offsets from each column to its left and right neighbours
  #2:end-1 = list(range(2,MAX_INDIVIDUALS-1)) #Center index
  #1:end-2 = np.roll(2:end-1,shift= 1)           #Left index
  #2: = np.roll(2:end-1,shift=-1)           #Right index

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately following.
  G[2:end-1] = (
      Gprev[2:end-1] 
    - m*(Gprev[2:end-1] - Gprev[1:end-2])/Nint
    + Gprev[1:end-2].*(     b0*expected_E1                 ) .* Nvalue[1:end-2].^(4/3).*log.(Nvalue[1:end-2]).^(1/3)/Nint
    - Gprev[2:end-1].*((b0+d0)*expected_E1 + d1*expected_E2) .* Nvalue[2:end-1].^(4/3).*log.(Nvalue[2:end-1]).^(1/3)/Nint
    + Gprev[3:end  ].*(     d0*expected_E1 + d1*expected_E2) .* Nvalue[3:end  ].^(4/3).*log.(Nvalue[3:end  ]).^(1/3)/Nint
  )

  #######################
  #G matrix special cases
  #######################

  #Outer columns are special cases: First column
  G[1] = (Gprev[1] - m*Gprev[1]/Nint
    + Gprev[ 2]*(     d0*expected_E1 + d1*expected_E2) * Nvalue[ 2].^(4/3).*log.(Nvalue[ 2]).^(1/3)/Nint
  )

  #Special case: Second column
  G[2] = (Gprev[2]- m*(Gprev[2]-Gprev[1])/Nint
    - Gprev[ 2].*((b0+d0)*expected_E1 + d1*expected_E2) .* Nvalue[ 2].^(4/3).*log.(Nvalue[ 2]).^(1/3)/Nint
    + Gprev[ 3].*(     d0*expected_E1 + d1*expected_E2) .* Nvalue[ 3].^(4/3).*log.(Nvalue[ 3]).^(1/3)/Nint
  )

  #Special case: last column
  G[end] = (
      Gprev[end  ] + m*Gprev[end-1]/Nint
    + Gprev[end-1].*(b0*expected_E1)                  .*log.(Nvalue[end-1]).^(1/3).*Nvalue[end-1].^(4/3)/Nint
    - Gprev[end  ].*(d0*expected_E1 + d1*expected_E2) .*log.(Nvalue[end  ]).^(1/3).*Nvalue[end  ].^(4/3)/Nint
  )

  ###################
  #Calculate F matrix
  ###################

  #Offsets from each column to its left and right neighbours
  #2:end-1 = list(range(2,MAX_SPECIES-1))  #Center index
  #1:end-2 = np.roll(2:end-1,shift= 1)          #Left index
  #2: = np.roll(2:end-1,shift=-1)          #Right index

  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately followingself.
  f[2:end-1] = (
    lam0*Svalue[1:end-2].*fprev[1:end-2] 
    + fprev[1:end-2].*m.*(1-Svalue[1:end-2]/Smeta)
    + fprev[2:end-1] - lam0*Svalue[2:end-1].*fprev[2:end-1] - fprev[2:end-1].*m.*(1-Svalue[2:end-1]/Smeta)
    - fprev[2:end-1].*Svalue[2:end-1].^(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
    + fprev[3:end  ].*Svalue[3:end  ].^(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
  )

  #######################
  #F matrix special cases
  #######################

  #Outer columns are special cases: First column
  f[1] = (fprev[1] - fprev[1]*m*(1-Svalue[1 ]/Smeta)
    + fprev[ 2]*Svalue[ 2].^(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
  )

  #Special case: last column
  f[end] = (fprev[end] + fprev[end-1]*m*(1-Svalue[end-1]/Smeta)
    - fprev[end]*Svalue[end].^(4/3)*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
    + lam0*Svalue[end-1]*fprev[end-1]
  )

  ####################
  #Check Normalization
  ####################

  #see wether the probabilities sum up to 1
  sum_H[t] = sum(H[ 2:end]) #the sum of the probabilities of Energy at time t
  sum_G[t] = sum(G[ 2:end]) #the sum of the probabilities of Individuals at time t
  sum_F[t] = sum(f[ 2:end]) #the sum of the probabilities of Species at time t

  #Calculate <E>, <N>, <S>
  avg_E[t] = sum(Evalue[3:end] .* H[ 3:end])
  avg_N[t] = sum(Nvalue[3:end] .* G[ 3:end])
  avg_S[t] = sum(Svalue[3:end] .* f[ 3:end])
# end

fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
ax[0].plot(sum_H, label="sum_H")
ax[0].set_ylim([-0.1,1.1])
ax[1].plot(sum_G, label="sum_G")
ax[2].plot(sum_F, label="sum_F")
fig.suptitle("Normalization Check (should be 1)")
plt.legend()
plt.show()

fig, ax = plt.subplots(1,3, sharex=True)
ax[0].plot(avg_S, label="avg_S")
ax[1].plot(avg_N, label="avg_N")
ax[2].plot(avg_S, label="avg_S")
fig.suptitle("Average of state variables")
plt.legend()
plt.show()

fig, ax = plt.subplots(1,3, sharex=True)
ax[0].plot(Svalue,f[-1,:], label="Svalue")
ax[1].plot(Nvalue,G[-1,:], label="Nvalue")
ax[2].plot(Evalue,H[-1,:], label="Evalue")
fig.suptitle("State variable distribution at final time step")
plt.legend()
plt.show()

# lp = LineProfiler()
# lp_wrapper = lp(main)
# lp_wrapper()
# lp.print_stats()

#main()
