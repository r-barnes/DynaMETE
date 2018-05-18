#!/usr/bin/env julia

import PyPlot



function UpdateH!(H,Hprev,expected_N5)
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

  return nothing
end



function UpdateG!(G,Gprev,expected_E1,expected_E2)
  #Our strategy is to perform our calculations as though all of the columns are
  #general cases. This results in incorrect values of the lestmost and rightmost
  #columns. We will fix these immediately following.
  G[3:end-1] = (
      Gprev[3:end-1] 
    - m*(Gprev[3:end-1] - Gprev[2:end-2])/Nint
    + Gprev[2:end-2].*(     b0*expected_E1                 ) .* Nvalue[2:end-2].^(4/3).*log.(Nvalue[2:end-2]).^(1/3)/Nint
    - Gprev[3:end-1].*((b0+d0)*expected_E1 + d1*expected_E2) .* Nvalue[3:end-1].^(4/3).*log.(Nvalue[3:end-1]).^(1/3)/Nint
    + Gprev[4:end  ].*(     d0*expected_E1 + d1*expected_E2) .* Nvalue[4:end  ].^(4/3).*log.(Nvalue[4:end  ]).^(1/3)/Nint
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
  return nothing
end



function UpdateF!(f,fprev,expected_N2,expected_E1,expected_E2)
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

  return nothing
end





######################
#CONSTANTS
######################
const b0              = 0.0025       #Birth rate per capita
const d0              = 0.001        #Death rate per capita
const lam0            = 0            #Speciation rate
const m               = 0.001        #Migration rate
const Emax            = 300000       #Soft maximum for energy (TODO)
const w0              = 0.01         #Ontogenic growth (growth of individual over its lifespan): See above
const w1              = 0.0003       #Ontogenic growth (growth of individual over its lifespan): See above
const Smeta           = 60           #Species richness of the meta community
const d1              = (b0-d0)/Emax #Density dependent contribution to death rate
const meta            = 100          #1000/(E/N) of meta
const Nint            = 10.          #Number of individuals per individual bin
const Eint            = 100.         #Number of energy units per energy bin
const Sint            = 1.           #Number of species per species bin
const MAX_TIMESTEP    = 50001        #Number of timesteps to take 
const MAX_INDIVIDUALS = 251          #Number of bins for individuals
const MAX_SPECIES     = 65           #Number of bins for species
const MAX_METABOLIC   = 3240         #Number of bins for energy

const Nvalue = Nint*collect(0:MAX_INDIVIDUALS-1)
const Evalue = Eint*collect(0:MAX_METABOLIC  -1)
const Svalue = Sint*collect(0:MAX_SPECIES    -1)



######################
#GLOBALS
######################

function main()
  fprev = zeros(MAX_SPECIES)
  Gprev = zeros(MAX_INDIVIDUALS)
  Hprev = zeros(MAX_METABOLIC)

  f = zeros(MAX_SPECIES)
  G = zeros(MAX_INDIVIDUALS)
  H = zeros(MAX_METABOLIC)

  times = []
  Fs    = []
  Gs    = []
  Hs    = []

  ######################
  #INITIAL CONDITIONS
  ######################

  fprev[6]   = 1 #100% probability of having 10 species at t=1
  Gprev[6]   = 1 #100% probability of having 10 individuals at t=1
  Hprev[101] = 1 #100% probability of having 10 units of metabolic rate (Watts) at t=1

  sum_H = zeros(MAX_TIMESTEP) #Use this later to check normalization
  sum_G = zeros(MAX_TIMESTEP) #Use this later to check normalization
  sum_F = zeros(MAX_TIMESTEP) #Use this later to check normalization

  avg_E = zeros(MAX_TIMESTEP) #Use this later to calculate average
  avg_N = zeros(MAX_TIMESTEP) #Use this later to calculate average
  avg_S = zeros(MAX_TIMESTEP) #Use this later to calculate average


  ######################
  #TIME LOOP
  ######################

  for t = 2:MAX_TIMESTEP
    if t%100==0
      println("t = $t")
    end
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
    UpdateH!(H,Hprev,expected_N5)


    ###################
    #Calculate G matrix
    ###################

    UpdateG!(G,Gprev,expected_E1,expected_E2)

    ###################
    #Calculate F matrix
    ###################

    UpdateF!(f,fprev,expected_N2,expected_E1,expected_E2)

    f[f.<1e-15] = 0
    G[G.<1e-15] = 0
    H[H.<1e-15] = 0

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

    append!(times,t)
    push!(Fs,copy(f))
    push!(Gs,copy(G))
    push!(Hs,copy(H))

    fprev = f
    Gprev = G
    Hprev = H
  end

  return sum_H, sum_G, sum_F, avg_S, avg_N, avg_E, f, G, H, times, Fs,Gs,Hs
end

sum_H, sum_G, sum_F, avg_S, avg_N, avg_E, f, G, H, times, Fs,Gs, Hs = main()






PyPlot.subplot(311)
PyPlot.plot(sum_H, label="sum_H")
PyPlot.legend()
PyPlot.ylim(-0.1,1.1)
PyPlot.subplot(312)
PyPlot.plot(sum_G, label="sum_G")
PyPlot.legend()
PyPlot.subplot(313)
PyPlot.plot(sum_F, label="sum_F")
PyPlot.legend()
PyPlot.suptitle("Normalization Check (should be 1)")



PyPlot.subplot(411)
PyPlot.plot(avg_S, label="avg_S")
PyPlot.legend()
PyPlot.subplot(412)
PyPlot.plot(avg_N, label="avg_N")
PyPlot.legend()
PyPlot.subplot(413)
PyPlot.plot(avg_E, label="avg_E")
PyPlot.legend()
PyPlot.subplot(414)
PyPlot.plot(avg_E./avg_N, label="avg_E/avg_N")
PyPlot.legend()
PyPlot.suptitle("Average of state variables")



PyPlot.subplot(311)
PyPlot.plot(Svalue,f, label="f Last")
PyPlot.legend()
PyPlot.subplot(312)
PyPlot.plot(Nvalue,G, label="G last")
PyPlot.legend()
PyPlot.subplot(313)
PyPlot.plot(Evalue,H, label="H Last")
PyPlot.legend()
PyPlot.suptitle("State variable distribution at final time step")



PyPlot.subplot(311)
PyPlot.plot(Svalue,Fs[25000], label="f Last")
PyPlot.legend()
PyPlot.subplot(312)
PyPlot.plot(Nvalue,Gs[25000], label="G last")
PyPlot.legend()
PyPlot.subplot(313)
PyPlot.plot(Evalue,Hs[25000], label="H Last")
PyPlot.legend()
PyPlot.suptitle("State variable distribution at final time step")




PyPlot.subplot(411)
PyPlot.plot(Nvalue,Gs[500], label="500")
PyPlot.legend()
PyPlot.subplot(412)
PyPlot.plot(Nvalue,Gs[1000], label="1000")
PyPlot.legend()
PyPlot.subplot(413)
PyPlot.plot(Nvalue,Gs[3500], label="3500")
PyPlot.legend()
PyPlot.subplot(414)
PyPlot.plot(Nvalue,Gs[4000], label="4000")
PyPlot.legend()


