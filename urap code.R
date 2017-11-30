#assign parameters values
b0 = 0.0025
d0 = 0.001
lam0 = 0
m = 0.002
Emax = 300000
EEmax = 1 #EEmax = 1:E/Emax
w0 = 0.01
Smeta = 60
w1 = 0.0003
d1 = (b0-d0)/Emax
meta = 100 #1000/(E/N) of meta

#create probability matrix 
G <- matrix(, nrow = 50001, ncol = 251)
f <- matrix(, nrow = 50001, ncol = 65)
H <- matrix(, nrow = 50001, ncol = 324)

f[2,]=0
f[,1]=0


G[2,]=0
G[,1]=0


H[2,]=0
H[,1]=0
H[,324]=0

f[2,11]=1
G[2,11]=1
H[2,51]=1

for (i in 2:65){
  f[1,i] = (i-1)
}

for (i in 2:251){
  G[1,i] = (i-1)*10
}

for (i in 2:323){
  H[1,i] = (i-1)*1000
}

#avg_N1 = <N^(1/3)>
#avg_N2 = <1/ln(N)>
#avg_N3 = <1/ln(N)^(1/3)>
#avg_N4 = <ln(N)^(1/3)>
#avg_N5 = <N^(1/3)/ln(N)^(2/3)
for (t in 2:50000){
  avg_N1 = 0
  avg_N2 = 0
  avg_N3 = 0
  avg_N4 = 0
  avg_N5 = 0
  for (N in 2:(ncol(G))){
    avg_N1 = avg_N1 + G[1,N]^(1/3)*G[t,N]
    avg_N2 = avg_N2 + 1/log(G[1,N])*G[t,N]
    avg_N3 = avg_N3 + 1/log(G[1,N])^(1/3)*G[t,N]
    avg_N4 = avg_N4 + log(G[1,N])^(1/3)*G[t,N]
    avg_N5 = avg_N5 + G[1,N]^(1/3)/log(G[1,N])^(2/3)*G[t,N]
  }
  
  #avg_E1 = <E^(-1/3)>
  #avg_E2 = <E^(2/3)>
  avg_E1 = 0
  avg_E2 = 0
  for (E in 2:(ncol(H)-1)){
    avg_E1 = avg_E1 + H[1,E]^(-1/3)*H[t,E]
    avg_E2 = avg_E2 + H[1,E]^(2/3)*H[t,E]
  }
  
  #H[,]
  H[t+1,1] = H[t,1] - m*H[t,1]/meta + (d0*H[1,2]^(2/3) + 
                                         d1*H[1,2]^(5/3))*avg_N5*H[t,2]/1000
  
  H[t+1,2] = H[t,2]- m*H[t,2]/meta + m*H[t,1]/meta + (d0*H[1,3]^(2/3) + 
                                             d1*H[1,3]^(5/3))*avg_N5*H[t,3]/1000 - 
    (d0*H[1,2]^(2/3) + d1*H[1,2]^(5/3))*avg_N5*H[t,2]/1000 - 
    w0*H[1,2]^(2/3)*avg_N5*H[t,2]/1000 +
    w1*H[1,2]*H[t,2]/1000
  
  for (E in 3:(ncol(H)-2)){
    H[t+1,E]=H[t,E]- m*H[t,E]/meta + m*H[t,E-1]/meta + (d0*H[1,E+1]^(2/3) + 
                                               d1*H[1,E+1]^(5/3))*avg_N5*H[t,E+1]/1000 - 
      (d0*H[1,E]^(2/3) + d1*H[1,E]^(5/3))*avg_N5*H[t,E]/1000 - 
      w0*H[1,E]^(2/3)*avg_N5*H[t,E]/1000 +
      w1*H[1,E]*H[t,E]/1000 +
      w0*H[1,E-1]^(2/3)*avg_N5*H[t,E-1]/1000 -
      w1*H[1,E-1]*H[t,E-1]/1000
  }
  H[t+1,323] = H[t,323] + m*H[t,322]/meta + (d0*H[1,324]^(2/3) + d1*H[1,324]^(5/3)) *
    avg_N5*H[t,324]/1000 - (d0*H[1,323]^(2/3) + d1*H[1,323]^(5/3)) * avg_N5*H[t,323]/1000
  + w0*H[1,322]^(2/3)*avg_N5*H[t,322]/1000 - w1*H[1,322]*H[t,322]/1000
  
  

  #TODO: LEFT OFF HERE

  #G[,] 
  G[t+1,1] =G[t,1] - m*G[t,1]/10 + G[t,2]*log(G[1,2])^(1/3)*(d0*G[1,2]^(4/3)*avg_E1 + 
                                                               d1*avg_E2*G[1,2]^(4/3))/10
  G[t+1,2] = G[t,2]- m*(G[t,2]-G[t,1])/10 - G[t,2]*((b0 + d0)*avg_E1*log(G[1,2])^(1/3) +
                                            d1*avg_E2*log(G[1,2])^(1/3))*G[1,2]^(4/3)/10 +
    G[t,3]*(d0*avg_E1*log(G[1,3])^(1/3) + d1*avg_E2*log(G[1,3])^(1/3))*G[1,3]^(4/3)/10
  
  for (N in 3:(ncol(G)-1)){
    G[t+1,N] = G[t,N] - m*(G[t,N] - G[t,N-1])/10 -
      ((b0+d0)*avg_E1 + d1*avg_E2)*G[1,N]^(4/3)*log(G[1,N])^(1/3)*G[t,N]/10 +
      G[t,N-1]*b0*avg_E1*G[1,N-1]^(4/3)*log(G[1,N-1])^(1/3)/10 +
      G[t,N+1]*(d0*avg_E1 + d1*avg_E2)*G[1,N+1]^(4/3)*log(G[1,N+1])^(1/3)/10
  }
  G[t+1,251] = G[t,251] + m*G[t,250]/10 - G[t,251]*log(G[1,251])^(1/3)*G[1,251]^(4/3) *
    (d0*avg_E1 + d1*avg_E2)/10 + G[t,250]*(b0*avg_E1*log(G[1,250])^(1/3)*G[1,250]^(4/3)/10)
  
  #f[,]
  f[t+1,1] = f[t,1] - f[t,1]*m*(1-f[1,1]/Smeta) + 
    f[t,2]*(d0*avg_E1*avg_N2 + d1*avg_E2*avg_N2)*f[1,2]^(4/3)
  
  for (S in 2:(ncol(f)-1)){
    f[t+1,S]=f[t,S] - lam0*G[1,N]*f[t,S] + lam0*f[1,S-1]*f[t,S-1] -
      f[t,S]*m*(1-f[1,S]/Smeta) + f[t,S-1]*m*(1-f[1,S-1]/Smeta) + 
      f[t,S+1]*(d0*avg_E1*avg_N2 + d1*avg_E2*avg_N2)*f[1,S+1]^(4/3) - 
      f[t,S]*f[1,S]^(4/3)*(d0*avg_E1*avg_N2 + d1*avg_E2*avg_N2)
  }
  f[t+1,65]=f[t,65] + lam0*f[1,64]*f[t,64]
  + f[t,64]*m*(1-f[1,64]/Smeta) -
    f[t,65]*f[1,65]^(4/3)*(d0*avg_E1*avg_N2 + d1*avg_E2*avg_N2)
}

#see wether the probabilities sum up to 1
sum_H <- vector(,length = 50001)
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

#make some diagnostic plots
t_value <- seq(1, 50000)
plot(S_avg[1:50000]~t_value[1:50000], xlab = "t", ylab = "<S>")
plot(N_avg[1:50000]~t_value[1:50000], xlab = "t", ylab = "<N>")
plot(E_avg[1:50000]~t_value[1:50000], xlab = "t", ylab = "<E>")

plot(f[7000,] ~ f[1,])
plot(H[7000,] ~ H[1,])
plot(G[7000,] ~ G[1,])

#create the lookup table 
library(nleqslv)
vlookup <- matrix(,nrow = 2000, ncol = 100)

for (N in 2:2000){
  for (S in 1:min(N,100)){
    fun <- function(X){
      top <- 0
      bottom <- 0
      for (n in 1:N){
        top <- top + X^n/n
        bottom <- bottom + X^n
      }
      top/bottom - S/N
    }
    vlookup[N,S] = nleqslv(1,fun)$x
  }
}

ctable <- matrix(,nrow = 2000, ncol = 100)
for (N in 2:2000){
  for (S in 1:min(N,100)){
    y <- 0
    for (n in 1:N){
      y <- y + vlookup[N,S]^n/n
    }
    ctable[N,S] = 1/y
  }
}

write.table(ctable, file="ctable.csv",sep=",",row.names=T)
