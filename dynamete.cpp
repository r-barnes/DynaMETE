#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>

// #include <boost/iostreams/filtering_stream.hpp>
// #include <boost/iostreams/filter/zlib.hpp>
// #include <boost/iostreams/filter/gzip.hpp>


class DynaSolver {
 public:
  ////////////////////////////
  //CONSTANTS
  //Rate Constants
  const double b0                    = 0.0025;        //Birth rate per capita
  const double d0                    = 0.001;         //Death rate per capita
  const double m                     = 0.001;         //Migration rate
  const double w0                    = 0.01;          //Ontogenic growth (growth of individual over its lifespan): See above
  const double w1                    = 0.0003;        //Ontogenic growth (growth of individual over its lifespan): See above
  const double lam0                  = 0;             //Speciation rate

  const double Emax                  = 300000;        //Soft maximum for energy (TODO)
  const double Smeta                 = 60;            //Species richness of the meta community
  const double d1                    = (b0-d0)/Emax;  //Density dependent contribution to death rate
  const double meta                  = 100;           //100/(E/N) of meta
  const double Nint                  = 10;            //Number of individuals per individual bin
  const double Eint                  = 100;           //Number of energy units per energy bin
  const double Sint                  = 1;             //Number of species per species bin
  const unsigned int MAX_TIMESTEP    = 20000;         //Number of timesteps to take
  const unsigned int MAX_INDIVIDUALS = 500;           //Number of bins for individuals
  const unsigned int MAX_SPECIES     = 65;            //Number of bins for species
  const unsigned int MAX_METABOLIC   = 3240;          //Number of bins for energy

  const double h = 1;                                 //Size of timestep

  typedef double ftype;
  typedef ftype* dvec;
  typedef std::list< std::vector<double> > savepoint_t;

  dvec f,      G,      H;

  dvec fnext,  Gnext,  Hnext;

  dvec Fvalue, Gvalue, Hvalue;

  dvec f_k1,   G_k1,   H_k1;
  dvec f_k2,   G_k2,   H_k2;
  dvec f_k3,   G_k3,   H_k3;
  dvec f_k4,   G_k4,   H_k4;

  dvec f_kin,   G_kin,   H_kin;

  std::vector<unsigned int> savetimes;
  savepoint_t Fsave, Gsave, Hsave;

  double* sum_H;
  double* sum_G;
  double* sum_F;
  double* avg_E;
  double* avg_N;
  double* avg_S;

  dvec Hvalue23,Hvalue53,Gvalue13,Gvalue43,Fvalue43;

 private:
  void SaveVec(double* v, const int len, std::ostream &fout) {
    fout.write(reinterpret_cast<const char*>( v ), len*sizeof( double ));
  }

  void SaveSavePoint(savepoint_t &sp, std::ostream &fout) {
    for(auto &v: sp)
      SaveVec(v.data(), v.size(), fout);
  }

 public:
  void saveAll(const std::string filename) {
    std::ofstream fout(filename, std::ios::out | std::ios::binary);

    // boost::iostreams::filtering_ostream fout;
    // fout.push(boost::iostreams::gzip_compressor());
    // fout.push(outfile);

    const unsigned int savepoints = Fsave.size();

    fout.write(reinterpret_cast<const char*>( &MAX_TIMESTEP   ), sizeof( unsigned int ));
    fout.write(reinterpret_cast<const char*>( &savepoints     ), sizeof( unsigned int ));
    fout.write(reinterpret_cast<const char*>( &MAX_SPECIES    ), sizeof( unsigned int ));    
    fout.write(reinterpret_cast<const char*>( &MAX_INDIVIDUALS), sizeof( unsigned int ));    
    fout.write(reinterpret_cast<const char*>( &MAX_METABOLIC  ), sizeof( unsigned int ));    

    SaveSavePoint(Fsave, fout);
    SaveSavePoint(Gsave, fout);
    SaveSavePoint(Hsave, fout);

    SaveVec(sum_H, MAX_TIMESTEP, fout);
    SaveVec(sum_G, MAX_TIMESTEP, fout);
    SaveVec(sum_F, MAX_TIMESTEP, fout);
    SaveVec(avg_E, MAX_TIMESTEP, fout);
    SaveVec(avg_N, MAX_TIMESTEP, fout);
    SaveVec(avg_S, MAX_TIMESTEP, fout);

    fout.flush();
  }

 private:
  ftype VecSum(const dvec v, const unsigned int len) const {
    ftype sum = 0;
    #pragma omp parallel for simd reduction(+:sum)
    #pragma acc parallel loop reduction(+:sum) present(v) present(this)
    for(unsigned int i=0;i<len;i++)
      sum += v[i];
    return sum;
  }

  ftype VecMultSum(const dvec a, const dvec b, const unsigned int len) const {
    ftype sum = 0;
    #pragma omp parallel for simd reduction(+:sum)
    #pragma acc parallel loop reduction(+:sum) present(a,b) present(this)
    for(unsigned int i=0;i<len;i++)
      sum += a[i]*b[i];
    return sum;
  }

  //Calculates result = \vec{a}+m*\vec{b}
  void VecFMA(const dvec a, const ftype m, const dvec b, const unsigned int len, dvec result) {
    #pragma omp parallel for simd
    #pragma acc parallel loop present(a,b,result,this)
    for(unsigned int i=0;i<len;i++)
      result[i] = a[i]+m*b[i];
  }

  void CopyVec(const dvec from, const unsigned int len, dvec to){
    #pragma omp parallel for simd
    #pragma acc parallel loop present(from,to,this)
    for(unsigned int i=0;i<len;i++)
      to[i]= from [i];
  }

  void CheckInitialization() const {
    assert(std::abs(1-VecSum(H,MAX_METABOLIC  ))<1e-6);
    assert(std::abs(1-VecSum(G,MAX_INDIVIDUALS))<1e-6);
    assert(std::abs(1-VecSum(f,MAX_SPECIES    ))<1e-6);
  }

  void GetNormalization(unsigned int t){
    //See whether the probabilities sum up to 1
    sum_H[t] = VecSum(H,MAX_METABOLIC  ); //the sum of the probabilities of Energy at time t
    sum_G[t] = VecSum(G,MAX_INDIVIDUALS); //the sum of the probabilities of Individuals at time t
    sum_F[t] = VecSum(f,MAX_SPECIES    ); //the sum of the probabilities of Species at time t    
  }

  void GetStateAverage(unsigned int t) { 
    avg_E[t] = VecMultSum(Hvalue, H, MAX_METABOLIC); //TODO: Check these
    avg_N[t] = VecMultSum(Gvalue, G, MAX_INDIVIDUALS);
    avg_S[t] = VecMultSum(Fvalue, f, MAX_SPECIES);
  }

  void VecPower(const dvec v, const unsigned int len, const ftype exp, dvec result) const {
    #pragma omp parallel for simd
    #pragma acc parallel loop present(v,result) present(this)
    for(unsigned int i=0;i<len;i++)
      result[i] = std::pow(v[i],exp);
  }

  void VecLogPower(const dvec v, const unsigned int len, const ftype exp, dvec result) const {
    #pragma omp parallel for simd
    #pragma acc parallel loop present(v,result) present(this)
    for(unsigned int i=0;i<len;i++)
      result[i] = std::pow(std::log(v[i]),exp);
  }  

  void MakeSavePoint(unsigned int t) {
    #pragma acc update host(f[0:MAX_SPECIES],G[0:MAX_INDIVIDUALS],H[0:MAX_METABOLIC])
    Fsave.emplace_back(f,f+MAX_SPECIES);
    Gsave.emplace_back(G,G+MAX_INDIVIDUALS);
    Hsave.emplace_back(H,H+MAX_METABOLIC);
    savetimes.emplace_back(t);
  }

 public:
  DynaSolver(){
    f      = new ftype[MAX_SPECIES];
    G      = new ftype[MAX_INDIVIDUALS];
    H      = new ftype[MAX_METABOLIC];

    fnext = new ftype[MAX_SPECIES];
    Gnext = new ftype[MAX_INDIVIDUALS];
    Hnext = new ftype[MAX_METABOLIC];


    Fvalue = new ftype[MAX_SPECIES];
    Gvalue = new ftype[MAX_INDIVIDUALS];
    Hvalue = new ftype[MAX_METABOLIC];

    sum_H = new double[MAX_TIMESTEP];
    sum_G = new double[MAX_TIMESTEP];
    sum_F = new double[MAX_TIMESTEP];
    avg_E = new double[MAX_TIMESTEP];
    avg_N = new double[MAX_TIMESTEP];
    avg_S = new double[MAX_TIMESTEP];

    Hvalue23 = new ftype[MAX_METABOLIC];
    Hvalue53 = new ftype[MAX_METABOLIC];
    Gvalue13 = new ftype[MAX_INDIVIDUALS];
    Gvalue43 = new ftype[MAX_INDIVIDUALS];
    Fvalue43 = new ftype[MAX_SPECIES];

    f_k1 = new ftype[MAX_SPECIES];
    G_k1 = new ftype[MAX_INDIVIDUALS];
    H_k1 = new ftype[MAX_METABOLIC];
    f_k2 = new ftype[MAX_SPECIES];
    G_k2 = new ftype[MAX_INDIVIDUALS];
    H_k2 = new ftype[MAX_METABOLIC];
    f_k3 = new ftype[MAX_SPECIES];
    G_k3 = new ftype[MAX_INDIVIDUALS];
    H_k3 = new ftype[MAX_METABOLIC];    
    f_k4 = new ftype[MAX_SPECIES];
    G_k4 = new ftype[MAX_INDIVIDUALS];
    H_k4 = new ftype[MAX_METABOLIC];   

    f_kin = new ftype[MAX_SPECIES]; 
    G_kin = new ftype[MAX_INDIVIDUALS]; 
    H_kin = new ftype[MAX_METABOLIC];

    for(unsigned int i=0;i<MAX_SPECIES;    i++) Fvalue[i] = i*Sint;
    for(unsigned int i=0;i<MAX_INDIVIDUALS;i++) Gvalue[i] = i*Nint;
    for(unsigned int i=0;i<MAX_METABOLIC;  i++) Hvalue[i] = i*Eint;

    for(unsigned int i=0;i<MAX_SPECIES;    i++) f[i] = 0;
    for(unsigned int i=0;i<MAX_INDIVIDUALS;i++) G[i] = 0;
    for(unsigned int i=0;i<MAX_METABOLIC;  i++) H[i] = 0;

    f[5]   = 1;
    G[5]   = 1;
    H[100] = 1;

    #pragma acc enter data copyin(this[0:1],fprev[0:MAX_SPECIES],f[0:MAX_SPECIES],Fvalue[0:MAX_SPECIES],Gvalue[0:MAX_INDIVIDUALS],Hvalue[0:MAX_METABOLIC],Gprev[0:MAX_INDIVIDUALS],Hprev[0:MAX_METABOLIC],G[0:MAX_INDIVIDUALS],H[0:MAX_METABOLIC]) create(sum_H[0:MAX_TIMESTEP],sum_G[0:MAX_TIMESTEP],sum_F[0:MAX_TIMESTEP],avg_E[0:MAX_TIMESTEP],avg_N[0:MAX_TIMESTEP],avg_S[0:MAX_TIMESTEP],Hvalue23[0:MAX_METABOLIC],Hvalue53[0:MAX_METABOLIC],Gvalue13[0:MAX_INDIVIDUALS],Gvalue43[0:MAX_INDIVIDUALS],Fvalue43[0:MAX_SPECIES])

    CheckInitialization();

    GetNormalization(0);
    GetStateAverage(0);
  }

  ~DynaSolver(){
    #pragma acc exit data delete(fprev[0:MAX_SPECIES],f[0:MAX_SPECIES],Fvalue[0:MAX_SPECIES],Gvalue[0:MAX_INDIVIDUALS],Hvalue[0:MAX_METABOLIC],Gprev[0:MAX_INDIVIDUALS],Hprev[0:MAX_METABOLIC],G[0:MAX_INDIVIDUALS],H[0:MAX_METABOLIC],sum_H[0:MAX_TIMESTEP],sum_G[0:MAX_TIMESTEP],sum_F[0:MAX_TIMESTEP],avg_E[0:MAX_TIMESTEP],avg_N[0:MAX_TIMESTEP],avg_S[0:MAX_TIMESTEP],Hvalue23[0:MAX_METABOLIC],Hvalue53[0:MAX_METABOLIC],Gvalue13[0:MAX_INDIVIDUALS],Gvalue43[0:MAX_INDIVIDUALS],Fvalue43[0:MAX_SPECIES])

    delete[] f;
    delete[] G;
    delete[] H;

    delete[] fnext;
    delete[] Gnext;
    delete[] Hnext;

    delete[] Fvalue;
    delete[] Gvalue;
    delete[] Hvalue;
    delete[] sum_H;
    delete[] sum_G;
    delete[] sum_F;
    delete[] avg_E;
    delete[] avg_N;
    delete[] avg_S;
    delete[] Hvalue23;
    delete[] Hvalue53;
    delete[] Gvalue13;
    delete[] Gvalue43;
    delete[] Fvalue43;
    delete[] f_k1;
    delete[] G_k1;
    delete[] H_k1;
    delete[] f_k2;
    delete[] G_k2;
    delete[] H_k2;
    delete[] f_k3;
    delete[] G_k3;
    delete[] H_k3;
    delete[] f_k4;
    delete[] G_k4;
    delete[] H_k4;
    delete[] f_kin;
    delete[] G_kin;
    delete[] H_kin;
  }

  void printConfig(){
    std::cout<<"c b0              = " << b0              <<std::endl;
    std::cout<<"c d0              = " << d0              <<std::endl;
    std::cout<<"c lam0            = " << lam0            <<std::endl;
    std::cout<<"c m               = " << m               <<std::endl;
    std::cout<<"c Emax            = " << Emax            <<std::endl;
    std::cout<<"c w0              = " << w0              <<std::endl;
    std::cout<<"c w1              = " << w1              <<std::endl;
    std::cout<<"c Smeta           = " << Smeta           <<std::endl;
    std::cout<<"c d1              = " << d1              <<std::endl;
    std::cout<<"c meta            = " << meta            <<std::endl;
    std::cout<<"c Nint            = " << Nint            <<std::endl;
    std::cout<<"c Eint            = " << Eint            <<std::endl;
    std::cout<<"c Sint            = " << Sint            <<std::endl;
    std::cout<<"c MAX_TIMESTEP    = " << MAX_TIMESTEP    <<std::endl;
    std::cout<<"c MAX_INDIVIDUALS = " << MAX_INDIVIDUALS <<std::endl;
    std::cout<<"c MAX_SPECIES     = " << MAX_SPECIES     <<std::endl;
    std::cout<<"c MAX_METABOLIC   = " << MAX_METABOLIC   <<std::endl;
  }

  void step(
    const double t,
    const dvec fprev,
    const dvec Gprev,
    const dvec Hprev,
    dvec f,
    dvec G,
    dvec H
  ) const {
    const auto avg_N_prev = VecMultSum(Gvalue, Gprev, MAX_INDIVIDUALS); //TODO: Check these
    const auto avg_S_prev = VecMultSum(Fvalue, fprev, MAX_SPECIES    ); //TODO: Check these

    const ftype n_s = avg_N_prev/avg_S_prev; //Constant avg_N divided by avg_S

    //Formula to calculate logN
    const ftype logN = std::log(n_s * std::log(n_s * std::log(n_s * std::log(n_s * std::log(n_s))))); 


    VecPower   (Hvalue,MAX_METABOLIC,  2./3,Hvalue23);
    VecPower   (Hvalue,MAX_METABOLIC,  5./3,Hvalue53);
    VecLogPower(Gvalue,MAX_INDIVIDUALS,1./3,Gvalue13);
    VecPower   (Gvalue,MAX_INDIVIDUALS,4./3,Gvalue43);    
    VecPower   (Fvalue,MAX_SPECIES,    4./3,Fvalue43);

    const auto Flen = MAX_SPECIES;
    const auto Hlen = MAX_METABOLIC;
    const auto Glen = MAX_INDIVIDUALS;

    //expected_N2 = <1/ln(N)>
    ftype expected_N2 = 0;
    #pragma omp parallel for simd reduction(+:expected_N2)
    #pragma acc parallel loop reduction(+:expected_N2) async(0) present(this)
    for(unsigned int i=0;i<MAX_INDIVIDUALS;i++)
      expected_N2 += 1/logN*Gprev[i];

    //expected_N3 = <1/ln(N)^(1/3)>
    ftype expected_N3 = 0;
    #pragma omp parallel for simd reduction(+:expected_N3)
    #pragma acc parallel loop reduction(+:expected_N3) async(1) present(this)
    for(unsigned int i=0;i<MAX_INDIVIDUALS;i++)
      expected_N3 += std::pow(1/logN,1./3) * Gprev[i];

    //expected_N4 = <ln(N)^(1/3)>
    ftype expected_N4 = 0;
    #pragma omp parallel for simd reduction(+:expected_N4)
    #pragma acc parallel loop reduction(+:expected_N4) async(2) present(this)
    for(unsigned int i=0;i<MAX_INDIVIDUALS;i++)
      expected_N4 += std::pow(logN,1./3) * Gprev[i];

    //expected_N5 = <N^(1/3)/ln(N)^(2/3)
    ftype expected_N5 = 0;
    #pragma omp parallel for simd reduction(+:expected_N5)
    #pragma acc parallel loop reduction(+:expected_N5) async(3) present(this)
    for(unsigned int i=0;i<MAX_INDIVIDUALS;i++)
      expected_N5 += std::pow(Gvalue[i],1./3)/std::pow(logN,2./3) * Gprev[i];

    ftype expected_E1 = 0;
    #pragma omp parallel for simd reduction(+:expected_E1)
    #pragma acc parallel loop reduction(+:expected_E1) async(4) present(this)
    for(unsigned int i=0;i<MAX_METABOLIC;i++)
      expected_E1 += std::pow(Hvalue[i],1./3)*Hprev[i];
    expected_E1 = 1/expected_E1;

    //avg_E2 = <E^(2/3)>
    ftype expected_E2 = 0;
    #pragma omp parallel for simd reduction(+:expected_E2)
    #pragma acc parallel loop reduction(+:expected_E2) async(5) present(this)
    for(unsigned int i=0;i<MAX_METABOLIC;i++)
      expected_E2 += std::pow(Hvalue[i],2./3)*Hprev[i];

    #pragma acc update host (Hprev[0:MAX_METABOLIC])    async(6)
    #pragma acc update host (Gprev[0:MAX_INDIVIDUALS])  async(7)
    #pragma acc update host (fprev[0:MAX_SPECIES])      async(8)
    // #pragma acc update host (Hprev[0:2],Hprev[MAX_METABOLIC-2:MAX_METABOLIC])     async(6)
    // #pragma acc update host (Gprev[0:2],Gprev[MAX_INDIVIDUALS-2:MAX_INDIVIDUALS]) async(7)
    // #pragma acc update host (fprev[0:2],fprev[MAX_SPECIES-2:MAX_SPECIES])         async(8)

    #pragma acc wait


    #pragma omp parallel
    {

    //##################
    //Calculate H matrix
    //##################

    //Leaves out edge cells
    #pragma omp for simd nowait
    #pragma acc parallel loop async(0) present(this)
    for(unsigned int i=2;i<Hlen-1;i++){
      H[i] = (
        -   m*Hprev[i  ]/meta
        +   m*Hprev[i-1]/meta
        +  w0*Hvalue23[i-1]                     *expected_N5*Hprev[i-1]/Eint
        -  w1*Hvalue[i-1]                                   *Hprev[i-1]/Eint
        - (d0*Hvalue23[i  ] + d1*Hvalue53[i])   *expected_N5*Hprev[i  ]/Eint
        -  w0*Hvalue23[i  ]                     *expected_N5*Hprev[i  ]/Eint
        +  w1*Hvalue[i  ]                                   *Hprev[i  ]/Eint
        + (d0*Hvalue23[i+1] + d1*Hvalue53[i+1]) *expected_N5*Hprev[i+1]/Eint
      );
    }

    //##################
    //Calculate G matrix
    //##################

    #pragma omp for simd nowait
    #pragma acc parallel loop async(1) present(this)
    for(unsigned int i=2;i<Glen-1;i++){
      G[i] = (
        - m*(Gprev[i  ] - Gprev[i-1])   
        +    Gprev[i-1]*(     b0*expected_E1                 ) * Gvalue43[i-1]*Gvalue13[i-1]
        -    Gprev[i  ]*((b0+d0)*expected_E1 + d1*expected_E2) * Gvalue43[i  ]*Gvalue13[i  ]
        +    Gprev[i+1]*(     d0*expected_E1 + d1*expected_E2) * Gvalue43[i+1]*Gvalue13[i+1]
        )/Nint;
    }    

    // ###################
    // #Calculate F matrix
    // ###################

    #pragma omp for simd
    #pragma acc parallel loop async(2) present(this)
    for(unsigned int i=1;i<Flen-1;i++){
      f[i] = (
          fprev[i-1]*lam0*Fvalue[i-1]
        + fprev[i-1]*m*(1-Fvalue[i-1]/Smeta)
        - fprev[i  ]*lam0*Fvalue[i]
        - fprev[i  ]*m*(1-Fvalue[i]/Smeta)
        - fprev[i  ]*Fvalue43[i]*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
        + fprev[i+1]*Fvalue43[i+1]*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
      );
    }

    }

    #pragma acc wait


    //######################
    //H matrix special cases
    //######################

    //Outer columns are special cases: First column
    H[0] = (
      - m*Hprev[0]/meta
      + (d0*Hvalue23[1] + d1*Hvalue53[1])*expected_N5*Hprev[1]/Eint
    );

    //Special case: Second column
    H[1] = (
      - m*Hprev[1]/meta + m*Hprev[0]/meta
      - (d0*Hvalue23[1] + d1*Hvalue53[1])*expected_N5*Hprev[1]/Eint
      -  w0*Hvalue23[1]                  *expected_N5*Hprev[1]/Eint
      +  w1*Hvalue[1]                                *Hprev[1]/Eint
      + (d0*Hvalue23[2] + d1*Hvalue53[2])*expected_N5*Hprev[2]/Eint
    );

    //Special case: last column
    H[Hlen-1] = (
      + m*Hprev[Hlen-2]/meta
      +  w0*Hvalue23[Hlen-2]                        * expected_N5*Hprev[Hlen-2]/Eint
      -  w1*Hvalue[Hlen-2]                                       *Hprev[Hlen-2]/Eint
      - (d0*Hvalue23[Hlen-1] + d1*Hvalue53[Hlen-1]) * expected_N5*Hprev[Hlen-1]/Eint
    );



    // #######################
    // #G matrix special cases
    // #######################

    //Outer columns are special cases: First column
    G[0] = (
      - m*Gprev[0]/Nint
      +   Gprev[1]*(     d0*expected_E1 + d1*expected_E2) * Gvalue43[ 1]*Gvalue13[ 1]/Nint
    );

    //Special case: Second column
    G[1] = (
      - m*(Gprev[1]-Gprev[0])/Nint
      -    Gprev[ 1]*((b0+d0)*expected_E1 + d1*expected_E2) * Gvalue43[ 1]*Gvalue13[ 1]/Nint
      +    Gprev[ 2]*(     d0*expected_E1 + d1*expected_E2) * Gvalue43[ 2]*Gvalue13[ 2]/Nint
    );

    //Special case: last column
    G[Glen-1] = (
        m*Gprev[Glen-2]/Nint
      +   Gprev[Glen-2]*(b0*expected_E1)                  *Gvalue13[Glen-2]*Gvalue43[Glen-2]/Nint
      -   Gprev[Glen-1]*(d0*expected_E1 + d1*expected_E2) *Gvalue13[Glen-1]*Gvalue43[Glen-1]/Nint
    );



    // #######################
    // #F matrix special cases
    // #######################

    //Outer columns are special cases: First column
    f[0] = (
      - fprev[0]*m*(1-Fvalue[0 ]/Smeta)
      + fprev[ 1]*Fvalue43[ 1]*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
    );

    //Special case: last column
    f[Flen-1] = (
      + fprev[Flen-2]*m*(1-Fvalue[Flen-2]/Smeta)
      - fprev[Flen-1]*Fvalue43[Flen-1]*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
      + fprev[Flen-2]*lam0*Fvalue[Flen-2]
    );



    #pragma acc update device (H[0:MAX_METABOLIC])
    #pragma acc update device (G[0:MAX_INDIVIDUALS])
    #pragma acc update device (f[0:MAX_SPECIES])

    // #pragma acc update device (H[0:2],H[MAX_METABOLIC-2:MAX_METABOLIC])    
    // #pragma acc update device (G[0:2],G[MAX_INDIVIDUALS-2:MAX_INDIVIDUALS])
    // #pragma acc update device (f[0:2],f[MAX_SPECIES-2:MAX_SPECIES])        

    // f[t][f[t]<0] = 0
    // G[t][G[t]<0] = 0
    // H[t][H[t]<0] = 0
  }

  void RungeOp(
    const unsigned int t,
    const double m,
    const bool input_is_prev,
    dvec f_kout,
    dvec G_kout,
    dvec H_kout    
  ){
    if(input_is_prev)
      step(t,f,G,H,f_kout,G_kout,H_kout);   
    else
      step(t,f_kin,G_kin,H_kin,f_kout,G_kout,H_kout);  

    VecFMA(f, m, f_kout, MAX_SPECIES,     f_kin);
    VecFMA(G, m, G_kout, MAX_INDIVIDUALS, G_kin);
    VecFMA(H, m, H_kout, MAX_METABOLIC,   H_kin);
  }

  void RungeSum(
    const dvec y,
    const dvec k1,
    const dvec k2,
    const dvec k3,
    const dvec k4,
    const unsigned int len,
    dvec ynext
  ){
    #pragma omp for simd nowait
    #pragma acc parallel loop async(0) present(this)
    for(unsigned int i=0;i<len;i++)
      ynext[i] = y[i] + 1./6*k1[i] + 1./3*(k2[i]+k3[i])+1./6*k4[i];
  }

  void RungeKuttaStep(const unsigned int t){
    RungeOp(t,     h/2, true,  f_k1, G_k1, H_k1);
    RungeOp(t+h/2, h/2, false, f_k2, G_k2, H_k2);
    RungeOp(t+h/2, h,   false, f_k3, G_k3, H_k3);
    RungeOp(t+h,   0,   false, f_k4, G_k4, H_k4);

    RungeSum(f,f_k1,f_k2,f_k3,f_k4,MAX_SPECIES,    fnext);
    RungeSum(G,G_k1,G_k2,G_k3,G_k4,MAX_INDIVIDUALS,Gnext);
    RungeSum(H,H_k1,H_k2,H_k3,H_k4,MAX_METABOLIC,  Hnext);

    CopyVec(fnext,MAX_SPECIES,    f);
    CopyVec(Gnext,MAX_INDIVIDUALS,G);
    CopyVec(Hnext,MAX_METABOLIC,  H);
  }

  void EulerStep(const unsigned int t){
    step(t,f,G,H,f_k1,G_k1,H_k1);

    VecFMA(f, h, f_k1, MAX_SPECIES,     f);
    VecFMA(G, h, G_k1, MAX_INDIVIDUALS, G);
    VecFMA(H, h, H_k1, MAX_METABOLIC,   H);
  }

  void run() {
    MakeSavePoint(0);
    for(unsigned int t=1;t<MAX_TIMESTEP;t++){
      if(t%100==0)
        std::cerr<<"p t = "<<t<<std::endl;
      RungeKuttaStep(t);
      GetNormalization(t);
      //Calculate <E>, <N>, <S>
      GetStateAverage(t);
      if(t%1==0)
        MakeSavePoint(t);
    }    
  }
};



int main(int argc, char **argv){
  if(argc!=2){
    std::cerr<<"Syntax: "<<argv[0]<<" <OUTPUT FILE>"<<std::endl;
    return -1;
  }

  DynaSolver ds;
  ds.printConfig();
  ds.run();
  ds.saveAll(argv[1]);
}
