#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>


class DynaSolver {
 public:
  ////////////////////////////
  //CONSTANTS
  const double b0              = 0.0025;        //Birth rate per capita
  const double d0              = 0.001;         //Death rate per capita
  const double lam0            = 0;             //Speciation rate
  const double m               = 0.001;         //Migration rate
  const double Emax            = 300000;        //Soft maximum for energy (TODO)
  const double w0              = 0.01;          //Ontogenic growth (growth of individual over its lifespan): See above
  const double w1              = 0.0003;        //Ontogenic growth (growth of individual over its lifespan): See above
  const double Smeta           = 60;            //Species richness of the meta community
  const double d1              = (b0-d0)/Emax;  //Density dependent contribution to death rate
  const double meta            = 100;           //100/(E/N) of meta
  const double Nint            = 10;            //Number of individuals per individual bin
  const double Eint            = 100;           //Number of energy units per energy bin
  const double Sint            = 1;             //Number of species per species bin
  const double MAX_TIMESTEP    = 20000;         //Number of timesteps to take
  const double MAX_INDIVIDUALS = 251;           //Number of bins for individuals
  const double MAX_SPECIES     = 65;            //Number of bins for species
  const double MAX_METABOLIC   = 3240;          //Number of bins for energy

  typedef std::vector<double> dvec;
  typedef std::list< dvec > savepoint_t;

  dvec fprev,  Gprev, Hprev;
  dvec f,      G,     H;
  dvec Fvalue, Gvalue, Hvalue;

  std::vector<unsigned int> savetimes;
  savepoint_t Fsave, Gsave, Hsave;

  dvec sum_H;
  dvec sum_G;
  dvec sum_F;
  dvec avg_E;
  dvec avg_N;
  dvec avg_S;

 private:
  void SaveVec(const dvec &v, boost::iostreams::filtering_ostream &fout) const {
    fout.write(reinterpret_cast<const char*>( v.data() ), v.size()*sizeof( double ));
  }

  void SaveSavePoint(const savepoint_t &sp, boost::iostreams::filtering_ostream &fout) const {
    for(const auto &v: sp)
      SaveVec(v, fout);
  }

 public:
  void saveAll(const std::string filename) const {
    std::ofstream outfile(filename, std::ios::out | std::ios::binary);

    boost::iostreams::filtering_ostream fout;
    fout.push(boost::iostreams::gzip_compressor());
    fout.push(outfile);

    const unsigned int timesteps  = sum_H.size();
    const unsigned int savepoints = Fsave.size();
    const unsigned int Fwidth     = Fsave.front().size();
    const unsigned int Gwidth     = Gsave.front().size();
    const unsigned int Hwidth     = Hsave.front().size();

    fout.write(reinterpret_cast<const char*>( &timesteps  ), sizeof( unsigned int ));
    fout.write(reinterpret_cast<const char*>( &savepoints ), sizeof( unsigned int ));
    fout.write(reinterpret_cast<const char*>( &Fwidth     ), sizeof( unsigned int ));    
    fout.write(reinterpret_cast<const char*>( &Gwidth     ), sizeof( unsigned int ));    
    fout.write(reinterpret_cast<const char*>( &Hwidth     ), sizeof( unsigned int ));    

    SaveSavePoint(Fsave, fout);
    SaveSavePoint(Gsave, fout);
    SaveSavePoint(Hsave, fout);

    SaveVec(sum_H, fout);
    SaveVec(sum_G, fout);
    SaveVec(sum_F, fout);
    SaveVec(avg_E, fout);
    SaveVec(avg_N, fout);
    SaveVec(avg_S, fout);

    fout.flush();
  }

 private:
  double VecSum(const dvec &v) const {
    double sum = 0;
    #pragma omp parallel for simd reduction(+:sum)
    for(unsigned int i=0;i<v.size();i++)
      sum += v[i];
    return sum;
  }

  double VecMultSum(const dvec &a, const dvec &b) const {
    double sum = 0;
    assert(a.size()==b.size());
    #pragma omp parallel for simd reduction(+:sum)
    for(unsigned int i=0;i<a.size();i++)
      sum += a[i]*b[i];
    return sum;
  }

  void CheckInitialization() const {
    assert(std::abs(1-VecSum(Hprev))<1e-6);
    assert(std::abs(1-VecSum(Gprev))<1e-6);
    assert(std::abs(1-VecSum(fprev))<1e-6);
  }

  void GetNormalization(unsigned int t){
    //See whether the probabilities sum up to 1
    sum_H[t] = VecSum(H); //the sum of the probabilities of Energy at time t
    sum_G[t] = VecSum(G); //the sum of the probabilities of Individuals at time t
    sum_F[t] = VecSum(f); //the sum of the probabilities of Species at time t    
  }

  void GetStateAverage(unsigned int t) { 
    avg_E[t] = VecMultSum(Hvalue,H); //TODO: Check these
    avg_N[t] = VecMultSum(Gvalue,G);
    avg_S[t] = VecMultSum(Fvalue,f);
  }

  dvec VecPower(const dvec &v, const double exp) const {
    dvec temp(v.size());
    #pragma omp parallel for simd
    for(unsigned int i=0;i<v.size();i++)
      temp[i] = std::pow(v[i],exp);
    return temp;
  }

  dvec VecLogPower(const dvec &v, const double exp) const {
    dvec temp(v.size());
    #pragma omp parallel for simd
    for(unsigned int i=0;i<v.size();i++)
      temp[i] = std::pow(std::log(v[i]),exp);
    return temp;
  }  

  void MakeSavePoint(unsigned int t) {
    Fsave.push_back(f);
    Gsave.push_back(G);
    Hsave.push_back(H);
    savetimes.push_back(t);
  }

 public:
  DynaSolver(){
    fprev.resize (MAX_SPECIES,0);
    f.resize     (MAX_SPECIES,0);
    Fvalue.resize(MAX_SPECIES,0);
    Gprev.resize (MAX_INDIVIDUALS,0);
    G.resize     (MAX_INDIVIDUALS,0);
    Gvalue.resize(MAX_INDIVIDUALS,0);
    Hprev.resize (MAX_METABOLIC,0);
    H.resize     (MAX_METABOLIC,0);
    Hvalue.resize(MAX_METABOLIC,0);

    sum_H.resize(MAX_TIMESTEP);
    sum_G.resize(MAX_TIMESTEP);
    sum_F.resize(MAX_TIMESTEP);
    avg_E.resize(MAX_TIMESTEP);
    avg_N.resize(MAX_TIMESTEP);
    avg_S.resize(MAX_TIMESTEP);

    for(int i=0;i<MAX_INDIVIDUALS;i++) Gvalue[i] = i*Nint;
    for(int i=0;i<MAX_METABOLIC;  i++) Hvalue[i] = i*Eint;
    for(int i=0;i<MAX_SPECIES;    i++) Fvalue[i] = i*Sint;

    f[5]   = 1;
    G[5]   = 1;
    H[100] = 1;

    fprev = f;
    Gprev = G;
    Hprev = H;

    CheckInitialization();

    GetNormalization(0);
    GetStateAverage(0);
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

  void step(unsigned int t){
    const double n_s = avg_N[t-1]/avg_S[t-1]; //Constant avg_N divided by avg_S

    //Formula to calculate logN
    const double logN = std::log(n_s * std::log(n_s * std::log(n_s * std::log(n_s * std::log(n_s))))); 

    //expected_N2 = <1/ln(N)>
    double expected_N2 = 0;
    for(const auto x: Gprev)
      expected_N2 += 1/logN*x;

    //expected_N3 = <1/ln(N)^(1/3)>
    double expected_N3 = 0;
    for(const auto x: Gprev)
      expected_N3 += std::pow(1/logN,1./3) * x;

    //expected_N4 = <ln(N)^(1/3)>
    double expected_N4 = 0;
    for(const auto x: Gprev)
      expected_N4 += std::pow(logN,1./3) * x;

    //expected_N5 = <N^(1/3)/ln(N)^(2/3)
    double expected_N5 = 0;
    for(unsigned int i=0;i<Gvalue.size();i++)
      expected_N5 += std::pow(Gvalue[i],1./3)/std::pow(logN,2./3) * Gprev[i];

    double expected_E1 = 0;
    for(unsigned int i=0;i<Hvalue.size();i++)
      expected_E1 += std::pow(Hvalue[i],1./3)*Hprev[i];
    expected_E1 = 1/expected_E1;

    //avg_E2 = <E^(2/3)>
    double expected_E2 = 0;
    for(unsigned int i=0;i<Hvalue.size();i++)
      expected_E2 += std::pow(Hvalue[i],2./3)*Hprev[i];



    //##################
    //Calculate H matrix
    //##################
    const auto Hvalue23 = VecPower(Hvalue,2./3);
    const auto Hvalue53 = VecPower(Hvalue,5./3);

    const auto Hlen = H.size();

    //Leaves out edge cells
    #pragma omp parallel for simd
    for(uint i=1;i<Hlen-1;i++){
      H[i] = (
              Hprev[i  ]
        -   m*Hprev[i  ]/meta
        +   m*Hprev[i-1]/meta
        +  w0*Hvalue23[i-1]                        *expected_N5*Hprev[i-1]/Eint
        -  w1*Hvalue[i-1]                                           *Hprev[i-1]/Eint
        - (d0*Hvalue23[i  ] + d1*Hvalue53[i]) *expected_N5*Hprev[i  ]/Eint
        -  w0*Hvalue23[i  ]                        *expected_N5*Hprev[i  ]/Eint
        +  w1*Hvalue[i  ]                                           *Hprev[i  ]/Eint
        + (d0*Hvalue23[i+1] + d1*Hvalue53[i+1]) *expected_N5*Hprev[i+1]/Eint
      );
    }

    //######################
    //H matrix special cases
    //######################

    //Outer columns are special cases: First column
    H[0] = (
          Hprev[0]
      - m*Hprev[0]/meta
      + (d0*Hvalue23[1] + d1*Hvalue53[1])*expected_N5*Hprev[1]/Eint
    );

    //Special case: Second column
    H[1] = (Hprev[1] - m*Hprev[1]/meta + m*Hprev[0]/meta
      - (d0*Hvalue23[1] + d1*Hvalue53[1])*expected_N5*Hprev[1]/Eint
      -  w0*Hvalue23[1]                  *expected_N5*Hprev[1]/Eint
      +  w1*Hvalue[1]                                *Hprev[1]/Eint
      + (d0*Hvalue23[2] + d1*Hvalue53[2])*expected_N5*Hprev[2]/Eint
    );

    //Special case: last column
    H[Hlen-1] = (
          Hprev[Hlen-1]
      + m*Hprev[Hlen-2]/meta
      +  w0*Hvalue23[Hlen-2]                        * expected_N5*Hprev[Hlen-2]/Eint
      -  w1*Hvalue[Hlen-2]                                       *Hprev[Hlen-2]/Eint
      - (d0*Hvalue23[Hlen-1] + d1*Hvalue53[Hlen-1]) * expected_N5*Hprev[Hlen-1]/Eint
    );

    //##################
    //Calculate G matrix
    //##################
    const auto Glen = G.size();

    const auto Gvalue13 = VecLogPower(Gvalue,1./3);
    const auto Gvalue43 = VecPower(Gvalue,4./3);

    #pragma omp parallel for simd
    for(unsigned int i=2;i<Glen-1;i++){
      G[i] = (
             Gprev[i  ] + (
        - m*(Gprev[i  ] - Gprev[i-1])   
        +    Gprev[i-1]*(     b0*expected_E1                 ) * Gvalue43[i-1]*Gvalue13[i-1]
        -    Gprev[i  ]*((b0+d0)*expected_E1 + d1*expected_E2) * Gvalue43[i  ]*Gvalue13[i  ]
        +    Gprev[i+1]*(     d0*expected_E1 + d1*expected_E2) * Gvalue43[i+1]*Gvalue13[i+1]
        )/Nint
      );
    }

    // #######################
    // #G matrix special cases
    // #######################

    //Outer columns are special cases: First column
    G[0] = (
          Gprev[0]
      - m*Gprev[0]/Nint
      +   Gprev[1]*(     d0*expected_E1 + d1*expected_E2) * Gvalue43[ 1]*Gvalue13[ 1]/Nint
    );

    //Special case: Second column
    G[1] = (
           Gprev[1]
      - m*(Gprev[1]-Gprev[0])/Nint
      -    Gprev[ 1]*((b0+d0)*expected_E1 + d1*expected_E2) * Gvalue43[ 1]*Gvalue13[ 1]/Nint
      +    Gprev[ 2]*(     d0*expected_E1 + d1*expected_E2) * Gvalue43[ 2]*Gvalue13[ 2]/Nint
    );

    //Special case: last column
    G[Glen-1] = (
          Gprev[Glen-1] +
        m*Gprev[Glen-2]/Nint
      +   Gprev[Glen-2]*(b0*expected_E1)                  *Gvalue13[Glen-2]*Gvalue43[Glen-2]/Nint
      -   Gprev[Glen-1]*(d0*expected_E1 + d1*expected_E2) *Gvalue13[Glen-1]*Gvalue43[Glen-1]/Nint
    );

    // ###################
    // #Calculate F matrix
    // ###################

    const auto Flen     = Fvalue.size();
    const auto Fvalue43 = VecPower(Fvalue,4./3);

    #pragma omp parallel for simd
    for(unsigned int i=1;i<Flen-1;i++){
      f[i] = (
          fprev[i  ]
        + fprev[i-1]*lam0*Fvalue[i-1]
        + fprev[i-1]*m*(1-Fvalue[i-1]/Smeta)
        - fprev[i  ]*lam0*Fvalue[i]
        - fprev[i  ]*m*(1-Fvalue[i]/Smeta)
        - fprev[i  ]*Fvalue43[i]*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
        + fprev[i+1]*Fvalue43[i+1]*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
      );
    }

    // #######################
    // #F matrix special cases
    // #######################

    //Outer columns are special cases: First column
    f[0] = (
        fprev[0]
      - fprev[0]*m*(1-Fvalue[0 ]/Smeta)
      + fprev[ 1]*Fvalue43[ 1]*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
    );

    //Special case: last column
    f[Flen-1] = (
      fprev[Flen-1]
      + fprev[Flen-2]*m*(1-Fvalue[Flen-2]/Smeta)
      - fprev[Flen-1]*Fvalue43[Flen-1]*(d0*expected_E1*expected_N2 + d1*expected_E2*expected_N2)
      + fprev[Flen-2]*lam0*Fvalue[Flen-2]
    );

    // ####################
    // #Check Normalization
    // ####################


    // f[t][f[t]<0] = 0
    // G[t][G[t]<0] = 0
    // H[t][H[t]<0] = 0

    fprev = f;
    Gprev = G;
    Hprev = H;
  }

  void run() {
    MakeSavePoint(0);
    for(int t=1;t<MAX_TIMESTEP;t++){
      if(t%100==0)
        std::cerr<<"p t = "<<t<<std::endl;
      step(t);
      GetNormalization(t);
      //Calculate <E>, <N>, <S>
      GetStateAverage(t);
      MakeSavePoint(t);
    }    
  }
};



int main(){
  DynaSolver ds;
  ds.printConfig();
  ds.run();
  ds.saveAll("/z/saved");
}
