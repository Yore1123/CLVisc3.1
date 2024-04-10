#ifndef __CL_SPEC__
#define __CL_SPEC__

#define __CL_ENABLE_EXCEPTIONS

#define USE_DEVICE_GPU

#include <cl.hpp>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <chrono>
#include <random>
#include <map>
#include <iomanip>



#define NBlocks 256
#define BSZ 64
#define DUMPDATA_FROM_GPU true



#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define USE_SINGLE_PRECISION

#ifdef USE_SINGLE_PRECISION
 typedef cl_float cl_real;   /*!< typedef cl_float to cl_real for easier switch from double to float */
 typedef cl_float2 cl_real2;   /*!< typedef cl_float to cl_real for easier switch from double to float */
 typedef cl_float4 cl_real4;
 typedef cl_float3 cl_real3;
 typedef cl_float8 cl_real8;

#else
 typedef cl_double cl_real;
 typedef cl_double2 cl_real2;
 typedef cl_double4 cl_real4;
 typedef cl_double3 cl_real3;
 typedef cl_double8 cl_real8;
#endif



typedef struct {
    cl_int  pidR;
    cl_real branch;
    cl_int  numpart;
    cl_real part[5];
}CDecay;

struct CParticle{
    int monval;
    std::string  name;
    cl_real mass;
    cl_real width; 
    cl_real	gspin;      /* spin degeneracy */
    cl_real	baryon;     /* baryon number   */
    cl_int	strange;
    cl_int	charm;
    cl_int	bottom;
    cl_int	gisospin;  /* isospin degeneracy */
    cl_real	charge;    
    cl_int	decays;    /* amount of decays listed for this resonance */
    cl_int	stable;     /* defines whether this particle is considered as stable */
    
    /// If antibaryon_spec_exists == 1, just copy, don't repeat calculation
    cl_int  antibaryon_spec_exists;
};



struct Particle{
    // for output
    cl_double4 pos;
    cl_double4 mon;
    cl_double mass;
    cl_int pdgcode;
    cl_int charge;
    
};



class Spec
{

    public:

    Spec(const std::string & pathin, int decay_on,int gpu_id,std::string & EOS_TYPE, int nsampling,const std::string & model,int vorticity_on);

    ~Spec();

    void initializeCL();

    cl::Context CreateContext(const cl_int& device_type);
    //cl::Context CreateContext();
    void AddProgram(const char * fname);

    void BuildPrograms(int program_id, const char* compile_options); 
    void ReadMuB(const std::string & muB_datafile);
 
    void ReadHyperSF(const std::string & sf_datafile);
    void Readtxyz(const std::string & sf_txyzfile);
    void ReadNmutp(const std::string & Nmutp_datafile);
    void ReadParticles(const std::string & particle_data_table);
    void ReadPimnSF(const std::string & piFile);
    void ReadQbSF(const std::string & qbFile);
    void Readdeltaf(const std::string & deltafFile);
    void SetPathOut(const std::string & path);
    void Readvorticity1(const std::string & vorticityDataFile1);
    void Readvorticity2(const std::string & vorticityDataFile2);
    void Readvorticity3(const std::string & vorticityDataFile3);
    void Readvorticity4(const std::string & vorticityDataFile4);
    void Readvorticity5(const std::string & vorticityDataFile5);

    void Sample_particle();
    bool neglect_partices(int pidid);
    


    
    
    cl_real Edfrz;
    
    cl_int TLENGTH;
    cl_int MULENGTH;
    
    cl_real T0;
    cl_real MU0;
    cl_real MU_STEP;
    cl_real T_STEP;
    std::map<cl_int, cl_real> muB;


    std::vector<cl_real8> h_SF; 
    std::vector<cl_real4> h_nmtp; 
    std::vector<cl_real4> h_txyz;
    std::vector<cl_real> h_pi;
    std::vector<cl_real> h_qb;

    std::vector<cl_real> h_omega_th;
    std::vector<cl_real> h_omega_shear1;
    std::vector<cl_real> h_omega_shear2;
    std::vector<cl_real> h_omega_accT;
    std::vector<cl_real> h_omega_chemical;


    std::vector<cl_real4> h_spin_th;
    std::vector<cl_real4> h_spin_shear;
    std::vector<cl_real4> h_spin_accT;
    std::vector<cl_real4> h_spin_chemical;

    std::vector<cl_real> h_dntot;
    std::vector<cl_int2> h_poisson;

    // std::vector<cl_real> h_ptc;
    std::vector<cl_real4> h_lptc;
    std::vector<cl_real4> h_hptc;
    std::vector<cl_real8> h_ptc;
    std::vector<cl_ulong> h_seed;
    std::vector<cl_real> h_deltaf_qmu; 
    cl_int h_Nhptc;
    cl_int h_Nlptc; 
    


    std::vector<CParticle> particles;
    std::vector<CDecay> decay; 
    std::map< cl_int, cl_int > newpid ;
    std::vector<cl_real> h_HadronInfo; 



    private:
    int gpu_id_;
    std::string  DataPath;
    int decay_on_;
    int nsampling_;
    std::string  model_;
    int vorticity_on_;
    std::string eos_type_;
    bool flag_mu_pce;

    bool quark_level;


    cl_uint   SizeSF;    /**< Size = h_SF.size() */
    cl_uint   SizePID;    /**< Size = h_SF.size() */
    

    cl::Context context;
    std::vector<cl::Device> devices;
    std::vector<cl::Program> programs;
    cl::CommandQueue queue;
    
    cl::Kernel kernel_sample;
    cl::Kernel sample_heavyhadron;
    cl::Kernel sample_lighthadron;
    cl::Kernel classify_ptc;
    cl::Kernel get_poisson;
    cl::Kernel sample_vorticity;
    cl::Kernel get_dntot;



    cl::Buffer d_HadronInfo;
    cl::Buffer d_SF;
    cl::Buffer d_pi;
    
    cl::Buffer d_omega_th;
    cl::Buffer d_omega_shear1;
    cl::Buffer d_omega_shear2;
    cl::Buffer d_omega_accT;
    cl::Buffer d_omega_chemical;

    cl::Buffer d_nmtp;
    cl::Buffer d_txyz;
    cl::Buffer d_qb;
    cl::Buffer d_seed;
    cl::Buffer d_lptc;
    cl::Buffer d_hptc;
    cl::Buffer d_Nhptc;
    cl::Buffer d_Nlptc;
    cl::Buffer d_ptc;
    cl::Buffer d_dntot;
    cl::Buffer d_poisson;
    cl::Buffer d_deltaf_qmu;
    

    cl::Buffer d_spin_th;
    cl::Buffer d_spin_shear;
    cl::Buffer d_spin_accT;
    cl::Buffer d_spin_chemical;




};



#endif
