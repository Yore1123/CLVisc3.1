#ifndef HYOERSF_H
#define HYOERSF_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <iomanip>
#include "cornelius.h"
#include "H5Cpp.h"
#include "time.h"
using namespace H5;


struct HpsfPara{

    double dt;
    double dx;
    double dy;
    double dz;
    int NX;
    int NY;
    int NZ;
    int ntskip; 
    int nxskip; 
    int nyskip;
    int nzskip;
    double Edfrz;
    bool Tfrz_on;
    double Tfrz;
    

};


class Hypersf{

    public:

        Hypersf(const std::string& pathin, double old_time, double new_time, int header,int corona,int vorticity);
        ~Hypersf();
        void readbulkinfo(int flag_file);
        void get_hypersf();
        void readsurfaceinfo();
        void equal_tau_corona();
        string findstr(const string& file_str, const string& find_str);
        int idn(int i, int j, int k, int m, int l);
        double Intp4D(double*** oldcell,double*** newcell,double* ratio);
        //void writehypersf();
       
    private:
        H5std_string FILE_NAME[2];
        H5std_string *DATASET_NAME;
        double ***bulkinfo;
        hsize_t ***dims;
        string DataPath;
        int viscous_on_;
        int num_of_dataset;    
        HpsfPara Para;
        double NewTime;
        double OldTime;
        int Header;
        bool flag_vorticity;
        int inx_nmtp;
        int inx_pimn;
        int inx_qb;
	int inx_bulkpr;
	int inx_deltaf;
        int inx_omega;
        int inx_omega_shear1;
        int inx_omega_shear2;
        int inx_omega_accT;
        int inx_omega_chemical;


        ofstream fqb;
        ofstream fpimn;
        ofstream fbulkpr;
        ofstream fnbmutp;
        ofstream fsurface;
        ofstream ftxyz;


	ofstream fomega;
        ofstream fomega_shear1;
        ofstream fomega_shear2;
        ofstream fomega_accT;
        ofstream fomega_chemical;




	
    
};


#endif


