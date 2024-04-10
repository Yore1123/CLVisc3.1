#include "hypersf.h"

Hypersf::Hypersf(const std::string& pathin,  double old_time, double new_time, int header,int corona,int vorticity){
    DataPath = pathin;

    NewTime = new_time;
    OldTime = old_time;
    Header = header;
    
    readsurfaceinfo();
   
    std::stringstream bulk_prev;
    std::stringstream bulk_curr;
    
    if (corona == 1){
       bulk_prev << DataPath << "/bulk_prev.h5";
       bulk_curr << DataPath << "/bulk_prev.h5"; 
    }
    else{
    bulk_prev << DataPath << "/bulk_prev.h5";
    bulk_curr << DataPath << "/bulk_curr.h5"; 
    }
    
 
    FILE_NAME[0] = bulk_prev.str();
    FILE_NAME[1] = bulk_curr.str();


    if (vorticity == 1)
    {

           num_of_dataset=10;
           DATASET_NAME = new H5std_string[num_of_dataset];
           inx_pimn = 1;
           inx_nmtp = 2;
           inx_qb = 3;
           inx_bulkpr = 4;
           inx_omega = 5;
           inx_omega_shear1 = 6;
           inx_omega_shear2 = 7;
           inx_omega_accT = 8;
           inx_omega_chemical = 9;
           DATASET_NAME[0] = "ev";
           DATASET_NAME[inx_pimn] = "pimn";
           DATASET_NAME[inx_nmtp] = "nbmutp";
           DATASET_NAME[inx_qb] = "qb";
           DATASET_NAME[inx_bulkpr] = "bulkpr";
           DATASET_NAME[inx_omega] = "omega";
           DATASET_NAME[inx_omega_shear1] = "omega_shear1";
           DATASET_NAME[inx_omega_shear2] = "omega_shear2";
           DATASET_NAME[inx_omega_accT] = "omega_accT";
           DATASET_NAME[inx_omega_chemical] = "omega_chemical";
           flag_vorticity = true;

    }
    else{

           num_of_dataset=5;
           DATASET_NAME = new H5std_string[num_of_dataset];

           inx_pimn = 1;
           inx_nmtp = 2;
           inx_qb = 3;
           inx_bulkpr = 4;
           DATASET_NAME[0] = "ev";
           DATASET_NAME[inx_pimn] = "pimn";
           DATASET_NAME[inx_nmtp] = "nbmutp";
           DATASET_NAME[inx_qb] = "qb";
           DATASET_NAME[inx_bulkpr] = "bulkpr";
           flag_vorticity = false;

    }
         
         
    

    
    clock_t t;
    t = clock();
    bulkinfo = new double**[2];
    dims = new hsize_t**[2];
    readbulkinfo(0);
    readbulkinfo(1);
    t = clock() - t;
   


    std::stringstream ssfile_surface;
    std::stringstream ssfile_txyz;
    std::stringstream ssfile_qb;
    std::stringstream ssfile_nbmutp;
    std::stringstream ssfile_pimn;
    std::stringstream ssfile_bulkpr;
    
    ssfile_surface<<DataPath<<"/hypersf.dat";
    ssfile_txyz<<DataPath<<"/sf_txyz.dat";
    ssfile_qb<<DataPath<<"/qbmusf.dat";
    ssfile_pimn<<DataPath<<"/pimnsf.dat";
    ssfile_bulkpr<<DataPath<<"/bulkprsf.dat";
    ssfile_nbmutp<<DataPath<<"/sf_nbmutp.dat";
    
    fqb.open(ssfile_qb.str().c_str(),ios::app);
    fpimn.open(ssfile_pimn.str().c_str(),ios::app);
    fbulkpr.open(ssfile_bulkpr.str().c_str(),ios::app);
    fnbmutp.open(ssfile_nbmutp.str().c_str(),ios::app);
    fsurface.open(ssfile_surface.str().c_str(),ios::app);
    ftxyz.open(ssfile_txyz.str().c_str(),ios::app);
    
    if (flag_vorticity) {

    std::stringstream ssfile_omega;
    std::stringstream ssfile_omega_shear1;
    std::stringstream ssfile_omega_shear2;
    std::stringstream ssfile_omega_accT;
    std::stringstream ssfile_omega_chemical;
    
    ssfile_omega<<DataPath<<"/omegamu_sf.dat";
    ssfile_omega_shear1<<DataPath<<"/omegamu_shear1_sf.dat";
    ssfile_omega_shear2<<DataPath<<"/omegamu_shear2_sf.dat";
    ssfile_omega_accT<<DataPath<<"/omegamu_accT_sf.dat";
    ssfile_omega_chemical<<DataPath<<"/omegamu_chemical_sf.dat";
    
    
    fomega.open(ssfile_omega.str().c_str(),ios::app);
    fomega_shear1.open(ssfile_omega_shear1.str().c_str(),ios::app);
    fomega_shear2.open(ssfile_omega_shear2.str().c_str(),ios::app);
    fomega_accT.open(ssfile_omega_accT.str().c_str(),ios::app);
    fomega_chemical.open(ssfile_omega_chemical.str().c_str(),ios::app);
    
    }
    
    







    if(Header == 0){
        if (Para.Tfrz_on  )
        {

            fsurface<<"# Tfrz="<<Para.Tfrz<<"; other rows: dS0, dS1, dS2, dS3, vx, vy, veta, etas"<<std::endl;
        }
        else{
            fsurface<<"# efrz="<<Para.Edfrz<<"; other rows: dS0, dS1, dS2, dS3, vx, vy, veta, etas"<<std::endl;
        }
        fqb<<"# qb0 qb1 qb2 qb3"<<std::endl;
        fpimn<<"#  pi00 01 02 03 11 12 13 22 23 33"<<std::endl;
        fbulkpr << "# bulk pressure" <<std::endl;
        fnbmutp<<"# nb, mu, T, Pr of hypersf elements"<<std::endl;
        ftxyz<<"# (t, x, y, z) the time-space coordinates of hypersf elements"<<std::endl;
        if (flag_vorticity) {
        
	    fomega<<"# omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
	    fomega_shear1<<"# omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
	    fomega_shear2<<"# omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
	    fomega_accT<<"# omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
	    fomega_chemical<<"# omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
	    
        }
    }



    if(corona == 1)
    {
       equal_tau_corona();
    }

    

}

Hypersf::~Hypersf(){
    for (int step = 0 ; step < 2 ; step++){
       for(int i = 0; i < num_of_dataset ; i++){
        delete [] bulkinfo[step][i];
        delete [] dims[step][i];

    }

    }


    for (int step = 0 ; step < 2 ; step++){
        delete [] bulkinfo[step];
        delete [] dims[step];
    }


    delete [] bulkinfo;
    delete [] dims;
    delete [] DATASET_NAME;


        
    fsurface.close();          
    fqb.close();
    fbulkpr.close();
    fpimn.close();
    fnbmutp.close();
    ftxyz.close();
    fomega.close();
    fomega_shear1.close();
    fomega_shear2.close();
    fomega_accT.close();
    fomega_chemical.close();

}

void Hypersf::readbulkinfo(int flag_file){

   H5File file(FILE_NAME[flag_file], H5F_ACC_RDONLY);
   DataSet dataset[num_of_dataset];
   DataSpace filespace[num_of_dataset];
   dims[flag_file] = new hsize_t*[num_of_dataset];
   bulkinfo[flag_file] = new double* [num_of_dataset];

   for(int i = 0 ; i < num_of_dataset; i ++){
       
       dataset[i] = file.openDataSet(DATASET_NAME[i]);
       filespace[i] = dataset[i].getSpace();
       int rank = filespace[i].getSimpleExtentNdims();
       dims[flag_file][i] = new hsize_t [num_of_dataset];
       rank = filespace[i].getSimpleExtentDims(dims[flag_file][i]);
       DataSpace mspace(rank,dims[flag_file][i]);
       bulkinfo[flag_file][i] = new double [dims[flag_file][i][0]*dims[flag_file][i][1]];
       dataset[i].read(bulkinfo[flag_file][i],PredType::NATIVE_DOUBLE, mspace,filespace[i]);
       
    
   }
   
   
}

string Hypersf::findstr(const string& file_str, const string& target_str){
    string::size_type position;
    string temp1;
    string temp;
    position = file_str.find(target_str);
    string substr1 = file_str.substr(position);
    stringstream buff2(substr1);
    buff2 >>temp1 >> temp;

    while(1){

    if (temp == "=" && temp1 == target_str){
        buff2 >> temp;
        break;
    }
    else{
        //position += target_str.length();
        //substr1 = file_str.substr(position);

        substr1 = substr1.substr( target_str.length() );
        position = substr1.find(target_str);
        substr1 = substr1.substr(position); 
        buff2.str("");
        buff2 << substr1;
        buff2 >> temp1 >> temp;
    }
    } 
    return temp;

}

void Hypersf::readsurfaceinfo(){
    std::stringstream hydro_info;
    hydro_info << DataPath << "/hydro.info";
   
    string file_str;
    ifstream fin;
    fin.open(hydro_info.str().c_str(),ios::in);
    stringstream file_buf;
    file_buf << fin.rdbuf();
    file_str = file_buf.str();
    
    
    string target_str = "dt";
    Para.dt = std::stod( findstr(file_str,target_str) );

    target_str = "dx";
    Para.dx = std::stod( findstr(file_str,target_str) );
    
    target_str = "dy";
    Para.dy = std::stod( findstr(file_str,target_str) );

    target_str = "dz";
    Para.dz = std::stod( findstr(file_str,target_str) );


    target_str = "nx";
    Para.NX = std::stoi(findstr(file_str,target_str));

    target_str = "ny";
    Para.NY = std::stoi(findstr(file_str,target_str));

    target_str = "nz";
    Para.NZ = std::stoi(findstr(file_str,target_str));


    target_str = "ntskip";
    Para.ntskip = std::stoi(findstr(file_str,target_str));

    target_str = "nxskip";
    Para.nxskip = std::stoi(findstr(file_str,target_str));

    target_str = "nyskip";
    Para.nyskip = std::stoi(findstr(file_str,target_str));

    target_str = "nzskip";
    Para.nzskip = std::stoi(findstr(file_str,target_str));
     
    target_str = "edfrz";
    Para.Edfrz = std::stod( findstr(file_str,target_str) );
    target_str = "tfrz";
    Para.Tfrz = std::stod( findstr(file_str,target_str) );
    target_str = "tfrz_on";
    
   
    std::string opt(findstr(file_str,target_str));
    if( opt == "1" || opt == "True" || opt == "true" || opt == "yes" || opt == "Yes" || opt == "YES" ){
        Para.Tfrz_on = true;
    }
    else{
        Para.Tfrz_on = false;
    }

    

    fin.close();

}


double Hypersf::Intp4D(double*** oldcell, double*** newcell,double* ratio){
    double intp_value = 0.0;
     intp_value = (1.0 - ratio[0])*(1.0 - ratio[1])*(1.0 - ratio[2])*(1.0 - ratio[3])*oldcell[0][0][0]
                 + (1.0 - ratio[0])*( ratio[1] )*(1.0 - ratio[2])*(1.0 - ratio[3])*oldcell[1][0][0]
                 + (1.0 - ratio[0])*(1.0 - ratio[1])*( ratio[2] )*(1.0 - ratio[3])*oldcell[0][1][0]
                 + (1.0 - ratio[0])*(1.0 - ratio[1])*(1.0 - ratio[2])*( ratio[3] )*oldcell[0][0][1]
                 + (1.0 - ratio[0])*(ratio[1])*( ratio[2] )*(1.0 - ratio[3])*oldcell[1][1][0]
                 + (1.0 - ratio[0])*(1.0 - ratio[1])*(ratio[2])*(ratio[3])*oldcell[0][1][1]
                 + (1.0 - ratio[0])*(ratio[1])*(1.0 - ratio[2])*(ratio[3])*oldcell[1][0][1]
                 + (1.0 - ratio[0])*(ratio[1])*(ratio[2])*(ratio[3])*oldcell[1][1][1]

               + (ratio[0])*(1.0 - ratio[1])*(1.0 - ratio[2])*(1.0 - ratio[3])*newcell[0][0][0]
               + (ratio[0])*( ratio[1] )*(1.0 - ratio[2])*(1.0 - ratio[3])*newcell[1][0][0]
               + (ratio[0])*(1.0 - ratio[1])*( ratio[2] )*(1.0 - ratio[3])*newcell[0][1][0]
               + (ratio[0])*(1.0 - ratio[1])*(1.0 - ratio[2])*( ratio[3] )*newcell[0][0][1]
               + (ratio[0])*(ratio[1])*( ratio[2] )*(1.0 - ratio[3])*newcell[1][1][0]
               + (ratio[0])*(1.0 - ratio[1])*(ratio[2])*(ratio[3])*newcell[0][1][1]
               + (ratio[0])*(ratio[1])*(1.0 - ratio[2])*(ratio[3])*newcell[1][0][1]
               + (ratio[0])*(ratio[1])*(ratio[2])*(ratio[3])*newcell[1][1][1];
    //intp_value = (1.0 - ratio[0])*(1.0 - ratio[1])*(1.0 - ratio[2])*( ratio[3] )*oldcell[0][0][1];
    
    return intp_value;

}


//
int Hypersf::idn(int i, int j, int k,int m,int l){
    int index = i*Para.NY*Para.NZ*l + j*Para.NZ*l + k*l + m;
    return index;
}

void  Hypersf::get_hypersf(){
    Cornelius cornelius0;

    const int dim = 4;
    int ntskip = Para.ntskip;
    int nxskip = Para.nxskip;
    int nyskip = Para.nyskip;
    int nzskip = Para.nzskip;

    
    int NX = Para.NX;
    int NY = Para.NY;
    int NZ = Para.NZ;

    double dt = Para.dt;
    double dx = Para.dx;
    double dy = Para.dy;
    double dz = Para.dz;

    double DTAU = ntskip*dt;
    double DX =   nxskip*dx;
    double DY =   nyskip*dy;
    double DZ =   nzskip*dz;

    double DSTEP[dim] = {DTAU, DX, DY, DZ};
    cornelius0.init(dim,Para.Edfrz,DSTEP);

    double ****cube = new double*** [2];
    for(int i = 0 ; i < 2 ; i++ ){
        cube[i] = new double** [2];
        for (int j = 0; j < 2; j++){
            cube[i][j] = new double *[2];
            for (int k = 0; k < 2; k++){
                cube[i][j][k] = new double[2];
                for (int m = 0 ; m < 2;m++)
                    cube[i][j][k][m] = 0.0;   

            }

        }

    }



    std::stringstream ssfile_surface;
    std::stringstream ssfile_txyz;
    std::stringstream ssfile_qb;
    std::stringstream ssfile_nbmutp;
    std::stringstream ssfile_pimn;
    std::stringstream ssfile_bulkpr;

    std::stringstream ssfile_omega;
    std::stringstream ssfile_omega_shear1;
    std::stringstream ssfile_omega_shear2;
    std::stringstream ssfile_omega_accT;
    std::stringstream ssfile_omega_chemical;

    ssfile_surface<<DataPath<<"/hypersf.dat";
    ssfile_txyz<<DataPath<<"/sf_txyz.dat";
    ssfile_qb<<DataPath<<"/qbmusf.dat";
    ssfile_pimn<<DataPath<<"/pimnsf.dat";
    ssfile_bulkpr<<DataPath<<"/bulkprsf.dat";
    ssfile_nbmutp<<DataPath<<"/sf_nbmutp.dat";


    ssfile_omega<<DataPath<<"/omegamu_sf.dat";
    ssfile_omega_shear1<<DataPath<<"/omegamu_shear1_sf.dat";
    ssfile_omega_shear2<<DataPath<<"/omegamu_shear2_sf.dat";
    ssfile_omega_accT<<DataPath<<"/omegamu_accT_sf.dat";
    ssfile_omega_chemical<<DataPath<<"/omegamu_chemical_sf.dat";
        
    ofstream fqb(ssfile_qb.str().c_str(),ios::app);
    ofstream fpimn(ssfile_pimn.str().c_str(),ios::app);
    ofstream fbulkpr(ssfile_bulkpr.str().c_str(),ios::app);
    ofstream fnbmutp(ssfile_nbmutp.str().c_str(),ios::app);
    ofstream fsurface(ssfile_surface.str().c_str(),ios::app);
    ofstream ftxyz(ssfile_txyz.str().c_str(),ios::app);

    ofstream fomega(ssfile_omega.str().c_str(),ios::app);
    ofstream fomega_shear1(ssfile_omega_shear1.str().c_str(),ios::app);
    ofstream fomega_shear2(ssfile_omega_shear2.str().c_str(),ios::app);
    ofstream fomega_accT(ssfile_omega_accT.str().c_str(),ios::app);
    ofstream fomega_chemical(ssfile_omega_chemical.str().c_str(),ios::app);
    
    
    
    //if(Header == 0){
    //    if (Para.Tfrz_on  )
    //    {    
    //
    //        fsurface<<"# Tfrz="<<Para.Tfrz<<"; other rows: dS0, dS1, dS2, dS3, vx, vy, veta, etas"<<std::endl;
    //    }
    //    else{
    //        fsurface<<"# efrz="<<Para.Edfrz<<"; other rows: dS0, dS1, dS2, dS3, vx, vy, veta, etas"<<std::endl;
    //    }
    //    fqb<<"# qb0 qb1 qb2 qb3"<<std::endl;
    //    fpimn<<"# one_o_2TsqrEplusP="<< bulkinfo[0][inx_deltaf][0] <<" pi00 01 02 03 11 12 13 22 23 33"<<std::endl;
    //    fbulkpr << "# bulk pressure" <<std::endl;
    //    fnbmutp<<"# nb, mu, T, Pr of hypersf elements"<<std::endl;
    //    fomega<<"'omega^{01, 02, 03, 12, 13, 23}'"<<std::endl;
    //    fomega_shear1<<"'omega^{01, 02, 03, 12, 13, 23}'"<<std::endl;
    //    fomega_shear2<<"'omega^{01, 02, 03, 12, 13, 23}'"<<std::endl;
    //    fomega_accT<<"'omega^{01, 02, 03, 12, 13, 23}'"<<std::endl;
    //    fomega_chemical<<"'omega^{01, 02, 03, 12, 13, 23}'"<<std::endl;
    //    ftxyz<<"(t, x, y, z) the time-space coordinates of hypersf elements"<<std::endl;
    //}
    
    //std::cout<< Header << std::endl;
    int Nintersection = 0;
    for(int i = 0; i < NX - nxskip; i += nxskip){
       double xx = i*dx - NX/2*dx;
       for (int j = 0; j < NY - nyskip; j += nyskip){
           double yy = j*dy - NY/2*dy;
           for(int k = 0 ; k < NZ - nzskip; k += nzskip){
               double zz = k*dz - NZ/2*dz;

              
                cube[0][0][0][0] = bulkinfo[0][0][idn(i,j,k,0,4)];
                cube[0][1][1][1] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k+nzskip,0,4)];
                cube[0][1][0][0] = bulkinfo[0][0][idn(i+nxskip,j,k,0,4)];
                cube[0][0][1][0] = bulkinfo[0][0][idn(i,j+nyskip,k,0,4)];
                cube[0][0][0][1] = bulkinfo[0][0][idn(i,j,k+nzskip,0,4)];
                cube[0][1][1][0] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k,0,4)];
                cube[0][0][1][1] = bulkinfo[0][0][idn(i,j+nyskip,k+nzskip,0,4)];
                cube[0][1][0][1] = bulkinfo[0][0][idn(i+nxskip,j,k+nzskip,0,4)];
                 

                cube[1][0][0][0] = bulkinfo[1][0][idn(i,j,k,0,4)];
                cube[1][1][1][1] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k+nzskip,0,4)];
                cube[1][1][0][0] = bulkinfo[1][0][idn(i+nxskip,j,k,0,4)];
                cube[1][0][1][0] = bulkinfo[1][0][idn(i,j+nyskip,k,0,4)];
                cube[1][0][0][1] = bulkinfo[1][0][idn(i,j,k+nzskip,0,4)];
                cube[1][1][1][0] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k,0,4)];
                cube[1][0][1][1] = bulkinfo[1][0][idn(i,j+nyskip,k+nzskip,0,4)];
                cube[1][1][0][1] = bulkinfo[1][0][idn(i+nxskip,j,k+nzskip,0,4)];
                
                

                int intersection = 1;
                
                if( (cube[1][1][1][1]-Para.Edfrz)*(cube[0][0][0][0]-Para.Edfrz)> 0.0
                &&  (cube[1][0][1][1]-Para.Edfrz)*(cube[0][1][0][0]-Para.Edfrz)> 0.0
                &&  (cube[1][1][0][1]-Para.Edfrz)*(cube[0][0][1][0]-Para.Edfrz)> 0.0
                &&  (cube[1][1][1][0]-Para.Edfrz)*(cube[0][0][0][1]-Para.Edfrz)> 0.0
                &&  (cube[1][0][0][1]-Para.Edfrz)*(cube[0][1][1][0]-Para.Edfrz)> 0.0
                &&  (cube[1][1][0][0]-Para.Edfrz)*(cube[0][0][1][1]-Para.Edfrz)> 0.0
                &&  (cube[1][0][1][0]-Para.Edfrz)*(cube[0][1][0][1]-Para.Edfrz)> 0.0
                &&  (cube[1][0][0][0]-Para.Edfrz)*(cube[0][1][1][1]-Para.Edfrz)> 0.0){
                    intersection = 0;
                }
           
                 if(intersection == 0 ) continue;
                 cornelius0.find_surface_4d(cube);
                 
                 for(int ID = 0; ID < cornelius0.get_Nelements();ID++){
                    double mass_center[4];
                    double dsigma[4];
                    double ratio[4];
                    mass_center[0] = cornelius0.get_centroid_elem(ID,0);
                    mass_center[1] = cornelius0.get_centroid_elem(ID,1);
                    mass_center[2] = cornelius0.get_centroid_elem(ID,2);
                    mass_center[3] = cornelius0.get_centroid_elem(ID,3);
                    
                    dsigma[0] = cornelius0.get_normal_elem(ID,0);
                    dsigma[1] = cornelius0.get_normal_elem(ID,1);
                    dsigma[2] = cornelius0.get_normal_elem(ID,2);
                    dsigma[3] = cornelius0.get_normal_elem(ID,3);
                    
                    //std::cout<<dsigma
                    ratio[0] = mass_center[0]/DTAU;
                    ratio[1] = mass_center[1]/DX;
                    ratio[2] = mass_center[2]/DY;
                    ratio[3] = mass_center[3]/DZ;
                    double cell_center[4];
                    cell_center[0] = OldTime + mass_center[0];
                    cell_center[1] = xx + mass_center[1];
                    cell_center[2] = yy + mass_center[2];
                    cell_center[3] = zz + mass_center[3];

                    double cell_position[4];
                    cell_position[0] = cell_center[0]*cosh(cell_center[3]);
                    cell_position[1] = cell_center[1];
                    cell_position[2] = cell_center[2];
                    cell_position[3] = cell_center[0]*sinh(cell_center[3]);

                    //vx
                    cube[0][0][0][0] = bulkinfo[0][0][idn(i,j,k,1,4)];
                    cube[0][1][1][1] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k+nzskip,1,4)];
                    cube[0][1][0][0] = bulkinfo[0][0][idn(i+nxskip,j,k,1,4)];
                    cube[0][0][1][0] = bulkinfo[0][0][idn(i,j+nyskip,k,1,4)];
                    cube[0][0][0][1] = bulkinfo[0][0][idn(i,j,k+nzskip,1,4)];
                    cube[0][1][1][0] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k,1,4)];
                    cube[0][0][1][1] = bulkinfo[0][0][idn(i,j+nyskip,k+nzskip,1,4)];
                    cube[0][1][0][1] = bulkinfo[0][0][idn(i+nxskip,j,k+nzskip,1,4)];
                        
                    cube[1][0][0][0] = bulkinfo[1][0][idn(i,j,k,1,4)];
                    cube[1][1][1][1] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k+nzskip,1,4)];
                    cube[1][1][0][0] = bulkinfo[1][0][idn(i+nxskip,j,k,1,4)];
                    cube[1][0][1][0] = bulkinfo[1][0][idn(i,j+nyskip,k,1,4)];
                    cube[1][0][0][1] = bulkinfo[1][0][idn(i,j,k+nzskip,1,4)];
                    cube[1][1][1][0] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k,1,4)];
                    cube[1][0][1][1] = bulkinfo[1][0][idn(i,j+nyskip,k+nzskip,1,4)];
                    cube[1][1][0][1] = bulkinfo[1][0][idn(i+nxskip,j,k+nzskip,1,4)];

                    double vx_center = Intp4D(cube[0],cube[1],ratio);
                    

                    //vy
                    cube[0][0][0][0] = bulkinfo[0][0][idn(i,j,k,2,4)];
                    cube[0][1][1][1] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k+nzskip,2,4)];
                    cube[0][1][0][0] = bulkinfo[0][0][idn(i+nxskip,j,k,2,4)];
                    cube[0][0][1][0] = bulkinfo[0][0][idn(i,j+nyskip,k,2,4)];
                    cube[0][0][0][1] = bulkinfo[0][0][idn(i,j,k+nzskip,2,4)];
                    cube[0][1][1][0] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k,2,4)];
                    cube[0][0][1][1] = bulkinfo[0][0][idn(i,j+nyskip,k+nzskip,2,4)];
                    cube[0][1][0][1] = bulkinfo[0][0][idn(i+nxskip,j,k+nzskip,2,4)];
                        
                    cube[1][0][0][0] = bulkinfo[1][0][idn(i,j,k,2,4)];
                    cube[1][1][1][1] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k+nzskip,2,4)];
                    cube[1][1][0][0] = bulkinfo[1][0][idn(i+nxskip,j,k,2,4)];
                    cube[1][0][1][0] = bulkinfo[1][0][idn(i,j+nyskip,k,2,4)];
                    cube[1][0][0][1] = bulkinfo[1][0][idn(i,j,k+nzskip,2,4)];
                    cube[1][1][1][0] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k,2,4)];
                    cube[1][0][1][1] = bulkinfo[1][0][idn(i,j+nyskip,k+nzskip,2,4)];
                    cube[1][1][0][1] = bulkinfo[1][0][idn(i+nxskip,j,k+nzskip,2,4)];

                    double vy_center = Intp4D(cube[0],cube[1],ratio);


                    //vz
                    cube[0][0][0][0] = bulkinfo[0][0][idn(i,j,k,3,4)];
                    cube[0][1][1][1] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k+nzskip,3,4)];
                    cube[0][1][0][0] = bulkinfo[0][0][idn(i+nxskip,j,k,3,4)];
                    cube[0][0][1][0] = bulkinfo[0][0][idn(i,j+nyskip,k,3,4)];
                    cube[0][0][0][1] = bulkinfo[0][0][idn(i,j,k+nzskip,3,4)];
                    cube[0][1][1][0] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k,3,4)];
                    cube[0][0][1][1] = bulkinfo[0][0][idn(i,j+nyskip,k+nzskip,3,4)];
                    cube[0][1][0][1] = bulkinfo[0][0][idn(i+nxskip,j,k+nzskip,3,4)];
                        
                    cube[1][0][0][0] = bulkinfo[1][0][idn(i,j,k,3,4)];
                    cube[1][1][1][1] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k+nzskip,3,4)];
                    cube[1][1][0][0] = bulkinfo[1][0][idn(i+nxskip,j,k,3,4)];
                    cube[1][0][1][0] = bulkinfo[1][0][idn(i,j+nyskip,k,3,4)];
                    cube[1][0][0][1] = bulkinfo[1][0][idn(i,j,k+nzskip,3,4)];
                    cube[1][1][1][0] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k,3,4)];
                    cube[1][0][1][1] = bulkinfo[1][0][idn(i,j+nyskip,k+nzskip,3,4)];
                    cube[1][1][0][1] = bulkinfo[1][0][idn(i+nxskip,j,k+nzskip,3,4)];

                    double vz_center = Intp4D(cube[0],cube[1],ratio);
                    
                    double pi00_center,pi01_center,pi02_center,pi03_center,
                           pi11_center,pi12_center,pi13_center,pi22_center,
                           pi23_center, pi33_center; 
                    
                    double omega_center[6]={0};
                    double omega_shear1_center[16]={0};
                    double omega_shear2_center[4]={0};
                    double omega_accT_center[6]={0};    
                    double omega_chemical_center[6]={0};   
                    //pi00
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,0,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,0,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,0,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,0,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,0,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,0,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,0,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,0,10)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,0,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,0,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,0,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,0,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,0,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,0,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,0,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,0,10)];

                    pi00_center = Intp4D(cube[0],cube[1],ratio);
                                 
                    
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,1,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,1,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,1,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,1,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,1,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,1,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,1,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,1,10)];

                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,1,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,1,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,1,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,1,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,1,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,1,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,1,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,1,10)];

                    pi01_center = Intp4D(cube[0],cube[1],ratio);
                    
                    
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,2,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,2,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,2,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,2,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,2,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,2,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,2,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,2,10)];
            
                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,2,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,2,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,2,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,2,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,2,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,2,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,2,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,2,10)];

                    pi02_center = Intp4D(cube[0],cube[1],ratio);
                    
                    //pi03
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,3,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,3,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,3,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,3,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,3,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,3,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,3,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,3,10)];

                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,3,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,3,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,3,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,3,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,3,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,3,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,3,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,3,10)];

                    pi03_center = Intp4D(cube[0],cube[1],ratio);
                    

                    //pi11
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,4,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,4,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,4,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,4,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,4,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,4,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,4,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,4,10)];

                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,4,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,4,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,4,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,4,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,4,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,4,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,4,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,4,10)];

                    pi11_center = Intp4D(cube[0],cube[1],ratio);
                    
                    
                    //pi12
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,5,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,5,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,5,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,5,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,5,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,5,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,5,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,5,10)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,5,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,5,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,5,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,5,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,5,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,5,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,5,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,5,10)];

                    pi12_center = Intp4D(cube[0],cube[1],ratio);
                    
                    //pi13
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,6,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,6,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,6,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,6,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,6,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,6,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,6,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,6,10)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,6,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,6,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,6,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,6,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,6,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,6,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,6,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,6,10)];

                    pi13_center = Intp4D(cube[0],cube[1],ratio);
                    

                    //pi22
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,7,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,7,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,7,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,7,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,7,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,7,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,7,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,7,10)];
            
                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,7,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,7,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,7,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,7,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,7,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,7,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,7,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,7,10)];

                    pi22_center = Intp4D(cube[0],cube[1],ratio);
                    

                    //pi23
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,8,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,8,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,8,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,8,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,8,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,8,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,8,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,8,10)];
            
                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,8,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,8,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,8,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,8,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,8,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,8,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,8,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,8,10)];

                    pi23_center = Intp4D(cube[0],cube[1],ratio);


                    //pi33
                    cube[0][0][0][0] = bulkinfo[0][inx_pimn][idn(i,j,k,9,10)];
                    cube[0][1][1][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,9,10)];
                    cube[0][1][0][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k,9,10)];
                    cube[0][0][1][0] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k,9,10)];
                    cube[0][0][0][1] = bulkinfo[0][inx_pimn][idn(i,j,k+nzskip,9,10)];
                    cube[0][1][1][0] = bulkinfo[0][inx_pimn][idn(i+nxskip,j+nyskip,k,9,10)];
                    cube[0][0][1][1] = bulkinfo[0][inx_pimn][idn(i,j+nyskip,k+nzskip,9,10)];
                    cube[0][1][0][1] = bulkinfo[0][inx_pimn][idn(i+nxskip,j,k+nzskip,9,10)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_pimn][idn(i,j,k,9,10)];
                    cube[1][1][1][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k+nzskip,9,10)];
                    cube[1][1][0][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k,9,10)];
                    cube[1][0][1][0] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k,9,10)];
                    cube[1][0][0][1] = bulkinfo[1][inx_pimn][idn(i,j,k+nzskip,9,10)];
                    cube[1][1][1][0] = bulkinfo[1][inx_pimn][idn(i+nxskip,j+nyskip,k,9,10)];
                    cube[1][0][1][1] = bulkinfo[1][inx_pimn][idn(i,j+nyskip,k+nzskip,9,10)];
                    cube[1][1][0][1] = bulkinfo[1][inx_pimn][idn(i+nxskip,j,k+nzskip,9,10)];

                    pi33_center = Intp4D(cube[0],cube[1],ratio);

                  

                    if (flag_vorticity){
                        for (int nele = 0; nele<6;nele++){
                            for(int ci = 0; ci <2 ;ci++){
                                for(int cj = 0; cj <2 ;cj++){
                                        for(int ck = 0; ck <2 ;ck++){
                                            int fi = i+ci*nxskip;
                                            int fj = j+cj*nyskip;
                                            int fk = k+ck*nzskip;
                                            cube[0][ci][cj][ck] = bulkinfo[0][inx_omega][idn(fi,fj,fk,nele,6)];
                                            cube[1][ci][cj][ck] = bulkinfo[1][inx_omega][idn(fi,fj,fk,nele,6)];
                                        }

                                }

                            }
                            omega_center[nele] = Intp4D(cube[0],cube[1],ratio);

                        }


                        for (int nele = 0; nele<16;nele++){
                            for(int ci = 0; ci <2 ;ci++){
                                for(int cj = 0; cj <2 ;cj++){
                                        for(int ck = 0; ck <2 ;ck++){
                                            int fi = i+ci*nxskip;
                                            int fj = j+cj*nyskip;
                                            int fk = k+ck*nzskip;
                                            cube[0][ci][cj][ck] = bulkinfo[0][inx_omega_shear1][idn(fi,fj,fk,nele,16)];
                                            cube[1][ci][cj][ck] = bulkinfo[1][inx_omega_shear1][idn(fi,fj,fk,nele,16)];
                                        }

                                }

                            }
                            omega_shear1_center[nele] = Intp4D(cube[0],cube[1],ratio);

                        }

                        for (int nele = 0; nele<4;nele++){
                            for(int ci = 0; ci <2 ;ci++){
                                for(int cj = 0; cj <2 ;cj++){
                                        for(int ck = 0; ck <2 ;ck++){
                                            int fi = i+ci*nxskip;
                                            int fj = j+cj*nyskip;
                                            int fk = k+ck*nzskip;
                                            cube[0][ci][cj][ck] = bulkinfo[0][inx_omega_shear2][idn(fi,fj,fk,nele,4)];
                                            cube[1][ci][cj][ck] = bulkinfo[1][inx_omega_shear2][idn(fi,fj,fk,nele,4)];
                                        }

                                }

                            }
                            omega_shear2_center[nele] = Intp4D(cube[0],cube[1],ratio);

                        }


                        for (int nele = 0; nele<6;nele++){
                            for(int ci = 0; ci <2 ;ci++){
                                for(int cj = 0; cj <2 ;cj++){
                                        for(int ck = 0; ck <2 ;ck++){
                                            int fi = i+ci*nxskip;
                                            int fj = j+cj*nyskip;
                                            int fk = k+ck*nzskip;
                                            cube[0][ci][cj][ck] = bulkinfo[0][inx_omega_accT][idn(fi,fj,fk,nele,6)];
                                            cube[1][ci][cj][ck] = bulkinfo[1][inx_omega_accT][idn(fi,fj,fk,nele,6)];
                                        }

                                }

                            }
                            omega_accT_center[nele] = Intp4D(cube[0],cube[1],ratio);
                        }


                            for (int nele = 0; nele<6;nele++){
                            for(int ci = 0; ci <2 ;ci++){
                                for(int cj = 0; cj <2 ;cj++){
                                        for(int ck = 0; ck <2 ;ck++){
                                            int fi = i+ci*nxskip;
                                            int fj = j+cj*nyskip;
                                            int fk = k+ck*nzskip;
                                            cube[0][ci][cj][ck] = bulkinfo[0][inx_omega_chemical][idn(fi,fj,fk,nele,6)];
                                            cube[1][ci][cj][ck] = bulkinfo[1][inx_omega_chemical][idn(fi,fj,fk,nele,6)];
                                        }

                                }

                            }
                            omega_chemical_center[nele] = Intp4D(cube[0],cube[1],ratio);
                        }

                        
                    }


                    

                    double nb_center,mu_center,T_center,P_center;
                
                    //nb
                    cube[0][0][0][0] = bulkinfo[0][inx_nmtp][idn(i,j,k,0,4)];
                    cube[0][1][1][1] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j+nyskip,k+nzskip,0,4)];
                    cube[0][1][0][0] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j,k,0,4)];
                    cube[0][0][1][0] = bulkinfo[0][inx_nmtp][idn(i,j+nyskip,k,0,4)];
                    cube[0][0][0][1] = bulkinfo[0][inx_nmtp][idn(i,j,k+nzskip,0,4)];
                    cube[0][1][1][0] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j+nyskip,k,0,4)];
                    cube[0][0][1][1] = bulkinfo[0][inx_nmtp][idn(i,j+nyskip,k+nzskip,0,4)];
                    cube[0][1][0][1] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j,k+nzskip,0,4)];
        
                    cube[1][0][0][0] = bulkinfo[1][inx_nmtp][idn(i,j,k,0,4)];
                    cube[1][1][1][1] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j+nyskip,k+nzskip,0,4)];
                    cube[1][1][0][0] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j,k,0,4)];
                    cube[1][0][1][0] = bulkinfo[1][inx_nmtp][idn(i,j+nyskip,k,0,4)];
                    cube[1][0][0][1] = bulkinfo[1][inx_nmtp][idn(i,j,k+nzskip,0,4)];
                    cube[1][1][1][0] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j+nyskip,k,0,4)];
                    cube[1][0][1][1] = bulkinfo[1][inx_nmtp][idn(i,j+nyskip,k+nzskip,0,4)];
                    cube[1][1][0][1] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j,k+nzskip,0,4)];
                    
                  
                   
                    nb_center = Intp4D(cube[0],cube[1],ratio);

                    //mu
                    cube[0][0][0][0] = bulkinfo[0][inx_nmtp][idn(i,j,k,1,4)];
                    cube[0][1][1][1] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j+nyskip,k+nzskip,1,4)];
                    cube[0][1][0][0] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j,k,1,4)];
                    cube[0][0][1][0] = bulkinfo[0][inx_nmtp][idn(i,j+nyskip,k,1,4)];
                    cube[0][0][0][1] = bulkinfo[0][inx_nmtp][idn(i,j,k+nzskip,1,4)];
                    cube[0][1][1][0] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j+nyskip,k,1,4)];
                    cube[0][0][1][1] = bulkinfo[0][inx_nmtp][idn(i,j+nyskip,k+nzskip,1,4)];
                    cube[0][1][0][1] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j,k+nzskip,1,4)];
        
                    cube[1][0][0][0] = bulkinfo[1][inx_nmtp][idn(i,j,k,1,4)];
                    cube[1][1][1][1] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j+nyskip,k+nzskip,1,4)];
                    cube[1][1][0][0] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j,k,1,4)];
                    cube[1][0][1][0] = bulkinfo[1][inx_nmtp][idn(i,j+nyskip,k,1,4)];
                    cube[1][0][0][1] = bulkinfo[1][inx_nmtp][idn(i,j,k+nzskip,1,4)];
                    cube[1][1][1][0] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j+nyskip,k,1,4)];
                    cube[1][0][1][1] = bulkinfo[1][inx_nmtp][idn(i,j+nyskip,k+nzskip,1,4)];
                    cube[1][1][0][1] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j,k+nzskip,1,4)];

                    mu_center = Intp4D(cube[0],cube[1],ratio);
                    

                    //T
                    cube[0][0][0][0] = bulkinfo[0][inx_nmtp][idn(i,j,k,2,4)];
                    cube[0][1][1][1] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j+nyskip,k+nzskip,2,4)];
                    cube[0][1][0][0] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j,k,2,4)];
                    cube[0][0][1][0] = bulkinfo[0][inx_nmtp][idn(i,j+nyskip,k,2,4)];
                    cube[0][0][0][1] = bulkinfo[0][inx_nmtp][idn(i,j,k+nzskip,2,4)];
                    cube[0][1][1][0] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j+nyskip,k,2,4)];
                    cube[0][0][1][1] = bulkinfo[0][inx_nmtp][idn(i,j+nyskip,k+nzskip,2,4)];
                    cube[0][1][0][1] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j,k+nzskip,2,4)];
 
                    cube[1][0][0][0] = bulkinfo[1][inx_nmtp][idn(i,j,k,2,4)];
                    cube[1][1][1][1] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j+nyskip,k+nzskip,2,4)];
                    cube[1][1][0][0] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j,k,2,4)];
                    cube[1][0][1][0] = bulkinfo[1][inx_nmtp][idn(i,j+nyskip,k,2,4)];
                    cube[1][0][0][1] = bulkinfo[1][inx_nmtp][idn(i,j,k+nzskip,2,4)];
                    cube[1][1][1][0] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j+nyskip,k,2,4)];
                    cube[1][0][1][1] = bulkinfo[1][inx_nmtp][idn(i,j+nyskip,k+nzskip,2,4)];
                    cube[1][1][0][1] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j,k+nzskip,2,4)];

                    T_center = Intp4D(cube[0],cube[1],ratio);
                    

                    //P
                    cube[0][0][0][0] = bulkinfo[0][inx_nmtp][idn(i,j,k,3,4)];
                    cube[0][1][1][1] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j+nyskip,k+nzskip,3,4)];
                    cube[0][1][0][0] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j,k,3,4)];
                    cube[0][0][1][0] = bulkinfo[0][inx_nmtp][idn(i,j+nyskip,k,3,4)];
                    cube[0][0][0][1] = bulkinfo[0][inx_nmtp][idn(i,j,k+nzskip,3,4)];
                    cube[0][1][1][0] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j+nyskip,k,3,4)];
                    cube[0][0][1][1] = bulkinfo[0][inx_nmtp][idn(i,j+nyskip,k+nzskip,3,4)];
                    cube[0][1][0][1] = bulkinfo[0][inx_nmtp][idn(i+nxskip,j,k+nzskip,3,4)];
        
                    cube[1][0][0][0] = bulkinfo[1][inx_nmtp][idn(i,j,k,3,4)];
                    cube[1][1][1][1] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j+nyskip,k+nzskip,3,4)];
                    cube[1][1][0][0] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j,k,3,4)];
                    cube[1][0][1][0] = bulkinfo[1][inx_nmtp][idn(i,j+nyskip,k,3,4)];
                    cube[1][0][0][1] = bulkinfo[1][inx_nmtp][idn(i,j,k+nzskip,3,4)];
                    cube[1][1][1][0] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j+nyskip,k,3,4)];
                    cube[1][0][1][1] = bulkinfo[1][inx_nmtp][idn(i,j+nyskip,k+nzskip,3,4)];
                    cube[1][1][0][1] = bulkinfo[1][inx_nmtp][idn(i+nxskip,j,k+nzskip,3,4)];

                    //double P_center = Intp4D(cube[0],cube[1],ratio)+Para.Edfrz;
                    P_center = Para.Edfrz;
                   
                    
                    
                    double qb0_center,qb1_center,qb2_center,qb3_center;
                
                    //qb0
                    cube[0][0][0][0] = bulkinfo[0][inx_qb][idn(i,j,k,0,4)];
                    cube[0][1][1][1] = bulkinfo[0][inx_qb][idn(i+nxskip,j+nyskip,k+nzskip,0,4)];
                    cube[0][1][0][0] = bulkinfo[0][inx_qb][idn(i+nxskip,j,k,0,4)];
                    cube[0][0][1][0] = bulkinfo[0][inx_qb][idn(i,j+nyskip,k,0,4)];
                    cube[0][0][0][1] = bulkinfo[0][inx_qb][idn(i,j,k+nzskip,0,4)];
                    cube[0][1][1][0] = bulkinfo[0][inx_qb][idn(i+nxskip,j+nyskip,k,0,4)];
                    cube[0][0][1][1] = bulkinfo[0][inx_qb][idn(i,j+nyskip,k+nzskip,0,4)];
                    cube[0][1][0][1] = bulkinfo[0][inx_qb][idn(i+nxskip,j,k+nzskip,0,4)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_qb][idn(i,j,k,0,4)];
                    cube[1][1][1][1] = bulkinfo[1][inx_qb][idn(i+nxskip,j+nyskip,k+nzskip,0,4)];
                    cube[1][1][0][0] = bulkinfo[1][inx_qb][idn(i+nxskip,j,k,0,4)];
                    cube[1][0][1][0] = bulkinfo[1][inx_qb][idn(i,j+nyskip,k,0,4)];
                    cube[1][0][0][1] = bulkinfo[1][inx_qb][idn(i,j,k+nzskip,0,4)];
                    cube[1][1][1][0] = bulkinfo[1][inx_qb][idn(i+nxskip,j+nyskip,k,0,4)];
                    cube[1][0][1][1] = bulkinfo[1][inx_qb][idn(i,j+nyskip,k+nzskip,0,4)];
                    cube[1][1][0][1] = bulkinfo[1][inx_qb][idn(i+nxskip,j,k+nzskip,0,4)];

                    qb0_center = Intp4D(cube[0],cube[1],ratio);


                    //qb1
                    cube[0][0][0][0] = bulkinfo[0][inx_qb][idn(i,j,k,1,4)];
                    cube[0][1][1][1] = bulkinfo[0][inx_qb][idn(i+nxskip,j+nyskip,k+nzskip,1,4)];
                    cube[0][1][0][0] = bulkinfo[0][inx_qb][idn(i+nxskip,j,k,1,4)];
                    cube[0][0][1][0] = bulkinfo[0][inx_qb][idn(i,j+nyskip,k,1,4)];
                    cube[0][0][0][1] = bulkinfo[0][inx_qb][idn(i,j,k+nzskip,1,4)];
                    cube[0][1][1][0] = bulkinfo[0][inx_qb][idn(i+nxskip,j+nyskip,k,1,4)];
                    cube[0][0][1][1] = bulkinfo[0][inx_qb][idn(i,j+nyskip,k+nzskip,1,4)];
                    cube[0][1][0][1] = bulkinfo[0][inx_qb][idn(i+nxskip,j,k+nzskip,1,4)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_qb][idn(i,j,k,1,4)];
                    cube[1][1][1][1] = bulkinfo[1][inx_qb][idn(i+nxskip,j+nyskip,k+nzskip,1,4)];
                    cube[1][1][0][0] = bulkinfo[1][inx_qb][idn(i+nxskip,j,k,1,4)];
                    cube[1][0][1][0] = bulkinfo[1][inx_qb][idn(i,j+nyskip,k,1,4)];
                    cube[1][0][0][1] = bulkinfo[1][inx_qb][idn(i,j,k+nzskip,1,4)];
                    cube[1][1][1][0] = bulkinfo[1][inx_qb][idn(i+nxskip,j+nyskip,k,1,4)];
                    cube[1][0][1][1] = bulkinfo[1][inx_qb][idn(i,j+nyskip,k+nzskip,1,4)];
                    cube[1][1][0][1] = bulkinfo[1][inx_qb][idn(i+nxskip,j,k+nzskip,1,4)];

                    qb1_center = Intp4D(cube[0],cube[1],ratio);



                    //qb2
                    cube[0][0][0][0] = bulkinfo[0][inx_qb][idn(i,j,k,2,4)];
                    cube[0][1][1][1] = bulkinfo[0][inx_qb][idn(i+nxskip,j+nyskip,k+nzskip,2,4)];
                    cube[0][1][0][0] = bulkinfo[0][inx_qb][idn(i+nxskip,j,k,2,4)];
                    cube[0][0][1][0] = bulkinfo[0][inx_qb][idn(i,j+nyskip,k,2,4)];
                    cube[0][0][0][1] = bulkinfo[0][inx_qb][idn(i,j,k+nzskip,2,4)];
                    cube[0][1][1][0] = bulkinfo[0][inx_qb][idn(i+nxskip,j+nyskip,k,2,4)];
                    cube[0][0][1][1] = bulkinfo[0][inx_qb][idn(i,j+nyskip,k+nzskip,2,4)];
                    cube[0][1][0][1] = bulkinfo[0][inx_qb][idn(i+nxskip,j,k+nzskip,2,4)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_qb][idn(i,j,k,2,4)];
                    cube[1][1][1][1] = bulkinfo[1][inx_qb][idn(i+nxskip,j+nyskip,k+nzskip,2,4)];
                    cube[1][1][0][0] = bulkinfo[1][inx_qb][idn(i+nxskip,j,k,2,4)];
                    cube[1][0][1][0] = bulkinfo[1][inx_qb][idn(i,j+nyskip,k,2,4)];
                    cube[1][0][0][1] = bulkinfo[1][inx_qb][idn(i,j,k+nzskip,2,4)];
                    cube[1][1][1][0] = bulkinfo[1][inx_qb][idn(i+nxskip,j+nyskip,k,2,4)];
                    cube[1][0][1][1] = bulkinfo[1][inx_qb][idn(i,j+nyskip,k+nzskip,2,4)];
                    cube[1][1][0][1] = bulkinfo[1][inx_qb][idn(i+nxskip,j,k+nzskip,2,4)];

                    qb2_center = Intp4D(cube[0],cube[1],ratio);
  
                    //qb3
                    cube[0][0][0][0] = bulkinfo[0][inx_qb][idn(i,j,k,3,4)];
                    cube[0][1][1][1] = bulkinfo[0][inx_qb][idn(i+nxskip,j+nyskip,k+nzskip,3,4)];
                    cube[0][1][0][0] = bulkinfo[0][inx_qb][idn(i+nxskip,j,k,3,4)];
                    cube[0][0][1][0] = bulkinfo[0][inx_qb][idn(i,j+nyskip,k,3,4)];
                    cube[0][0][0][1] = bulkinfo[0][inx_qb][idn(i,j,k+nzskip,3,4)];
                    cube[0][1][1][0] = bulkinfo[0][inx_qb][idn(i+nxskip,j+nyskip,k,3,4)];
                    cube[0][0][1][1] = bulkinfo[0][inx_qb][idn(i,j+nyskip,k+nzskip,3,4)];
                    cube[0][1][0][1] = bulkinfo[0][inx_qb][idn(i+nxskip,j,k+nzskip,3,4)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_qb][idn(i,j,k,3,4)];
                    cube[1][1][1][1] = bulkinfo[1][inx_qb][idn(i+nxskip,j+nyskip,k+nzskip,3,4)];
                    cube[1][1][0][0] = bulkinfo[1][inx_qb][idn(i+nxskip,j,k,3,4)];
                    cube[1][0][1][0] = bulkinfo[1][inx_qb][idn(i,j+nyskip,k,3,4)];
                    cube[1][0][0][1] = bulkinfo[1][inx_qb][idn(i,j,k+nzskip,3,4)];
                    cube[1][1][1][0] = bulkinfo[1][inx_qb][idn(i+nxskip,j+nyskip,k,3,4)];
                    cube[1][0][1][1] = bulkinfo[1][inx_qb][idn(i,j+nyskip,k+nzskip,3,4)];
                    cube[1][1][0][1] = bulkinfo[1][inx_qb][idn(i+nxskip,j,k+nzskip,3,4)];

                    qb3_center = Intp4D(cube[0],cube[1],ratio);



                    double bulkpr_center;
                
                    //bulkpr
                    cube[0][0][0][0] = bulkinfo[0][inx_bulkpr][idn(i,j,k,0,1)];
                    cube[0][1][1][1] = bulkinfo[0][inx_bulkpr][idn(i+nxskip,j+nyskip,k+nzskip,0,1)];
                    cube[0][1][0][0] = bulkinfo[0][inx_bulkpr][idn(i+nxskip,j,k,0,1)];
                    cube[0][0][1][0] = bulkinfo[0][inx_bulkpr][idn(i,j+nyskip,k,0,1)];
                    cube[0][0][0][1] = bulkinfo[0][inx_bulkpr][idn(i,j,k+nzskip,0,1)];
                    cube[0][1][1][0] = bulkinfo[0][inx_bulkpr][idn(i+nxskip,j+nyskip,k,0,1)];
                    cube[0][0][1][1] = bulkinfo[0][inx_bulkpr][idn(i,j+nyskip,k+nzskip,0,1)];
                    cube[0][1][0][1] = bulkinfo[0][inx_bulkpr][idn(i+nxskip,j,k+nzskip,0,1)];
                
                    cube[1][0][0][0] = bulkinfo[1][inx_bulkpr][idn(i,j,k,0,1)];
                    cube[1][1][1][1] = bulkinfo[1][inx_bulkpr][idn(i+nxskip,j+nyskip,k+nzskip,0,1)];
                    cube[1][1][0][0] = bulkinfo[1][inx_bulkpr][idn(i+nxskip,j,k,0,1)];
                    cube[1][0][1][0] = bulkinfo[1][inx_bulkpr][idn(i,j+nyskip,k,0,1)];
                    cube[1][0][0][1] = bulkinfo[1][inx_bulkpr][idn(i,j,k+nzskip,0,1)];
                    cube[1][1][1][0] = bulkinfo[1][inx_bulkpr][idn(i+nxskip,j+nyskip,k,0,1)];
                    cube[1][0][1][1] = bulkinfo[1][inx_bulkpr][idn(i,j+nyskip,k+nzskip,0,1)];
                    cube[1][1][0][1] = bulkinfo[1][inx_bulkpr][idn(i+nxskip,j,k+nzskip,0,1)];

                    bulkpr_center = Intp4D(cube[0],cube[1],ratio);
                    
                    //savefile
                    
                    fsurface<<dsigma[0]*cell_center[0]<<" "<<-dsigma[1]*cell_center[0]<<" "<<-dsigma[2]*cell_center[0]<<" "<<-dsigma[3]<<" "
                            <<vx_center<<" "<<vy_center<<" "<<vz_center<<" "<<cell_center[3]<<std::endl;
                    ftxyz<<cell_position[0]<<" "<<cell_position[1]<<" "<<cell_position[2]<<" "<<cell_position[3]<<std::endl;
                   
                    
                    

                    
                    fnbmutp<<nb_center<<" "<<mu_center<<" "<<T_center<<" "<<P_center<<std::endl;
                    
                    fqb << qb0_center << " "<<qb1_center<<" "<<qb2_center<<" "<<qb3_center<<std::endl;
                    fbulkpr << bulkpr_center <<std::endl;
                    fpimn<< pi00_center<<" "<<pi01_center<<" "<<pi02_center<<" "<<pi03_center<<" "
                         << pi11_center<<" "<<pi12_center<<" "<<pi13_center<<" "<<pi22_center<<" "
                         <<pi23_center<<" "<< pi33_center<<std::endl;    

               
                   
                    
           


                    if(flag_vorticity){

                
                    for(int omegaid = 0 ; omegaid < 6;omegaid ++){
                           fomega<<omega_center[omegaid]<<" ";
                           fomega_accT<<omega_accT_center[omegaid]<<" "; 
                    }
                    fomega<<std::endl;
                    fomega_accT<<std::endl;
                    for(int omegaid = 0 ; omegaid < 16;omegaid ++){
                        fomega_shear1<<omega_shear1_center[omegaid]<<" "; 
                    }
                    fomega_shear1<<std::endl;
                    for(int omegaid = 0 ; omegaid < 4;omegaid ++){
                        fomega_shear2 << omega_shear2_center[omegaid] <<" "; 
                    }
                    fomega_shear2<<std::endl;
                        
                
                    for(int omegaid = 0 ; omegaid < 6;omegaid ++){
                        fomega_chemical<< omega_chemical_center[omegaid]<<" ";
                    }
                    fomega_chemical<<std::endl;
                    
                    }

              
                

                
                    Nintersection++; 
                    

                }
                


              
           }

       }
    }

    
    // fsurface.close();          
    // fqb.close();
    // fbulkpr.close();
    // fpimn.close();
    // fnbmutp.close();
    // ftxyz.close();
    // fomega.close();
    // fomega_shear1.close();
    // fomega_shear2.close();
    // fomega_accT.close();
    // fomega_chemical.close();
    

    







    

}


void  Hypersf::equal_tau_corona(){
    
    const int dim = 4;
    int ntskip = Para.ntskip;
    int nxskip = Para.nxskip;
    int nyskip = Para.nyskip;
    int nzskip = Para.nzskip;

    
    int NX = Para.NX;
    int NY = Para.NY;
    int NZ = Para.NZ;

    double dt = Para.dt;
    double dx = Para.dx;
    double dy = Para.dy;
    double dz = Para.dz;

    double DTAU = ntskip*dt;
    double DX =   nxskip*dx;
    double DY =   nyskip*dy;
    double DZ =   nzskip*dz;

    double DSTEP[dim] = {DTAU, DX, DY, DZ};
    
    double Edhi = Para.Edfrz;
    double Edlo = 0.05;

    double ****cube = new double*** [2];
    for(int i = 0 ; i < 2 ; i++ ){
        cube[i] = new double** [2];
        for (int j = 0; j < 2; j++){
            cube[i][j] = new double *[2];
            for (int k = 0; k < 2; k++){
                cube[i][j][k] = new double[2];
                for (int m = 0 ; m < 2;m++)
                    cube[i][j][k][m] = 0.0;   

            }

        }

    }


    std::stringstream ssfile_surface;
    std::stringstream ssfile_qb;
    std::stringstream ssfile_pimn;
    std::stringstream ssfile_bulkpr;
    std::stringstream ssfile_nbmutp;
    std::stringstream ssfile_txyz;
    
    std::stringstream ssfile_omega;
    std::stringstream ssfile_omega_shear1;
    std::stringstream ssfile_omega_shear2;
    std::stringstream ssfile_omega_accT;
    std::stringstream ssfile_omega_chemical;

    

    ssfile_surface<<DataPath<<"/hypersf.dat";
    ssfile_qb<<DataPath<<"/qbmusf.dat";
    ssfile_pimn<<DataPath<<"/pimnsf.dat";
    ssfile_bulkpr<<DataPath<<"/bulkprsf.dat";
    ssfile_nbmutp<<DataPath<<"/sf_nbmutp.dat";
    ssfile_txyz<<DataPath<<"/sf_txyz.dat";

    ssfile_omega<<DataPath<<"/omegamu_sf.dat";
    ssfile_omega_shear1<<DataPath<<"/omegamu_shear1_sf.dat";
    ssfile_omega_shear2<<DataPath<<"/omegamu_shear2_sf.dat";
    ssfile_omega_accT<<DataPath<<"/omegamu_accT_sf.dat";
    ssfile_omega_chemical<<DataPath<<"/omegamu_chemical_sf.dat";
    
    
 
    ofstream fsurface(ssfile_surface.str().c_str(),ios::app);
    ofstream fqb(ssfile_qb.str().c_str(),ios::app);
    ofstream fbulkpr(ssfile_bulkpr.str().c_str(),ios::app);
    ofstream fpimn(ssfile_pimn.str().c_str(),ios::app);
    ofstream fnbmutp(ssfile_nbmutp.str().c_str(),ios::app);
    ofstream ftxyz(ssfile_txyz.str().c_str(),ios::app);


    ofstream fomega(ssfile_omega.str().c_str(),ios::app);
    ofstream fomega_shear1(ssfile_omega_shear1.str().c_str(),ios::app);
    ofstream fomega_shear2(ssfile_omega_shear2.str().c_str(),ios::app);
    ofstream fomega_accT(ssfile_omega_accT.str().c_str(),ios::app);
    ofstream fomega_chemical(ssfile_omega_chemical.str().c_str(),ios::app);
    
    

    //if(Header == 0){
    //    if (Para.Tfrz_on )
    //    {    
    //
    //        fsurface<< scientific << setprecision(6)<<"# Tfrz="<<Para.Tfrz<<"; other rows: dS0, dS1, dS2, dS3, vx, vy, veta, etas"<<std::endl;
    //    }
    //    else{
    //        fsurface<<"# efrz="<<Para.Edfrz<<"; other rows: dS0, dS1, dS2, dS3, vx, vy, veta, etas"<<std::endl;
    //    }
    //    
    //    fqb<<"# qb0 qb1 qb2 qb3"<<std::endl;
    //    fbulkpr<<" # bulk pressure"<<std::endl;
    //    fpimn<<"# one_o_2TsqrEplusP="<< bulkinfo[0][inx_deltaf][0] <<" pi00 01 02 03 11 12 13 22 23 33"<<std::endl;
    //    fnbmutp<<"# nb, mu, T, Pr of hypersf elements"<<std::endl;
    //    fomega<<"omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
    //    fomega_shear1<<"omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
    //    fomega_shear2<<"omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
    //    fomega_accT<<"omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
    //    fomega_chemical<<"omega^{01, 02, 03, 12, 13, 23}"<<std::endl;
    //    ftxyz<<"(t, x, y, z) the time-space coordinates of hypersf elements"<<std::endl;
    //}
      
    //std::cout<< Header << std::endl;
    int Nintersection = 0;
    for(int i = 0; i < NX - nxskip; i += nxskip){
       double xx = i*dx - NX/2*dx;
       for (int j = 0; j < NY - nyskip; j += nyskip){
           double yy = j*dy - NY/2*dy;
           for(int k = 0 ; k < NZ - nzskip; k += nzskip){
               double zz = k*dz - NZ/2*dz;

              
                cube[0][0][0][0] = bulkinfo[0][0][idn(i,j,k,0,4)];
                cube[0][1][1][1] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k+nzskip,0,4)];
                cube[0][1][0][0] = bulkinfo[0][0][idn(i+nxskip,j,k,0,4)];
                cube[0][0][1][0] = bulkinfo[0][0][idn(i,j+nyskip,k,0,4)];
                cube[0][0][0][1] = bulkinfo[0][0][idn(i,j,k+nzskip,0,4)];
                cube[0][1][1][0] = bulkinfo[0][0][idn(i+nxskip,j+nyskip,k,0,4)];
                cube[0][0][1][1] = bulkinfo[0][0][idn(i,j+nyskip,k+nzskip,0,4)];
                cube[0][1][0][1] = bulkinfo[0][0][idn(i+nxskip,j,k+nzskip,0,4)];
                 

                cube[1][0][0][0] = bulkinfo[1][0][idn(i,j,k,0,4)];
                cube[1][1][1][1] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k+nzskip,0,4)];
                cube[1][1][0][0] = bulkinfo[1][0][idn(i+nxskip,j,k,0,4)];
                cube[1][0][1][0] = bulkinfo[1][0][idn(i,j+nyskip,k,0,4)];
                cube[1][0][0][1] = bulkinfo[1][0][idn(i,j,k+nzskip,0,4)];
                cube[1][1][1][0] = bulkinfo[1][0][idn(i+nxskip,j+nyskip,k,0,4)];
                cube[1][0][1][1] = bulkinfo[1][0][idn(i,j+nyskip,k+nzskip,0,4)];
                cube[1][1][0][1] = bulkinfo[1][0][idn(i+nxskip,j,k+nzskip,0,4)];
                
                

                int intersection = 1;
                double EDC = 0.0;
            
                
                 if( (cube[0][0][0][0]) > Edhi  ) continue;
                 if( (cube[0][0][0][0]) < Edlo  ) continue; 
                 EDC = cube[0][0][0][0];

                 double dsigma[4] = { OldTime*DX*DY*DZ, 0.0, 0.0, 0.0 };
                 double cell_center[4] = {OldTime, xx, yy, zz};

                 double cell_position[4];
                 cell_position[0] = cell_center[0]*cosh(cell_center[3]);
                 cell_position[1] = cell_center[1];
                 cell_position[2] = cell_center[2];
                 cell_position[3] = cell_center[0]*sinh(cell_center[3]);
                
                 double vx_center = bulkinfo[0][0][idn(i,j,k,1,4)];
                 double vy_center = bulkinfo[0][0][idn(i,j,k,2,4)];
                 double vz_center = bulkinfo[0][0][idn(i,j,k,3,4)];


                 double qb0_center,qb1_center,qb2_center,qb3_center;
                 double nb_center,mu_center,T_center,P_center;

                 double pi00_center,pi01_center,pi02_center,pi03_center,
                           pi11_center,pi12_center,pi13_center,pi22_center,
                           pi23_center, pi33_center;
                 double bulkpr_center;
                 double omega_center[6]={0};
                 double omega_shear1_center[16]={0};
                 double omega_shear2_center[4]={0};
                 double omega_accT_center[6]={0};    
                 double omega_chemical_center[6]={0};    

                pi00_center = bulkinfo[0][inx_pimn][idn(i,j,k,0,10)];
                pi01_center = bulkinfo[0][inx_pimn][idn(i,j,k,1,10)];
                pi02_center = bulkinfo[0][inx_pimn][idn(i,j,k,2,10)];
                pi03_center = bulkinfo[0][inx_pimn][idn(i,j,k,3,10)];
                pi11_center = bulkinfo[0][inx_pimn][idn(i,j,k,4,10)];
                pi12_center = bulkinfo[0][inx_pimn][idn(i,j,k,5,10)];
                pi13_center = bulkinfo[0][inx_pimn][idn(i,j,k,6,10)];
                pi22_center = bulkinfo[0][inx_pimn][idn(i,j,k,7,10)];
                pi23_center = bulkinfo[0][inx_pimn][idn(i,j,k,8,10)];
                pi33_center = bulkinfo[0][inx_pimn][idn(i,j,k,9,10)];

                bulkpr_center = bulkinfo[0][inx_bulkpr][idn(i,j,k,0,1)];
                
                if (flag_vorticity){
                
                for(int omegaid = 0 ; omegaid < 6;omegaid ++){
                    omega_center[omegaid] = bulkinfo[0][inx_omega][idn(i,j,k,omegaid,6)];
                    omega_accT_center[omegaid] = bulkinfo[0][inx_omega_accT][idn(i,j,k,omegaid,6)];
                }
                for(int omegaid = 0 ; omegaid < 16;omegaid ++){
                    omega_shear1_center[omegaid] = bulkinfo[0][inx_omega_shear1][idn(i,j,k,omegaid,16)];
                }
                for(int omegaid = 0 ; omegaid < 4;omegaid ++){
                    omega_shear2_center[omegaid] = bulkinfo[0][inx_omega_shear2][idn(i,j,k,omegaid,4)];
                }
                    
                
                for(int omegaid = 0 ; omegaid < 6;omegaid ++){
                        omega_chemical_center[omegaid] = bulkinfo[0][inx_omega_chemical][idn(i,j,k,omegaid,6)];
                }
                
                }
                 
                nb_center = bulkinfo[0][inx_nmtp][idn(i,j,k,0,4)];
                mu_center = bulkinfo[0][inx_nmtp][idn(i,j,k,1,4)];
                T_center = bulkinfo[0][inx_nmtp][idn(i,j,k,2,4)];
                P_center = EDC;
                qb0_center = bulkinfo[0][inx_qb][idn(i,j,k,0,4)];
                qb1_center = bulkinfo[0][inx_qb][idn(i,j,k,1,4)];
                qb2_center = bulkinfo[0][inx_qb][idn(i,j,k,2,4)];
                qb3_center = bulkinfo[0][inx_qb][idn(i,j,k,3,4)];

                
                fsurface<<dsigma[0]<<" "<<dsigma[1]<<" "<<dsigma[2]<<" "<<dsigma[3]<<" "
                            <<vx_center<<" "<<vy_center<<" "<<vz_center<<" "<<cell_center[3]<<std::endl;
                ftxyz<<cell_position[0]<<" "<<cell_position[1]<<" "<<cell_position[2]<<" "<<cell_position[3]<<std::endl;
		
		        
                    
                fnbmutp<<nb_center<<" "<<mu_center<<" "<<T_center<<" "<<P_center<<std::endl;
                
                fqb << qb0_center << " "<<qb1_center<<" "<<qb2_center<<" "<<qb3_center<<std::endl;
                fbulkpr << bulkpr_center << std::endl;
                fpimn<< pi00_center<<" "<<pi01_center<<" "<<pi02_center<<" "<<pi03_center<<" "
                         << pi11_center<<" "<<pi12_center<<" "<<pi13_center<<" "<<pi22_center<<" "
                         <<pi23_center<<" "<< pi33_center<<std::endl;    

                
        

                
           
                
                
                if(flag_vorticity){

                
                for(int omegaid = 0 ; omegaid < 6;omegaid ++){
                       fomega<<omega_center[omegaid]<<" ";
                       fomega_accT<<omega_accT_center[omegaid]<<" "; 
                }
                fomega<<std::endl;
                fomega_accT<<std::endl;
                for(int omegaid = 0 ; omegaid < 16;omegaid ++){
                    fomega_shear1<<omega_shear1_center[omegaid]<<" "; 
                }
                fomega_shear1<<std::endl;
                for(int omegaid = 0 ; omegaid < 4;omegaid ++){
                    fomega_shear2 << omega_shear2_center[omegaid] <<" "; 
                }
                fomega_shear2<<std::endl;
                    
            
                
                for(int omegaid = 0 ; omegaid < 6;omegaid ++){
                    fomega_chemical<< omega_chemical_center[omegaid]<<" ";
                }
                fomega_chemical<<std::endl;
                
                }


                

            
                
              
           }

       }
    }

    

    
    
    // fqb.close();
    // fpimn.close();
    // fnbmutp.close();
    // fsurface.close(); 
    // ftxyz.close();
    // fomega.close();
    // fomega_shear1.close();
    // fomega_shear2.close();
    // fomega_accT.close();
    // fomega_chemical.close();
    // fbulkpr.close();
    

}

