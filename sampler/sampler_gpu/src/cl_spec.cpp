#include "cl_spec.h"

#define idx(i,j) (((i)<(j))?((7*(i)+2*(j)-(i)*(i))/2):((7*(j)+2*(i)-(j)*(j))/2))

cl_real excutionTime( cl::Event & event )
{
    cl_ulong tstart, tend; 
    event.getProfilingInfo( CL_PROFILING_COMMAND_START, & tstart ) ;
    event.getProfilingInfo( CL_PROFILING_COMMAND_END, & tend ) ;
//    //std::cout<<"#run time="<<(tend - tstart )/1000<<"ms\n";
    return ( tend - tstart ) * 1.0E-9 ;
}


void tensor_boost(cl_real u[4], cl_real Pi[][4], cl_real PiTilde[][4]) {
    /** lorentz boost a tensor pi[mu][nu] to pi*[mu][nu] */
	int alpha,beta,gamma,delta;
	cl_real Linv[4][4],L[4][4];  // Lorentz Boost Matrices
	cl_real ucontra[4]={u[0],-u[1],-u[2],-u[3]},n[4]={1.0,0,0,0};
	for(alpha=0;alpha<4;alpha++){
		for(beta=0;beta<4;beta++){
			Linv[alpha][beta]=2.0*n[alpha]*ucontra[beta]-(u[alpha]+n[alpha])
                                   *(ucontra[beta]+n[beta])/(1.0+u[0]);
			if(alpha==beta) Linv[alpha][beta]+=1.0;
			L[beta][alpha]=Linv[alpha][beta];
		}
	}
	for(alpha=0;alpha<4;alpha++){
		for(delta=0;delta<4;delta++){
			PiTilde[alpha][delta]=0.0;
			for(beta=0;beta<4;beta++){
				for(gamma=0;gamma<4;gamma++){
					PiTilde[alpha][delta]+=
                         Linv[alpha][beta]*Pi[beta][gamma]*L[gamma][delta];
				}
			}
		}
	}
}




Spec::Spec(const std::string & pathin, int decay_on,int gpu_id,std::string & EOS_TYPE, int nsampling,const std::string & model,int vorticity_on){

    gpu_id_ = gpu_id;
    DataPath = pathin;
    decay_on_ = decay_on;
    nsampling_ = nsampling;
    model_ = model;
    eos_type_ = EOS_TYPE;
    vorticity_on_ = vorticity_on;



    flag_mu_pce = false;
    if(eos_type_ == "LATTICE_PCE165" || eos_type_ == "LATTICE_PCE150"||
       eos_type_ == "IDEAL_GAS" || eos_type_ == "LATTICE_WB"||
       eos_type_ == "PURE_GAUGE"|| eos_type_ == "FIRST_ORDER"||
       eos_type_ == "HOTQCD2014"){
            flag_mu_pce = true;

    }


    std::stringstream hypsfDataFile;
    std::stringstream pathout;
    pathout<<pathin;
    hypsfDataFile<<pathin<<"/hypersf.dat";
    // hypsfDataFile stores comments in the first row, Tfrz in the second row
    // dS^{0} dS^{1} dS^{2} dS^{3} vx vy veta eta_s for all other rows
    ReadHyperSF(hypsfDataFile.str());


    std::stringstream txyzFile;
    txyzFile<<pathin<<"/sf_txyz.dat";
    Readtxyz(txyzFile.str());

    std::stringstream NmutpDataFile;
    NmutpDataFile<<pathin<<"/sf_nbmutp.dat";
    ReadNmutp(NmutpDataFile.str());

    if(flag_mu_pce){
         std::stringstream chemical_potential_datafile;
         chemical_potential_datafile<<pathin<<"/chemical_potential.dat";
         ReadMuB(chemical_potential_datafile.str());
    }

    
    std::stringstream pdgfile;
    pdgfile<<pathin<<"/pdgfile.dat"; 
    ReadParticles( pdgfile.str() );
    std::cout<<"stable particle: \n";
    for( int i=0; i<particles.size(); i++ ){
        if(particles.at(i).stable == true) std::cout<<particles.at(i).monval<<' ';
    }
    std::cout<<'\n';


    // pisfDataFile stores comments in the first row, 1.0/(2.0*T^2(e+P)) in the second row
    // pi^{00} 01 02 03 11 12 13 22 23 33 on the freeze out hyper surface for other rows
    std::stringstream pisfDataFile;
    pisfDataFile<<pathin<<"/pimnsf.dat";
    ReadPimnSF(pisfDataFile.str());

    std::stringstream qbsfDataFile;
    qbsfDataFile<<pathin<<"/qbmusf.dat";
    ReadQbSF(qbsfDataFile.str());
    std::stringstream deltaf_qb_File;
    deltaf_qb_File<<pathin<<"/Coefficients_RTA_diffusion.dat";
    Readdeltaf(deltaf_qb_File.str());

    if(vorticity_on_){
        std::stringstream vorticityDataFile1;
        vorticityDataFile1<<pathin<<"/omegamu_sf.dat";
        Readvorticity1(vorticityDataFile1.str());

        std::stringstream vorticityDataFile2;
        vorticityDataFile2<<pathin<<"/omegamu_shear1_sf.dat";
        Readvorticity2(vorticityDataFile2.str());

        std::stringstream vorticityDataFile3;
        vorticityDataFile3<<pathin<<"/omegamu_shear2_sf.dat";
        Readvorticity3(vorticityDataFile3.str());


        std::stringstream vorticityDataFile4;
        vorticityDataFile4<<pathin<<"/omegamu_accT_sf.dat";
        Readvorticity4(vorticityDataFile4.str());


        std::stringstream vorticityDataFile5;
        vorticityDataFile5<<pathin<<"/omegamu_chemical_sf.dat";
        Readvorticity5(vorticityDataFile5.str());
    }






    SetPathOut(pathout.str());

    
   

}



void Spec::Readvorticity1(const std::string &vorticityDataFile1)
{

    std::ifstream fin(vorticityDataFile1);
    char buf[256];
    cl_real omega[6];
    if (fin.is_open() ){
        fin.getline(buf,256);
        while(fin.good() ){
            for (int i = 0 ; i < 6 ; i++)
            {
                fin >> omega[i];
            }
            if(fin.eof() ) break;
            for (int i = 0; i <6 ;i++)
            {
                h_omega_th.push_back(omega[i]);
            }
        }
        fin.close();
        std::cout<<"#vorticity thermal size="<< h_omega_th.size()/6<<std::endl;

    }
    else{
        std::cerr<<"#Can't open hyper-surface data file for vorticity thermal!\n";
        exit(0);
    }

    if (SizeSF != h_omega_th.size()/6) {
        std::cout << "num of vorticity thermal on sf is not correct!\n";
    }


}


void Spec::Readvorticity2(const std::string &vorticityDataFile2)
{

    std::ifstream fin(vorticityDataFile2);
    char buf[256];
    cl_real omega[16];
    if (fin.is_open() ){
        fin.getline(buf,256);
        while(fin.good() ){
            for (int i = 0 ; i < 16 ; i++)
            {
                fin >> omega[i];
            }
            if(fin.eof() ) break;
            for (int i = 0; i <16 ;i++)
            {
                h_omega_shear1.push_back(omega[i]);
            }
        }
        fin.close();
        std::cout<<"#vorticity shear1 size="<< h_omega_shear1.size()/16<<std::endl;

    }
    else{
        std::cerr<<"#Can't open hyper-surface data file for vorticity shear1!\n";
        exit(0);
    }

    if (SizeSF != h_omega_shear1.size()/16) {
        std::cout << "num of vorticity shear1 on sf is not correct!\n";
    }


}


void Spec::Readvorticity3(const std::string &vorticityDataFile3)
{

    std::ifstream fin(vorticityDataFile3);
    char buf[256];
    cl_real omega[4];
    if (fin.is_open() ){
        fin.getline(buf,256);
        while(fin.good() ){
            for (int i = 0 ; i < 4 ; i++)
            {
                fin >> omega[i];
            }
            if(fin.eof() ) break;
            for (int i = 0; i <4 ;i++)
            {
                h_omega_shear2.push_back(omega[i]);
            }
        }
        fin.close();
        std::cout<<"#vorticity shear2 size="<< h_omega_shear2.size()/4<<std::endl;

    }
    else{
        std::cerr<<"#Can't open hyper-surface data file for vorticity shear2!\n";
        exit(0);
    }

    if (SizeSF != h_omega_shear2.size()/4) {
        std::cout << "num of vorticity shear2 on sf is not correct!\n";
    }


}



void Spec::Readvorticity4(const std::string &vorticityDataFile4)
{

    std::ifstream fin(vorticityDataFile4);
    char buf[256];
    cl_real omega[6];
    if (fin.is_open() ){
        fin.getline(buf,256);
        while(fin.good() ){
            for (int i = 0 ; i < 6 ; i++)
            {
                fin >> omega[i];
            }
            if(fin.eof() ) break;
            for (int i = 0; i <6 ;i++)
            {
                h_omega_accT.push_back(omega[i]);
            }
        }
        fin.close();
        std::cout<<"#vorticity accT size="<< h_omega_accT.size()/6<<std::endl;

    }
    else{
        std::cerr<<"#Can't open hyper-surface data file for vorticity accT!\n";
        exit(0);
    }

    if (SizeSF != h_omega_accT.size()/6) {
        std::cout << "num of vorticity accT on sf is not correct!\n";
    }


}

void Spec::Readvorticity5(const std::string &vorticityDataFile5)
{

    std::ifstream fin(vorticityDataFile5);
    char buf[256];
    cl_real omega[6];
    if (fin.is_open() ){
        fin.getline(buf,256);
        while(fin.good() ){
            for (int i = 0 ; i < 6 ; i++)
            {
                fin >> omega[i];
            }
            if(fin.eof() ) break;
            for (int i = 0; i <6 ;i++)
            {
                h_omega_chemical.push_back(omega[i]);
            }
        }
        fin.close();
        std::cout<<"#vorticity chemical size="<< h_omega_chemical.size()/6<<std::endl;

    }
    else{
        std::cerr<<"#Can't open hyper-surface data file for vorticity chemical!\n";
        exit(0);
    }

    if (SizeSF != h_omega_chemical.size()/6) {
        std::cout << "num of vorticity chemical on sf is not correct!\n";
    }
}



void Spec::ReadHyperSF( const std::string & dataFile )
{
    cl_real dA0, dA1, dA2, dA3, vx, vy, vh, tau, x, y, etas;
    std::ifstream fin(dataFile);
    char buf[256];
    std::cout<< dataFile<<std::endl;
    if ( fin.is_open() ) {
        fin.getline(buf, 256);  // readin the comment
        std::string comments(buf);
        //Tfrz = std::stof(comments.substr(7));
        //Edfrz = std::stof(comments.substr(7));
        while( fin.good() ){
            fin>>dA0>>dA1>>dA2>>dA3>>vx>>vy>>vh>>etas;
            if( fin.eof() )break;  // eof() repeat the last line
            if ( std::isnan(dA0) || std::isnan(dA1) || std::isnan(dA2) || std::isnan(dA3) ) {
                dA0 = 0.0; dA1 = 0.0; dA2 = 0.0; dA3=0.0;
                vx = 0.0; vy = 0.0; vh = 0.0; etas = 0.0;
                std::cout << "nan in hypersf data file!" << std::endl;
            }
            h_SF.push_back( (cl_real8){ dA0, dA1, dA2, dA3, vx, vy, vh, etas } );
        }
        fin.close();
        SizeSF = h_SF.size();
        std::cout<<"#hypersf size="<<SizeSF<<std::endl;
        //std::cout<<"#Edfrz = "<< Edfrz << " GeV/fm^3"<< std::endl;
    }

    else{
        std::cerr<<"#Can't open hyper-surface data file!\n";
        exit(0);
    }
}


void Spec::ReadMuB( const std::string & dataFile )
{
    cl_int pid;
    cl_real chemicalpotential;
    std::ifstream fin( dataFile );
    if(fin.is_open()){
        while( fin.good() ){
            fin>>pid>>chemicalpotential;
            if( fin.eof() )break;  // eof() repeat the last line
            muB[ pid ] = chemicalpotential;
            std::cout<<"pid="<<pid<<" muB="<<muB[pid]<<std::endl;
        }
        fin.close();
    }
    else{
        std::cerr<<"#Can't open muB data file!\n";
        exit(0);
    }
}

void Spec::Readtxyz(const std::string & sf_txyzfile)
{
    cl_real  t, x, y, z;
    std::cout<< sf_txyzfile<<std::endl;
    std::ifstream fin(sf_txyzfile);
    char buf[256];
    if ( fin.is_open() ) {
        while( fin.good() ){
            fin.getline(buf, 256);  // readin the comment
            fin>>t>>x>>y>>z;
            if( fin.eof() )break;  // eof() repeat the last line
            if ( std::isnan(t) || std::isnan(x) || std::isnan(y) || std::isnan(z) ) {
                std::cout << "nan in txyz data file!" << std::endl;
            }
            h_txyz.push_back( (cl_real4){ t,x,y,z } );
        }
        fin.close();
        std::cout<<"#txyz size= "<<h_txyz.size()<<std::endl;
        
    }

    else{
        std::cerr<<"#Can't open hyper-surface data file!\n";
        exit(0);
    }
}


void Spec::ReadNmutp(const std::string &Nmutp_datafile)
{
    cl_real nb_frz, mu_frz, T_frz, pr_frz;
    std::ifstream fin(Nmutp_datafile);
    char buf[256];
    int i =0;
    if(fin.is_open() )
    {
        fin.getline(buf, 256);
        while(fin.good())
	{
	    fin>>nb_frz>>mu_frz>>T_frz>>pr_frz;
	    if (fin.eof()) break;
	    h_nmtp.push_back( (cl_real4){nb_frz, mu_frz, T_frz, pr_frz } );
	}

	fin.close();
	std::cout<<"#nmtp size="<< h_nmtp.size()<<"T_frz = "<< T_frz   <<std::endl;
    }
    else{

        std::cerr<<"#Can't open nmtp data file!\n";
        exit(0);
    }
}

bool Spec::neglect_partices(int pdgid)
{
    std::vector<int> sigma_meason{9000221};
    bool neglect = false;
    for (int i =0 ; i < sigma_meason.size();i++)
    {
        if(abs(pdgid) == sigma_meason[i])
        {
            neglect = true;
            std::cout<< "neglect paticle PID "<<pdgid << std::endl;
        }
    }

    return neglect;
}


void Spec::ReadParticles(const std::string& particle_data_table)
{
    cl_real buff;
    
    std::ifstream fin(particle_data_table);
    CParticle p;

    if( fin.is_open() ){
        while(fin.good()){
            p.stable = 0;
            fin>>p.monval>>p.name>>p.mass>>p.width         \
                >>p.gspin>>p.baryon>>p.strange>>p.charm     \
                >>p.bottom>>p.gisospin>>p.charge>>p.decays;
            

            bool neglect = neglect_partices(p.monval);

            p.antibaryon_spec_exists = 0;

            if( fin.eof() ) break;
            CDecay dec;

            if(p.width < 1.0E-8)p.stable=1;

            /* one special case in pdg05.dat: eta with 4 decay channels,
             * but its width is smaller than 1.0E-8 GeV */
            for(int k=0; k<p.decays; k++){
                fin>>dec.pidR>>dec.numpart>>dec.branch>>dec.part[0] \
                        >>dec.part[1]>> dec.part[2]>> dec.part[3]>> dec.part[4];

                if ((!p.stable) && (dec.numpart!=1)&&(!neglect)) {
                    decay.push_back(dec);
                }
            }

            if (!neglect) 
            {
                particles.push_back(p);
	            h_HadronInfo.push_back(p.mass);
                h_HadronInfo.push_back(p.gspin);
                h_HadronInfo.push_back(fabs(p.baryon)>0.01?1.0f:-1.0f);
                h_HadronInfo.push_back(cl_real(p.baryon));
                if(flag_mu_pce){
                     h_HadronInfo.push_back(muB[p.monval]);
                }
                else{
                     h_HadronInfo.push_back(0.0f);

                }
            }
        
    } 
    fin.close();
    }else{
        std::cerr<<"#Failed to open pdg data table\n";
        exit(0);
    }
    


    CParticle antiB;//anti-baryon
    int N=particles.size();
    for(std::vector<CParticle>::size_type i=0; i!=N; i++){
        /** If unstable, pt range 0-8; if stable, pt range 0-4 */
        //cl_real resizePtRange = particles[i].stable ? 1.0 : 1.0 ;
        cl_real chem = muB[ particles[i].monval ];
        if (fabs(particles[i].baryon)>0.001) {
            antiB.monval = -particles[i].monval;
            antiB.name   = "A";
            antiB.name.append(particles[i].name);
            antiB.mass = particles[i].mass;
            antiB.width = particles[i].width;
            antiB.gspin = particles[i].gspin;
            antiB.baryon = -particles[i].baryon;
            antiB.strange= -particles[i].strange;
            antiB.charm = -particles[i].charm;
            antiB.bottom = -particles[i].bottom;
            antiB.gisospin=particles[i].gisospin;
            antiB.charge=-particles[i].charge;
            antiB.decays=particles[i].decays;
            antiB.stable=particles[i].stable;
            antiB.antibaryon_spec_exists = 1;
            //particles.push_back(antiB);
	    bool neglect = neglect_partices(antiB.monval);

	    if(!neglect)
	    {    particles.push_back(antiB);
		 h_HadronInfo.push_back(antiB.mass);
                 h_HadronInfo.push_back(antiB.gspin);
                 h_HadronInfo.push_back(fabs(antiB.baryon)>0.01?1.0f:-1.0f);
                 h_HadronInfo.push_back(cl_real(antiB.baryon));
                 if(flag_mu_pce){
                    h_HadronInfo.push_back(chem);
                 }
                 else{
                    h_HadronInfo.push_back(0.0f);

                 }
 

	    }
        }
    }

    SizePID = h_HadronInfo.size()/5;
    

    for( int i =0; i< SizePID; i++ ){
        newpid[ particles[i].monval ] =  i;
        
        
    }

    quark_level = false;
    if (SizePID < 8 && newpid.find(211) == newpid.end() ){
        quark_level = true;

    }

    std::cout<<"newpid of pion = "<<newpid[ 211 ]<<std::endl;
    std::cout<<"newpid of proton = "<<newpid[ -2212 ]<<std::endl;
    std::cout<<"newpid of -13334 = "<<newpid[ 2212 ]<<std::endl;
    std::cout<<"number of particle = "<<SizePID<<std::endl;
}


void Spec::ReadPimnSF(const std::string & piFile)
{
    std::ifstream fin2(piFile);
    char buf[256];
    cl_real pimn[4][4];
    cl_real pimn_star[4][4];
    cl_real umu[4];
    if ( fin2.is_open() ) {
        fin2.getline(buf, 256);  // readin the comment
        //std::string comments(buf);
        //one_over_2TsqrEplusP = std::stof(comments.substr(20));
        for(int n = 0 ; n < SizeSF; n++ ){
            
            for ( int i=0; i < 4; i++ ) {
                for ( int j=i; j<4; j++ ) {
                    fin2 >> pimn[i][j];
                    if ( i != j ) pimn[j][i] = pimn[i][j];
                }
            }
            cl_real8 SF = h_SF.at(n);
            cl_real  vsqr = SF.s[4]*SF.s[4] + SF.s[5]*SF.s[5] + SF.s[6]*SF.s[6];
            cl_real  utau;

            if(vsqr < 1.0 ){
                utau = 1.0/ std::sqrt(1.0 - vsqr );
                umu[0] = utau;
                umu[1] = utau*SF.s[4];
                umu[2] = utau*SF.s[5];
                umu[3] = utau*SF.s[6];
            }
            else{

                umu[0] = 1.0;
                umu[1] = 0.0;
                umu[2] = 0.0;
                umu[3] = 0.0;
            }

            tensor_boost( umu,pimn,pimn_star );
            
            double pimn_max = 0.0;
            for ( int i=0; i < 4; i++ ) {
                for ( int j=i; j<4; j++ ) {
                    double pi_ij = pimn_star[i][j];
                    h_pi.push_back ( pi_ij );
                }
            }

            


        }
          
        // while( fin2.good() ){
        //     for ( int i=0; i < 10; i++ ) {
        //         fin2 >> pimn[i];
        //     }
        //     if( fin2.eof() )break;  // eof() repeat the last line
        //     for ( int i=0; i < 10; i++ ) {
        //         if ( std::isnan(pimn[i]) ) pimn[i] = 0.0;
        //         h_pi.push_back(pimn[i]); 
        //     }
        // }
        fin2.close();
    }
    else{
        std::cerr<<"#Can't open hyper-surface data file for pimn!\n";
        exit(0);
    }

    if (SizeSF != h_pi.size()/10) {
        std::cout << "num of pi on sf is not correct!\n";
    }
}


void Spec::ReadQbSF(const std::string &qbFile)
{

    std::ifstream fin(qbFile);
    char buf[256];
    cl_real qb[4];
    if (fin.is_open() ){
        fin.getline(buf,256);
	while(fin.good() ){
	    for (int i = 0 ; i < 4 ; i++)
            {
		fin >> qb[i];
	    }
	    if(fin.eof() ) break;
	    for (int i = 0; i <4 ;i++)
	    {
                h_qb.push_back(qb[i]);
	    }
	}
	fin.close();
        std::cout<<"#qb size="<< h_qb.size()/4<<std::endl;

    }
    else{
        std::cerr<<"#Can't open hyper-surface data file for qb!\n";
        exit(0);
    }
    
    if (SizeSF != h_qb.size()/4) {
        std::cout << "num of qb on sf is not correct!\n";
    }


}

void Spec::Readdeltaf(const std::string &deltafFile)
{

    std::ifstream fin(deltafFile);
    char buf[256];
    double dump;
    double df = 0.0;
    TLENGTH = 150;
    MULENGTH = 100;
    T0 = 0.05;
    MU0 = 0.0;
    MU_STEP = 0.007892;
    T_STEP = 0.001;
    if (fin.is_open() ){
        
	while(fin.good() ){
	    
		fin >> dump >> dump >> df;
	    
	    if(fin.eof() ) break;
	   
        h_deltaf_qmu.push_back(df);
	    
	}
	fin.close();
    std::cout<<"#deltaf size="<< h_deltaf_qmu.size()<<std::endl;

    }
    else{
        std::cerr<<"#Can't open delta_f_qmu data file!\n";
        exit(0);
    }
    


}



void Spec::initializeCL(){
   try {

       cl_int device_type= CL_DEVICE_TYPE_CPU;
#ifdef USE_DEVICE_GPU
        device_type = CL_DEVICE_TYPE_GPU;
#endif
       std::cout<< CL_DEVICE_TYPE_CPU << "  "<< CL_DEVICE_TYPE_GPU<<" "<<device_type<<std::endl;
       context = CreateContext(device_type);

       devices = context.getInfo<CL_CONTEXT_DEVICES>();

       for (std::vector<cl::Device>::size_type i=0; i!=devices.size(); i++ ){
           std::cout <<"#" <<devices[i].getInfo<CL_DEVICE_NAME>()<<std::endl;
           std::cout<<"#Max compute units = "<< devices[i].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()<<std::endl;
       }

       std::stringstream compile_options;
       std::string dev_vendor = devices[gpu_id_].getInfo<CL_DEVICE_VENDOR>();
       std::cout<<"#using device = "<<dev_vendor<<std::endl;

       int LenVector = devices[gpu_id_].getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>();
       std::cout <<"#preferred vector width float = "<< LenVector<<std::endl; 
       compile_options << "-I../src"<<" ";
       compile_options << "-I../inc"<<" ";
 

       if(sizeof(cl_real) == 4){
           compile_options << "-D USE_SINGLE_PRECISION"<<" ";
       }
       compile_options << "-D SizeSF="<<SizeSF<<" "; 
       compile_options << "-D SizePID="<<SizePID<<" ";
       compile_options << "-D TLENGTH=" <<TLENGTH<< " ";
       compile_options << "-D MULENGTH=" <<MULENGTH<< " ";
       compile_options << "-D T0=" <<T0<< " ";
       compile_options << "-D MU0=" <<MU0<< " ";
       compile_options << "-D MU_STEP=" <<MU_STEP<< " ";
       compile_options << "-D T_STEP=" <<T_STEP<< " ";
       
       if(flag_mu_pce)
       {
        compile_options << "-D FLAG_MU_PCE" <<" ";

       }

       if(vorticity_on_)
       {
	   compile_options << "-D VORTICITY_ON" << " ";

       }
        std::cout<< compile_options.str()<<std::endl;
       

       queue = cl::CommandQueue( context, devices[gpu_id_], CL_QUEUE_PROFILING_ENABLE );

       


        AddProgram("../src/kernel_sample_particle.cl");

        BuildPrograms(0, compile_options.str().c_str());
        
        get_dntot = cl::Kernel(programs.at(0),"get_dntot");
        classify_ptc = cl::Kernel(programs.at(0),"classify_ptc");
        sample_lighthadron = cl::Kernel(programs.at(0),"sample_lighthadron");
        sample_heavyhadron = cl::Kernel(programs.at(0),"sample_heavyhadron");
	sample_vorticity = cl::Kernel(programs.at(0),"sample_vorticity");
    get_poisson = cl::Kernel(programs.at(0),"get_poisson");

   
       

       d_SF = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SizeSF*sizeof(cl_real8), h_SF.data()); //global memory
       d_nmtp = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SizeSF*sizeof(cl_real4), h_nmtp.data());
       d_txyz = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SizeSF*sizeof(cl_real4), h_txyz.data());
       if (!h_pi.empty()) {
            d_pi = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 10*SizeSF*sizeof(cl_real), h_pi.data()); //global memory
       }

        if (!h_qb.empty()) {
            d_qb = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4*SizeSF*sizeof(cl_real), h_qb.data()); //global memory
        }
        d_HadronInfo = cl::Buffer( context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR, 5*SizePID*sizeof(cl_real) , h_HadronInfo.data()); //constant memory
        
        if(!h_deltaf_qmu.empty()){
            d_deltaf_qmu = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, TLENGTH*MULENGTH*sizeof(cl_real), h_deltaf_qmu.data()); //global memory

        }


	if(!h_omega_th.empty()){
            d_omega_th = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 6*SizeSF*sizeof(cl_real), h_omega_th.data()); //global memory
        }

        if(!h_omega_shear1.empty()){
            d_omega_shear1 = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 16*SizeSF*sizeof(cl_real), h_omega_shear1.data()); //global memory
        }

        if(!h_omega_shear2.empty()){
            d_omega_shear2 = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4*SizeSF*sizeof(cl_real), h_omega_shear2.data()); //global memory
        }


        if(!h_omega_accT.empty()){
            d_omega_accT = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 6*SizeSF*sizeof(cl_real), h_omega_accT.data()); //global memory
        }

        if(!h_omega_chemical.empty()){
            d_omega_chemical = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 6*SizeSF*sizeof(cl_real), h_omega_chemical.data()); //global memory
        }

        

        for (int i = 0; i < SizeSF; i++) {
            h_hptc.push_back( (cl_real4){ 0.0 , 0.0 , 0.0 , 0.0} );
            h_lptc.push_back( (cl_real4){ 0.0 , 0.0 , 0.0 , 0.0} );
            h_ptc.push_back((cl_real8) {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } );
            h_poisson.push_back((cl_int2) {0, 0 } );
            //test
            h_dntot.push_back(0.0); 
	    }
       
    

    }
    catch (cl::Error & err){
        std::cerr<<"Error: "<< err.what()<<"("<<err.err()<<")\n";
    }
}



cl::Context Spec::CreateContext(const cl_int& device_type)
{

    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);
    if(platforms.size()==0){
        std::cerr<<"NO PLATFORM FOUND!!! \n";
        exit(-1);
    }
    else{
        for(int i = 0 ;i < platforms.size(); i ++){
            std::vector<cl::Device> supportDevices;
            platforms.at(i).getDevices(CL_DEVICE_TYPE_ALL, &supportDevices);
            for (int j = 0; j< supportDevices.size();j++){
                if( supportDevices.at(j).getInfo<CL_DEVICE_TYPE>() == device_type )
                {
                    std::cout<<"#Found device "<<supportDevices[j].getInfo<CL_DEVICE_NAME>()<<" on platform "<< i <<std::endl;
                    cl_context_properties properties[] = 
                    { CL_CONTEXT_PLATFORM, (cl_context_properties) (platforms.at(i))(), 0 };
                    return cl::Context(device_type, properties);
                }
            }
        }
    }
    std::cerr<<"No platform support device type "<< device_type<<std::endl;
    exit(-1);


}

void Spec::AddProgram(const char * fname){
    std::ifstream kernelFile( fname );
    if( !kernelFile.is_open() ) std::cerr<<"Open "<<fname << " failed!"<<std::endl;

    std::string sprog( std::istreambuf_iterator<char> (kernelFile), (std::istreambuf_iterator<char> ()) );
    cl::Program::Sources prog(1, std::make_pair(sprog.c_str(), sprog.length()));
    programs.push_back( cl::Program( context, prog ) );


}

void Spec::BuildPrograms( int i, const char * compile_options )
{ //// build programs and output the compile error if there is
    //for(std::vector<cl::Program>::size_type i=0; i!=programs.size(); i++)
    {
        try{
            programs.at(i).build(devices, compile_options);
        }
        catch(cl::Error & err){
            std::cerr<<err.what()<<"("<<err.err()<<")\n"<< programs.at(i).getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[gpu_id_]);
        }
    }

}



void Spec::SetPathOut( const std::string & path )
{
    DataPath = path;
    std::cout<<"path="<<DataPath<<std::endl;
}



Spec::~Spec(){

}


void Spec::Sample_particle(){
    initializeCL();

    cl_int BlockSize = BSZ;
    cl_int Size = BlockSize*NBlocks;


    //cl::NDRange globalSize = cl::NDRange( Size );
    //cl::NDRange localSize = cl::NDRange( 1 );
     cl::NDRange globalSize = cl::NDRange(BlockSize*NBlocks);
     cl::NDRange localSize = cl::NDRange( BlockSize );

    
    long long rnd_init = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937_64 generator(rnd_init);
	for (int i = 0; i < NBlocks*BSZ; i++) {
            h_seed.push_back( generator() );

	} 


    d_hptc  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SizeSF*sizeof(cl_real4), h_hptc.data() );
    d_lptc  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SizeSF*sizeof(cl_real4), h_lptc.data() );
    d_ptc  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SizeSF*sizeof(cl_real8), h_ptc.data() );
    d_dntot  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, SizeSF*sizeof(cl_real), h_dntot.data() );
    
    for (int ii = 0; ii < NBlocks*BSZ; ii++) {
        h_seed[ii]=generator();
        
	}

    d_seed  = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NBlocks*BSZ*sizeof(cl_ulong), h_seed.data() );
    get_dntot.setArg(0, d_SF);
    get_dntot.setArg(1, d_nmtp);
    get_dntot.setArg(2, d_txyz);
    get_dntot.setArg(3, d_HadronInfo);
    get_dntot.setArg(4, d_seed);
    get_dntot.setArg(5, d_dntot);

    double excution_time_step4 = 0.0;
    cl::Event event4;

    queue.enqueueNDRangeKernel(get_dntot, cl::NullRange, \
       globalSize, localSize, NULL, &event4); //256*64, 256
    event4.wait();
    excution_time_step4 += excutionTime(event4);
    queue.enqueueReadBuffer(d_dntot,CL_TRUE, 0, SizeSF* sizeof(cl_real),h_dntot.data() );
    double mean_size2 = 0.0; 
    double NTOT = 0.0;
    

    
    std::stringstream fname_particle_list;
    fname_particle_list << DataPath  <<"/mc_particle_list0";
    std::ofstream fpmag(fname_particle_list.str());

    if (model_ == "SMASH" && !quark_level){

        fpmag << "#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge" <<std::endl;
        fpmag << "# Units: fm fm fm fm GeV GeV GeV GeV GeV none none none" <<std::endl;
    }
    
    std::stringstream fname_particle_vorticity;
    fname_particle_vorticity << DataPath  <<"/mc_particle_vorticity_list0";
    std::ofstream fvorticity(fname_particle_vorticity.str());
    fvorticity << "# spin_vector th^{mu} th^{mu} th^{mu} th^{mu}, mu = t,x,y,z , lab frame"<<std::endl;
 
   
 
    for ( int i = 0 ; i< nsampling_  ; i++){


   


    
    double etot = 0.0; 


///////////////////////////////////////////

  for (int ii = 0; ii < NBlocks*BSZ; ii++) {
           h_seed[ii]=generator() ;
       }
   
   d_seed  = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NBlocks*BSZ*sizeof(cl_ulong), h_seed.data() );
   d_poisson = cl::Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 2*SizeSF*sizeof(cl_int), h_poisson.data()); //global memory
    cl_int h_Npoisson =0;
    cl::Buffer d_Npoisson = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &h_Npoisson);
     get_poisson.setArg(0, d_poisson);
     get_poisson.setArg(1, d_seed);
     get_poisson.setArg(2, d_dntot);
     get_poisson.setArg(3, d_Npoisson);
     cl::Event event0;
   queue.enqueueNDRangeKernel(get_poisson, cl::NullRange, \
       globalSize, localSize, NULL, &event0); //256*64, 256
   event0.wait();
   queue.enqueueReadBuffer(d_Npoisson,CL_TRUE, 0, sizeof(cl_uint),&h_Npoisson );
   
   


      






/////////////////////////////////////////

      //h_poisson.resize(0);
      //double tst1 = 0;
      //double size2 = 0;
      //for(int ii = 0 ; ii < SizeSF ; ii++){
      //    std::poisson_distribution<int> poisson(h_dntot[ii]);
      //    tst1 += h_dntot[ii];
      //    cl_int Ni = poisson(generator);
      //  size2 += Ni;
      //    if(Ni != 0)
      //    { 
    

      //        h_poisson.push_back( (cl_int2){Ni,ii} );
      //    }
    
      //}
      //h_Npoisson = h_poisson.size();
      //mean_size2 += size2;
      //d_poisson  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_Npoisson*sizeof(cl_int2), h_poisson.data() );

      //std::cout<<h_Npoisson <<" " << size2<<std::endl;


/////////////////////////////////////////////////
    
    
    


    for (int ii = 0; ii < NBlocks*BSZ; ii++) {
            h_seed[ii]=generator() ;
	}

    d_seed  = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NBlocks*BSZ*sizeof(cl_ulong), h_seed.data() );

    h_Nhptc = 0;
    h_Nlptc = 0;
    
    d_Nhptc = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &h_Nhptc);
    d_Nlptc = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_int), &h_Nlptc);
    
    classify_ptc.setArg(0, d_SF);
    classify_ptc.setArg(1, d_nmtp);
    classify_ptc.setArg(2, d_txyz);
    classify_ptc.setArg(3, d_HadronInfo);
    classify_ptc.setArg(4, d_seed);
    classify_ptc.setArg(5, d_hptc);
    classify_ptc.setArg(6, d_lptc);
    classify_ptc.setArg(7, d_Nhptc);
    classify_ptc.setArg(8, d_Nlptc);
    classify_ptc.setArg(9, d_poisson);
    classify_ptc.setArg(10, d_dntot);
    //classify_ptc.setArg(11, size1);
    classify_ptc.setArg(11, h_Npoisson);
    
    
    double excution_time_step1 = 0.0;
    cl::Event event;
    queue.enqueueNDRangeKernel(classify_ptc, cl::NullRange, \
        globalSize, localSize, NULL, &event); //256*64, 256
    event.wait();
    excution_time_step1 += excutionTime(event);
    //std::cout<<"classify ptc finished! "<< excution_time_step1<<std::endl;
    
    for (int ii = 0; ii < NBlocks*BSZ; ii++) {
		    double mm = generator();     
            h_seed[ii]=generator() ;

	}

    d_seed  = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NBlocks*BSZ*sizeof(cl_ulong), h_seed.data() );
    
    sample_heavyhadron.setArg(0, d_SF);
    sample_heavyhadron.setArg(1, d_nmtp);
    sample_heavyhadron.setArg(2, d_txyz);
    sample_heavyhadron.setArg(3, d_HadronInfo);
    sample_heavyhadron.setArg(4, d_pi);
    sample_heavyhadron.setArg(5, d_qb);
    sample_heavyhadron.setArg(6, d_deltaf_qmu);
    sample_heavyhadron.setArg(7, d_seed);
    sample_heavyhadron.setArg(8, d_hptc);
    sample_heavyhadron.setArg(9, d_Nhptc);
    sample_heavyhadron.setArg(10, d_ptc);

     
    double excution_time_step2 = 0.0;
    cl::Event event1;
    queue.enqueueNDRangeKernel(sample_heavyhadron, cl::NullRange, \
        globalSize, localSize, NULL, &event1); //256*64, 256
    event1.wait();
    excution_time_step2 += excutionTime(event1);
    //std::cout<<"sample_heavy_hadron finished! "<< excution_time_step2<<std::endl;
    

    for (int ii = 0; ii < NBlocks*BSZ; ii++) {
		    double mm = generator();     
            h_seed[ii]=generator() ;

	}

    d_seed  = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, NBlocks*BSZ*sizeof(cl_ulong), h_seed.data() );

    sample_lighthadron.setArg(0, d_SF);
    sample_lighthadron.setArg(1, d_nmtp);
    sample_lighthadron.setArg(2, d_txyz);
    sample_lighthadron.setArg(3, d_HadronInfo);
    sample_lighthadron.setArg(4, d_pi);
    sample_lighthadron.setArg(5, d_qb);
    sample_lighthadron.setArg(6, d_deltaf_qmu);
    sample_lighthadron.setArg(7, d_seed);
    sample_lighthadron.setArg(8, d_lptc);
    sample_lighthadron.setArg(9, d_Nlptc);
    sample_lighthadron.setArg(10, d_Nhptc);
    sample_lighthadron.setArg(11, d_ptc);


     
    double excution_time_step3 = 0.0;
    cl::Event event2;
    queue.enqueueNDRangeKernel(sample_lighthadron, cl::NullRange, \
        globalSize, localSize, NULL, &event2); //256*64, 256
    event2.wait();
    excution_time_step3 += excutionTime(event2);



    double t_tot = excution_time_step1+excution_time_step2+excution_time_step3;//+excution_time_step4;

    //std::cout<<"Finish event " << i <<" cost " << excution_time_step3<< " s "<< std::endl; 
    std::cout<<"Finish event " << i <<" cost " << t_tot<< " s "<< std::endl; 
    //std::cout<<"Finish event " << i << std::endl; 
    

    
   





    queue.enqueueReadBuffer(d_hptc,CL_TRUE, 0, SizeSF* sizeof(cl_real4),h_hptc.data() );
    queue.enqueueReadBuffer(d_Nhptc,CL_TRUE, 0, sizeof(cl_uint),&h_Nhptc );

    queue.enqueueReadBuffer(d_lptc,CL_TRUE, 0, SizeSF* sizeof(cl_real4),h_lptc.data() );
    queue.enqueueReadBuffer(d_Nlptc,CL_TRUE, 0, sizeof(cl_uint),&h_Nlptc );
    
    queue.enqueueReadBuffer(d_ptc,CL_TRUE, 0, SizeSF*sizeof(cl_real8),h_ptc.data() );
    //std::cout<<"Finish event " << i <<" cost " << h_Nlptc<< " s "<<h_Nhptc <<" "<<h_Npoisson<< std::endl; 

    if (vorticity_on_){


    for (int i = 0; i < h_Nhptc + h_Nlptc; i++) {
        h_spin_th.push_back( (cl_real4){ 0.0 , 0.0 , 0.0 , 0.0} );
        h_spin_shear.push_back( (cl_real4){ 0.0 , 0.0 , 0.0 , 0.0} );
        h_spin_accT.push_back( (cl_real4){ 0.0 , 0.0 , 0.0 , 0.0} );
        h_spin_chemical.push_back( (cl_real4){ 0.0 , 0.0 , 0.0 , 0.0} );
        }



    d_spin_th  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (h_Nhptc + h_Nlptc)*sizeof(cl_real4), h_spin_th.data() );
    d_spin_shear  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (h_Nhptc + h_Nlptc)*sizeof(cl_real4), h_spin_shear.data() );
    d_spin_accT  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (h_Nhptc + h_Nlptc)*sizeof(cl_real4), h_spin_accT.data() );
    d_spin_chemical  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (h_Nhptc + h_Nlptc)*sizeof(cl_real4), h_spin_chemical.data() );



    sample_vorticity.setArg(0, d_SF);
    sample_vorticity.setArg(1, d_nmtp);
    sample_vorticity.setArg(2, d_HadronInfo);
    sample_vorticity.setArg(3, d_omega_th);
    sample_vorticity.setArg(4, d_omega_shear1);
    sample_vorticity.setArg(5, d_omega_shear2);
    sample_vorticity.setArg(6, d_omega_accT);
    sample_vorticity.setArg(7, d_omega_chemical);
    sample_vorticity.setArg(8, d_lptc);
    sample_vorticity.setArg(9, d_hptc);
    sample_vorticity.setArg(10, d_Nlptc);
    sample_vorticity.setArg(11, d_Nhptc);
    sample_vorticity.setArg(12, d_spin_th);
    sample_vorticity.setArg(13, d_spin_shear);
    sample_vorticity.setArg(14, d_spin_accT);
    sample_vorticity.setArg(15, d_spin_chemical);
    sample_vorticity.setArg(16, d_ptc);

    cl::Event event3;
    queue.enqueueNDRangeKernel(sample_vorticity, cl::NullRange, \
       globalSize, localSize, NULL, &event3); //256*64, 256
    event3.wait();


    queue.enqueueReadBuffer(d_spin_th,CL_TRUE, 0, (h_Nhptc + h_Nlptc)* sizeof(cl_real4),h_spin_th.data() );
    queue.enqueueReadBuffer(d_spin_shear,CL_TRUE, 0, (h_Nhptc + h_Nlptc)* sizeof(cl_real4),h_spin_shear.data() );
    queue.enqueueReadBuffer(d_spin_accT,CL_TRUE, 0, (h_Nhptc + h_Nlptc)* sizeof(cl_real4),h_spin_accT.data() );
    queue.enqueueReadBuffer(d_spin_chemical,CL_TRUE, 0, (h_Nhptc + h_Nlptc)* sizeof(cl_real4),h_spin_chemical.data() );

    fvorticity << "# event "<<i<<std::endl;
    for(int ii = 0 ; ii <(h_Nhptc + h_Nlptc) ; ii ++){
    if(!std::isnan(h_ptc.at(ii).s[0]) )
         {
        fvorticity << std::setprecision(9);
        fvorticity << h_spin_th[ii].s[0]<<" "<<h_spin_th[ii].s[1]<<" "<<h_spin_th[ii].s[2]<<" "<<h_spin_th[ii].s[3]<<" "
                   << h_spin_shear[ii].s[0]<<" "<<h_spin_shear[ii].s[1]<<" "<<h_spin_shear[ii].s[2]<<" "<<h_spin_shear[ii].s[3]<<" "
                   << h_spin_accT[ii].s[0]<<" "<<h_spin_accT[ii].s[1]<<" "<<h_spin_accT[ii].s[2]<<" "<<h_spin_accT[ii].s[3]<<" "
                   << h_spin_chemical[ii].s[0]<<" "<<h_spin_chemical[ii].s[1]<<" "<<h_spin_chemical[ii].s[2]<<" "<<h_spin_chemical[ii].s[3]<<std::endl;
         }

    }



    }





     
    if(model_ == "SMASH" && !quark_level)
    {
        fpmag << "# event "<<i<<std::endl;
    }
    
    int ptc_id = 0;

    std::vector<Particle> output_particle;
    Particle tem_ptc;
    for(int ii = 0 ; ii < (h_Nhptc); ii ++  )
    {
        int nid = h_hptc.at(ii).s[1];
        cl_real mass = h_HadronInfo[nid*5 + 0];
        int pdgcode = particles.at(nid).monval;
        
    
        if(!std::isnan(h_ptc.at(ii).s[0]) )
        {
    
        
        
        tem_ptc.pos.s[0] = h_ptc.at(ii).s[4];
        tem_ptc.pos.s[1] = h_ptc.at(ii).s[5];
        tem_ptc.pos.s[2] = h_ptc.at(ii).s[6];
        tem_ptc.pos.s[3] = h_ptc.at(ii).s[7];

        tem_ptc.mass = mass;

        tem_ptc.mon.s[0] = h_ptc.at(ii).s[0]; 
        tem_ptc.mon.s[1] = h_ptc.at(ii).s[1];
        tem_ptc.mon.s[2] = h_ptc.at(ii).s[2];
        tem_ptc.mon.s[3] = h_ptc.at(ii).s[3];
        tem_ptc.pdgcode = pdgcode;
        tem_ptc.charge = particles.at(nid).charge;
        output_particle.push_back(tem_ptc);
        ptc_id ++;
        etot += h_ptc.at(ii).s[0];

        }
        
    }

    for(int ii = 0 ; ii < (h_Nlptc); ii ++  )
    {
        int nid = h_lptc.at(ii).s[1];
        cl_real mass =h_HadronInfo[nid*5+0];
        int pdgcode = particles.at(nid).monval;

        if(! std::isnan(h_ptc.at(ii+h_Nhptc).s[0]) ){
        
        
        

        tem_ptc.pos.s[0] = h_ptc.at(ii+h_Nhptc).s[4];
        tem_ptc.pos.s[1] = h_ptc.at(ii+h_Nhptc).s[5];
        tem_ptc.pos.s[2] = h_ptc.at(ii+h_Nhptc).s[6];
        tem_ptc.pos.s[3] = h_ptc.at(ii+h_Nhptc).s[7];

        tem_ptc.mass = mass;


        tem_ptc.mon.s[0] = h_ptc.at(ii+h_Nhptc).s[0]; 
        tem_ptc.mon.s[1] = h_ptc.at(ii+h_Nhptc).s[1];
        tem_ptc.mon.s[2] = h_ptc.at(ii+h_Nhptc).s[2];
        tem_ptc.mon.s[3] = h_ptc.at(ii+h_Nhptc).s[3];
        tem_ptc.pdgcode = pdgcode;
        tem_ptc.charge = particles.at(nid).charge;
        output_particle.push_back(tem_ptc);
        
        etot += h_ptc.at(ii+h_Nhptc).s[0];
        ptc_id++;
        
        }
        
    }

    NTOT+= h_Nhptc + h_Nlptc;

    


    // std::stringstream fname_spectator;
    // fname_spectator << DataPath  <<"/spectators.dat";
    // std::ifstream fspectator(fname_spectator.str());
    // double tp,xp,yp,zp,pmass,pp0,pp1,pp2,pp3,ppid,dumpy,pcharged;
    // if(fspectator.is_open()){
    //    while(fspectator.good()){
    //        fspectator>> tp >> xp >> yp >> zp
    //                  >> pmass >> pp0 >> pp1>>
    //                  pp2>>pp3>>ppid>>dumpy>>pcharged;
    //        if(fspectator.eof()) break;
    //     //   fpmag << std::setprecision(9);

    //     // fpmag << tp <<" "<< xp <<" "<< yp <<" "<< zp;
         
    //     // fpmag << std::setprecision(9);
    //     // fpmag << " "<< pmass << " "<< pp0
    //     //      << " "<< pp1 << " "<< pp2
    //     //      << " "<< pp3 << " "<< int(ppid)
    //     //      << " "<< ptc_id+1
    //     //      << " "<< int(pcharged) <<std::endl;
    //     tem_ptc.pos.s[0] = tp;
    //     tem_ptc.pos.s[1] = xp;
    //     tem_ptc.pos.s[2] = yp;
    //     tem_ptc.pos.s[3] = zp;

    //     tem_ptc.mass = pmass;

    //     tem_ptc.mon.s[0] = pp0; 
    //     tem_ptc.mon.s[1]= pp1;
    //     tem_ptc.mon.s[2] = pp2;
    //     tem_ptc.mon.s[3] = pp3;
    //     tem_ptc.pdgcode =int(ppid);
    //     tem_ptc.charge = int(pcharged);
    //     output_particle.push_back(tem_ptc);

    //    ptc_id++;
    //    }
    //    fspectator.close();
    // }
    // else{

    //    //std::cout<<" No Spectators file"<<std::endl;
    // }



    // std::stringstream fname_corona;
    // fname_corona << DataPath  <<"/corona_particles.dat";
    // std::ifstream fcorona(fname_corona.str());

    // if(fcorona.is_open()){
    //    while(fcorona.good()){
    //        fcorona>> tp >> xp >> yp >> zp
    //                  >> pmass >> pp0 >> pp1>>
    //                  pp2>>pp3>>ppid>>dumpy>>pcharged;
    //        if(fcorona.eof()) break;
    //     // fpmag << std::setprecision(9);

    //     // fpmag << tp <<" "<< xp <<" "<< yp <<" "<< zp;
         
    //     // fpmag << std::setprecision(9);
    //     // fpmag << " "<< pmass << " "<< pp0
    //     //      << " "<< pp1 << " "<< pp2
    //     //      << " "<< pp3 << " "<< int(ppid)
    //     //      << " "<< ptc_id+1
    //     //      << " "<< int(pcharged) <<std::endl;

    //     tem_ptc.pos.s[0] = tp;
    //     tem_ptc.pos.s[1] = xp;
    //     tem_ptc.pos.s[2] = yp;
    //     tem_ptc.pos.s[3] = zp;

    //     tem_ptc.mass = pmass;

    //     tem_ptc.mon.s[0] = pp0; 
    //     tem_ptc.mon.s[1] = pp1;
    //     tem_ptc.mon.s[2] = pp2;
    //     tem_ptc.mon.s[3] = pp3;
    //     tem_ptc.pdgcode =int(ppid);
    //     tem_ptc.charge = int(pcharged);
    //     output_particle.push_back(tem_ptc);


    //     ptc_id++;
    //    }

    //    fcorona.close();

    // }
    // else{

    //     //std::cout<<" No corona particle file"<<std::endl;
    // }

    if (model_ == "URQMD"){
        
        if(quark_level){
            int Nptg1p5 = 0;

            for (std::vector<Particle>::size_type outi= 0 ; outi<output_particle.size();outi++){


                double  pT_tem = sqrt(output_particle[outi].mon.s[1]*output_particle[outi].mon.s[1]
                         +output_particle[outi].mon.s[2]*output_particle[outi].mon.s[2]);
                
                if(pT_tem > 1.5) Nptg1p5++;
            }


            fpmag<<Nptg1p5<<std::endl;
            for (std::vector<Particle>::size_type outi= 0 ; outi<output_particle.size();outi++){
                double  pT_tem = sqrt(output_particle[outi].mon.s[1]*output_particle[outi].mon.s[1]
                         +output_particle[outi].mon.s[2]*output_particle[outi].mon.s[2]);
                double p0_tem = sqrt(output_particle[outi].mon.s[1]*output_particle[outi].mon.s[1]
                             +output_particle[outi].mon.s[2]*output_particle[outi].mon.s[2]
                             +output_particle[outi].mon.s[3]*output_particle[outi].mon.s[3]
                             +output_particle[outi].mass*output_particle[outi].mass);

                if(pT_tem > 1.5) 
                {
                    
                fpmag <<std::fixed<< std::setprecision(9)<< output_particle[outi].pdgcode
                <<" "<<output_particle[outi].mon.s[1]<<" "<<output_particle[outi].mon.s[2]
                <<" "<<output_particle[outi].mon.s[3]<<" "<<p0_tem
                <<" "<<output_particle[outi].pos.s[1]<<" "<<output_particle[outi].pos.s[2]
                <<" "<<output_particle[outi].pos.s[3]<<" "<<output_particle[outi].pos.s[0] <<std::endl; 
                }
            }

        }


        else{
            fpmag<<"# "<<output_particle.size()<<std::endl;
            for (std::vector<Particle>::size_type outi= 0 ; outi<output_particle.size();outi++)
            {
                
                
                fpmag << std::fixed << std::setprecision(9) << output_particle[outi].pdgcode<<" "<<output_particle[outi].pos.s[0]
                <<" "<<output_particle[outi].pos.s[1]<<" "<<output_particle[outi].pos.s[2]
                <<" "<<output_particle[outi].pos.s[3];
        
                
        
                double  p0_tem = sqrt(output_particle[outi].mon.s[1]*output_particle[outi].mon.s[1]
                                 +output_particle[outi].mon.s[2]*output_particle[outi].mon.s[2]
                                 +output_particle[outi].mon.s[3]*output_particle[outi].mon.s[3]
                                 +output_particle[outi].mass*output_particle[outi].mass);
                
                fpmag <<std::fixed<< std::setprecision(9) <<" "<<p0_tem
                <<" "<<output_particle[outi].mon.s[1]<<" "<<output_particle[outi].mon.s[2]
                <<" "<<output_particle[outi].mon.s[3]<<std::endl;
            }
        }

    }


    if (model_ == "SMASH"){

        if(quark_level){


            int Nptg1p5 = 0;

            for (std::vector<Particle>::size_type outi= 0 ; outi<output_particle.size();outi++){


                double  pT_tem = sqrt(output_particle[outi].mon.s[1]*output_particle[outi].mon.s[1]
                         +output_particle[outi].mon.s[2]*output_particle[outi].mon.s[2]);
                
                if(pT_tem > 1.5) Nptg1p5++;
            }


            fpmag<<Nptg1p5<<std::endl;
            for (std::vector<Particle>::size_type outi= 0 ; outi<output_particle.size();outi++){
                
                
                double  pT_tem = sqrt(output_particle[outi].mon.s[1]*output_particle[outi].mon.s[1]
                         +output_particle[outi].mon.s[2]*output_particle[outi].mon.s[2]);
                
                double p0_tem = sqrt(output_particle[outi].mon.s[1]*output_particle[outi].mon.s[1]
                             +output_particle[outi].mon.s[2]*output_particle[outi].mon.s[2]
                             +output_particle[outi].mon.s[3]*output_particle[outi].mon.s[3]
                             +output_particle[outi].mass*output_particle[outi].mass);
          

                if(pT_tem > 1.5) 
                {
                    
                fpmag << std::fixed << std::setprecision(9)<< output_particle[outi].pdgcode
                <<" "<<output_particle[outi].mon.s[1]<<" "<<output_particle[outi].mon.s[2]
                <<" "<<output_particle[outi].mon.s[3]<<" "<<p0_tem
                <<" "<<output_particle[outi].pos.s[1]<<" "<<output_particle[outi].pos.s[2]
                <<" "<<output_particle[outi].pos.s[3]<<" "<<output_particle[outi].pos.s[0] <<std::endl; 
                }
            }

        }
    

        
        else{

        for (std::vector<Particle>::size_type outi= 0 ; outi<output_particle.size();outi++){
           
            fpmag << std::fixed<< std::setprecision(9) <<output_particle[outi].pos.s[0]<<" "<<output_particle[outi].pos.s[1]
            <<" "<<output_particle[outi].pos.s[2]<<" "<<output_particle[outi].pos.s[3]
            <<" ";
            fpmag<<std::setprecision(5)<< output_particle[outi].mass;
    
            double p0_tem = sqrt(output_particle[outi].mon.s[1]*output_particle[outi].mon.s[1]
                             +output_particle[outi].mon.s[2]*output_particle[outi].mon.s[2]
                             +output_particle[outi].mon.s[3]*output_particle[outi].mon.s[3]
                             +output_particle[outi].mass*output_particle[outi].mass);
          
    
            fpmag << std::fixed << std::setprecision(9)<<" "<<p0_tem
            <<" "<<output_particle[outi].mon.s[1]<<" "<<output_particle[outi].mon.s[2]
            <<" "<<output_particle[outi].mon.s[3];
            fpmag<<std::setprecision(1)<<" "<<output_particle[outi].pdgcode<<" "<<outi<<" "<<output_particle[outi].charge <<std::endl;
     

     
    
        }

        fpmag << "# event "<<i<<" end"<<std::endl;

        }


    }


    
    

    }

    fpmag.close();
    fvorticity.close();

    
    



}
