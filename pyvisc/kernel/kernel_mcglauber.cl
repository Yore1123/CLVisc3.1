#include<helper.h>

#pragma OPENCL EXTENSION cl_amd_printf : enable


inline real create_longitudinal_profile(real etas){
   if (fabs(etas) > Eta_flat_c){
       return exp(-(fabs(etas)- Eta_flat_c)*(fabs(etas) - Eta_flat_c)/(2.0*Eta_gw_c*Eta_gw_c) );
   }
   else{
       return 1.0f;
   }


}


inline real Penvelope_s(real eta)
{
   real gamma = SQRTS/(2.0f*0.938f);
   //real gamma = SQRTS/(2.0f);
   real eta_max = atanh(sqrt(gamma*gamma-1.0f)/gamma);

   real heta = 0.0f; 
   heta = step( 0.0f, eta_max-fabs(eta) )*( 1.0f + eta/eta_max )\
        * ( step(0.0f,fabs(eta)-Eta0_s)\
	* exp(-pow(fabs(eta)-Eta0_s,2.0f)/(2.0f*pow(Eta_gw,2.0f)))\
	+ step(0.0f,Eta0_s-fabs(eta)));
    return heta;
}


inline real Menvelope_s(real eta)
{
   real gamma = SQRTS/(2.0f*0.938f);
   //real gamma = SQRTS/(2.0f);
   real eta_max = atanh(sqrt(gamma*gamma-1.0f)/gamma);

    
   real heta = 0.0f; 
   heta = step( 0.0f, eta_max-fabs(eta) )*( 1.0f - eta/ eta_max  )\
        * ( step(0.0f,fabs(eta)-Eta0_s)\
	* exp(-pow(fabs(eta)-Eta0_s,2.0f)/(2.0f*pow(Eta_gw,2.0f)))\
	+ step(0.0f,Eta0_s-fabs(eta)));
   return heta;
}



inline real Penvelope_nb(real eta)
{


   real eta_0 = fabs(Eta0_nb);
   real norm = 2.0/(sqrt(M_PI)*(Sigma0_m+Sigma0_p));
   real exp_arg=0.0;
   if (eta < eta_0){
       exp_arg = (eta - eta_0)/Sigma0_m;
   }
   else{
       exp_arg = (eta - eta_0)/Sigma0_p;
   }

   real heta = norm * exp(- exp_arg*exp_arg);
   return heta;

}

inline real Menvelope_nb(real eta)
{

    real eta_0 = -fabs(Eta0_nb);
    real norm = 2.0/(sqrt(M_PI)*(Sigma0_m+Sigma0_p));
    real exp_arg=0.0;
    if (eta < eta_0){
       exp_arg = (eta - eta_0)/Sigma0_p;
    }
    else{
       exp_arg = (eta - eta_0)/Sigma0_m;
    }

    real heta = norm * exp(- exp_arg*exp_arg);
    return heta;

}

__kernel void generate_s(__global real * d_tar_x,
	                     __global real * d_tar_y,
			            __global real * d_pro_x,
			            __global real * d_pro_y,
			            const int Ntar,
			            const int Npro,
                        __global real * d_nbinaryx,
                        __global real * d_nbinaryy,
                        const int Ncoll,
			            __global real4* d_ev1,
			            __global real * d_nb1,
                        __global real * d_Ta,
                        __global real * d_Tb,
			            const real Factor,
                          __global real* eos_table){
    
    int i = get_global_id(0);
    int j = get_global_id(1);
    real x = (i - NX/2)*DX;
    real y = (j - NY/2)*DY;
    int index = i*NY + j;
    //real sigma_r = 0.5f;
    real one_over_2pisigma = 1.0f/(2.0f*M_PI_F*sigma_r*sigma_r);
    real s_p = 0.0f;
    real s_m = 0.0f;
    real s_coll = 0.0f;
    for ( int mn = 0; mn < Ntar; mn++ ){
	real distence = (x - d_tar_x[mn])*(x - d_tar_x[mn])+(y - d_tar_y[mn])*(y - d_tar_y[mn]);
        s_m += one_over_2pisigma * exp(-distence/(2.0f*sigma_r*sigma_r));
    }

    for ( int mn = 0; mn < Npro; mn++ ){
	real distence = (x - d_pro_x[mn])*(x - d_pro_x[mn])+(y - d_pro_y[mn])*(y - d_pro_y[mn]);
        s_p += one_over_2pisigma * exp(-distence/(2.0f*sigma_r*sigma_r));
    }

     for ( int mn = 0; mn < Ncoll; mn++ ){
	real distence = (x - d_nbinaryx[mn])*(x - d_nbinaryx[mn])+(y - d_nbinaryy[mn])*(y - d_nbinaryy[mn]);
        s_coll += one_over_2pisigma * exp(-distence/(2.0f*sigma_r*sigma_r));
    }

    d_Ta[index] = s_m;
    d_Tb[index] = s_p;
    
 
    for (int k = 0 ; k < NZ ; k++){
        real etas = (k - NZ/2)*DZ;
        real seta_p = Penvelope_s( etas );
        real seta_m = Menvelope_s( etas );
        real nbeta_p = Penvelope_nb( etas );
        real nbeta_m = Menvelope_nb( etas );
        real s_density1 = (s_m*seta_m + s_p*seta_p)/(TAU0*1.0f);
        //real s_density = 0.5*(s_m+s_p)*(seta_m +seta_p)/(TAU0*1.0f);
	    real hetas = create_longitudinal_profile(etas);
        real s_density2 = s_coll*hetas/(TAU0*1.0f);

        //real s_density = 0.5*(s_m+s_p)*hetas/(TAU0*1.0f);
        real nb_density = (s_m*nbeta_m + s_p*nbeta_p )/(TAU0*1.0f);
#ifdef BARYON_ON
        d_nb1[i*NY*NZ+j*NZ + k] = nb_density;
#else
        d_nb1[i*NY*NZ+j*NZ + k] = 0.0;
#endif          
        int indx = i*NY*NZ+j*NZ + k;
        real s_density = Hwn*s_density1 + (1-Hwn)*s_density2; 
        s_density = eos_s2ed(Factor*s_density, nb_density, eos_table);
        real ed_min = 1e-11;
        s_density = max(ed_min,s_density);
	
        d_ev1[i*NY*NZ + j*NZ + k] = (real4)(s_density, 0.0f, 0.0f, 0.0f );


    }

}



__kernel void generate_s2(__global real * d_Ta,
	                  __global real * d_Tb,
			  __global real4* d_ev1,
			  __global real * d_nb1,
			  const real Factor,
                          __global real* eos_table){
    
    int i = get_global_id(0);
    int j = get_global_id(1);
    real x = (i - NX/2)*DX;
    real y = (j - NY/2)*DY;
    int index = i*NY + j;
    if (index < NX*NY)
    {
    real s_p = d_Ta[index];
    real s_m = d_Tb[index];
    
 
    for (int k = 0 ; k < NZ ; k++){
        real etas = (k - NZ/2)*DZ;
        real seta_p = Penvelope_s( etas );
        real seta_m = Menvelope_s( etas );
        real nbeta_p = Penvelope_nb( etas );
        real nbeta_m = Menvelope_nb( etas );
        real s_density = (s_m*seta_m + s_p*seta_p)/(TAU0*1.0f);
        real nb_density = (s_m*nbeta_m + s_p*nbeta_p )/(TAU0*1.0f);
        d_nb1[i*NY*NZ+j*NZ + k] = nb_density;
        s_density = eos_s2ed(Factor*s_density, nb_density, eos_table);


        real ed_min = 1e-11;
        s_density = max(ed_min,s_density);
        d_ev1[i*NY*NZ + j*NZ + k] = (real4)(s_density, 0.0f, 0.0f, 0.0f );


    }

    }
}


__kernel void generate_phi(__global real * d_tar_x,
	                   __global real * d_tar_y,
			   __global real * d_pro_x,
			   __global real * d_pro_y,
			   const int Ntar,
			   const int Npro,
			   const real Factor,
			   __global real * d_phi2_re,
			   __global real * d_phi2_im,
			   __global real * d_phi2_wt

			  ){
    
    int i = get_global_id(0);
    int j = get_global_id(1);
    real x = (i - NX/2)*DX;
    real y = (j - NY/2)*DY;

    //real sigma_r = 0.5f;
    real one_over_2pisigma = 1.0f/(2.0f*M_PI_F*sigma_r*sigma_r);
    real s_p = 0.0f;
    real s_m = 0.0f;
    for ( int mn = 0; mn < Ntar; mn++ ){
	real distence = (x - d_tar_x[mn])*(x - d_tar_x[mn])+(y - d_tar_y[mn])*(y - d_tar_y[mn]);
        s_m += one_over_2pisigma * exp(-distence/(2.0f*sigma_r*sigma_r));
    }

    for ( int mn = 0; mn < Npro; mn++ ){
	real distence = (x - d_pro_x[mn])*(x - d_pro_x[mn])+(y - d_pro_y[mn])*(y - d_pro_y[mn]);
        s_p += one_over_2pisigma * exp(-distence/(2.0f*sigma_r*sigma_r));
    }

    real s_density = (s_m + s_p)/(2.0f);

    real x2 = x*x;
    real y2 = y*y;
    real r2 = x2+y2;
    real xy = x*y;

    //d_phi2_re[i*NY+j] = -s_density*(x2 - y2);
    d_phi2_re[i*NY+j] = s_density*(x2 - y2);
    d_phi2_im[i*NY+j] = s_density*2.0*xy;
    d_phi2_wt[i*NY+j] = s_density*r2;

}

