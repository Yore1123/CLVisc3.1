#include<helper.h>


#pragma OPENCL EXTENSION cl_amd_printf : enable

inline real create_longitudinal_profile(real etas){
   if (fabs(etas) > Eta_flat){
       return exp(-(fabs(etas)- Eta_flat)*(fabs(etas) - Eta_flat)/(2.0*Eta_gw*Eta_gw) );
   }
   else{
       return 1.0f;
   }


}




__kernel void from_sd_to_ed(__global real *d_s,
	                    __global real4 *d_ev1,
			    __global real * d_nb1,
			    __global real * eos_table
			    ){

    int i = get_global_id(0);
    int j = get_global_id(1);
    int ind = i*NY + j;
    real local_s = d_s[ind];
    real local_nb = 0.0f;
    real local_ed0 = eos_s2ed(local_s,local_nb,eos_table);

    for (int k = 0 ; k < NZ; k++){
	real etas = (k - NZ/2)*DZ; 
	real hetas = create_longitudinal_profile(etas);
	real local_ed = local_ed0*hetas;

	real local_ed1 = 1e-15*hbarc;
	real local_nb1 = 0.0f;
        real local_s  = eos_s(local_ed1,local_nb1,eos_table);


	d_ev1[i*NY*NZ + j*NZ + k] = (real4)(local_ed, 0.0f, 0.0f, 0.0f );
#ifdef BARYON_ON
        d_nb1[i*NY*NZ+j*NZ + k] = 0.0f;
#else
        d_nb1[i*NY*NZ+j*NZ + k] = 0.0f;
#endif


    }






}
