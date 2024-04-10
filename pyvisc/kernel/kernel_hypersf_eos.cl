#include<helper.h>
#pragma OPENCL EXTENSION cl_amd_printf : enable

 __kernel void get_tpsmu_eos(
 	__global real4 * d_tpsmu,
 	__global real  * eos_table,
 	const int nline){


     int I = get_global_id(0);
     for(int i = get_global_id(0); i < nline; i= i+get_global_size(0))
     {
         real ed = d_tpsmu[i].s3;
         real nb = d_tpsmu[i].s0;
	 real pr = eos_P(ed, nb, eos_table);
         real mu = eos_mu(ed, nb, eos_table);
         real T = eos_T(ed, nb, eos_table);
         d_tpsmu[i] = (real4)(nb,mu,T,ed+pr);
         

     }

}


__kernel void reconst_hypersurface(
	        __global real *  d_J,
		__global real4 * d_nbmutp,
		__global real4 *  d_ev_old,
		__global real4 * d_Tmn,
		__global real* eos_table){

    int I = get_global_id(0);
    if ( I < NX*NY*NZ ) {
    
    real4 T0m = d_Tmn[I]; 


    real T00 = T0m.s0;
    real T01 = T0m.s1;
    real T02 = T0m.s2;
    real T03 = T0m.s3;
    real M = sqrt(T01*T01 + T02*T02 + T03*T03);
    

    
    real Jid0 = d_J[I];


    real ed_find = 0.0f;
    real nb_find = 0.0f;
    
    int state = 0; 

    rootFinding_newton(&ed_find, &nb_find,&state, T00, M, Jid0,eos_table);
    
    ed_find = max(acu, ed_find);
    real pr = eos_P(ed_find,nb_find, eos_table);
    real epv = max(acu, T00 + pr);
    d_ev_old[I] = (real4)(ed_find, T01/epv, T02/epv, T03/epv);
    
    real vsq = d_ev_old[I].s1*d_ev_old[I].s1+d_ev_old[I].s2*d_ev_old[I].s2
           + d_ev_old[I].s3*d_ev_old[I].s3;



    if ( vsq -1.0 > 0.0)
   {
     real scale = sqrt(0.999/vsq);
     d_ev_old[I].s1 = d_ev_old[I].s1*scale;
     d_ev_old[I].s2 = d_ev_old[I].s2*scale;
     d_ev_old[I].s3 = d_ev_old[I].s3*scale;
   }
    real mu = eos_mu(ed_find,nb_find, eos_table);
    real T = eos_T(ed_find, nb_find, eos_table);
    real entropy = eos_s(ed_find, nb_find, eos_table);


    d_nbmutp[I] = (real4) (nb_find, mu, T , pr);


    }

}



