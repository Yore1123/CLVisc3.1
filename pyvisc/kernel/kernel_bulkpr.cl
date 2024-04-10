#include<helper.h>


#pragma OPENCL EXTENSION cl_amd_printf : enable

constant real gmn[4][4] = {{1.0f, 0.0f, 0.0f, 0.0f},
                           {0.0f,-1.0f, 0.0f, 0.0f},
                           {0.0f, 0.0f,-1.0f, 0.0f},
                           {0.0f, 0.0f, 0.0f,-1.0f}};


__kernel void bulkpr_src_christoffel(
            __global real* d_Src,
            __global real* d_bulkpr1,
            const int step){

    int I = get_global_id(0);
    if(I < NX*NY*NZ) {
        if (step == 1) {
            d_Src[I]=0.0f;
        }
    }
}



__kernel void bulkpr_src_alongx(
             __global real* d_Src,
             __global real* d_bulkpr1,
	         __global real4* d_ev,
             __global real* d_nb,
             __global real* eos_table,
	     const real tau) {


    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    __local real bulkpr[NX+4];
    __local real nb[NX+4];

    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[I+2] = d_ev[IND];
        nb[I+2] = d_nb[IND];
        bulkpr[I+2] = d_bulkpr1[IND];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // set boundary condition (constant extrapolation)
    if ( get_local_id(0) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NX+3] = ev[NX+1];
       ev[NX+2] = ev[NX+1];

       nb[0] = nb[2];
       nb[1] = nb[2];
       nb[NX+3] = nb[NX+1];
       nb[NX+2] = nb[NX+1];


       bulkpr[0] = bulkpr[2];
       bulkpr[1] = bulkpr[2];
       bulkpr[NX+3] = bulkpr[NX+1];
       bulkpr[NX+2] = bulkpr[NX+1];

    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = I + 2;
        real4 ev_im2 = ev[i-2];
        real4 ev_im1 = ev[i-1];
        real4 ev_i   = ev[i];
        real4 ev_ip1 = ev[i+1];
        real4 ev_ip2 = ev[i+2];
        
        real nb_[5] = {nb[i-2], nb[i-1], nb[i], nb[i+1],nb[i+2] }; 
        real4 ev_[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};

        real u0_im2 = gamma_real4(ev_im2);
        real u0_im1 = gamma_real4(ev_im1);
        real u0_i = gamma_real4(ev_i);
        real u0_ip1 = gamma_real4(ev_ip1);
        real u0_ip2 = gamma_real4(ev_ip2);
        // .s1 -> .s2 or .s3 in other dirctions
        real v_mh = 0.5f*(ev_im1.s1 + ev_i.s1);
        real v_ph = 0.5f*(ev_ip1.s1 + ev_i.s1);

        real nb_mh = 0.5f*(nb_[1]+nb_[2]);
        real nb_ph = 0.5f*(nb_[2]+nb_[3]);

        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i),nb_mh, v_mh, eos_table);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i),nb_ph, v_ph, eos_table);
        
        
        real bulkpr_[5] = {u0_im2*bulkpr[i-2], u0_im1*bulkpr[i-1],u0_i*bulkpr[i], u0_ip1*bulkpr[i+1],u0_ip2*bulkpr[i+2]};     
	    d_Src[IND] = d_Src[IND] - kt1d_real2( bulkpr_, ev_, nb_, tau, ALONG_X,eos_table)/DX;

    }
}



__kernel void bulkpr_src_alongy(
             __global real* d_Src,
             __global real* d_bulkpr1,
	         __global real4* d_ev,
             __global real* d_nb,
             __global real* eos_table,
	     const real tau) {

    int I = get_global_id(0);
    int K = get_global_id(2);


    __local real4 ev[NY+4];
    __local real bulkpr[NY+4];
    __local real nb[NY+4];

    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[J+2] = d_ev[IND];
        nb[J+2] = d_nb[IND];
        bulkpr[J+2] = d_bulkpr1[IND];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // set boundary condition (constant extrapolation)
    if ( get_local_id(1) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NY+3] = ev[NY+1];
       ev[NY+2] = ev[NY+1];

       nb[0] = nb[2];
       nb[1] = nb[2];
       nb[NY+3] = nb[NY+1];
       nb[NY+2] = nb[NY+1];


       bulkpr[0] = bulkpr[2];
       bulkpr[1] = bulkpr[2];
       bulkpr[NY+3] = bulkpr[NY+1];
       bulkpr[NY+2] = bulkpr[NY+1];

    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = J + 2;
        real4 ev_im2 = ev[i-2];
        real4 ev_im1 = ev[i-1];
        real4 ev_i   = ev[i];
        real4 ev_ip1 = ev[i+1];
        real4 ev_ip2 = ev[i+2];
        
        real nb_[5] = {nb[i-2], nb[i-1], nb[i], nb[i+1],nb[i+2] }; 
        real4 ev_[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};

        real u0_im2 = gamma_real4(ev_im2);
        real u0_im1 = gamma_real4(ev_im1);
        real u0_i = gamma_real4(ev_i);
        real u0_ip1 = gamma_real4(ev_ip1);
        real u0_ip2 = gamma_real4(ev_ip2);
        // .s1 -> .s2 or .s3 in other dirctions
        real v_mh = 0.5f*(ev_im1.s2 + ev_i.s2);
        real v_ph = 0.5f*(ev_ip1.s2 + ev_i.s2);

        real nb_mh = 0.5f*(nb_[1]+nb_[2]);
        real nb_ph = 0.5f*(nb_[2]+nb_[3]);

        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i),nb_mh, v_mh, eos_table);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i),nb_ph, v_ph, eos_table);
        
        
        real bulkpr_[5] = {u0_im2*bulkpr[i-2], u0_im1*bulkpr[i-1],u0_i*bulkpr[i], u0_ip1*bulkpr[i+1],u0_ip2*bulkpr[i+2]};     
	    d_Src[IND] = d_Src[IND] - kt1d_real2( bulkpr_, ev_, nb_, tau, ALONG_Y,eos_table)/DY;

    }
}


__kernel void bulkpr_src_alongz(
             __global real* d_Src,
             __global real* d_bulkpr1,
	     __global real4* d_ev,
             __global real* d_nb,
             __global real* eos_table,
	     const real tau) {

    int I = get_global_id(0);
    int J = get_global_id(1);

    __local real4 ev[NZ+4];
    __local real bulkpr[NZ+4];
    __local real nb[NZ+4];

    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[K+2] = d_ev[IND];
        nb[K+2] = d_nb[IND];
        bulkpr[K+2] = d_bulkpr1[IND];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // set boundary condition (constant extrapolation)
    if ( get_local_id(2) == 0 ) {
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NZ+3] = ev[NZ+1];
       ev[NZ+2] = ev[NZ+1];

       nb[0] = nb[2];
       nb[1] = nb[2];
       nb[NZ+3] = nb[NZ+1];
       nb[NZ+2] = nb[NZ+1];


       bulkpr[0] = bulkpr[2];
       bulkpr[1] = bulkpr[2];
       bulkpr[NZ+3] = bulkpr[NZ+1];
       bulkpr[NZ+2] = bulkpr[NZ+1];

    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = K + 2;
        real4 ev_im2 = ev[i-2];
        real4 ev_im1 = ev[i-1];
        real4 ev_i   = ev[i];
        real4 ev_ip1 = ev[i+1];
        real4 ev_ip2 = ev[i+2];
        
        real nb_[5] = {nb[i-2], nb[i-1], nb[i], nb[i+1],nb[i+2] }; 
        real4 ev_[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};

        real u0_im2 = gamma_real4(ev_im2);
        real u0_im1 = gamma_real4(ev_im1);
        real u0_i = gamma_real4(ev_i);
        real u0_ip1 = gamma_real4(ev_ip1);
        real u0_ip2 = gamma_real4(ev_ip2);
        // .s1 -> .s2 or .s3 in other dirctions
        real v_mh = 0.5f*(ev_im1.s3 + ev_i.s3);
        real v_ph = 0.5f*(ev_ip1.s3 + ev_i.s3);

        real nb_mh = 0.5f*(nb_[1]+nb_[2]);
        real nb_ph = 0.5f*(nb_[2]+nb_[3]);

        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i),nb_mh, v_mh, eos_table);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i),nb_ph, v_ph, eos_table);
        
        
        real bulkpr_[5] = {u0_im2*bulkpr[i-2], u0_im1*bulkpr[i-1],u0_i*bulkpr[i], u0_ip1*bulkpr[i+1],u0_ip2*bulkpr[i+2]};     
	    d_Src[IND] = d_Src[IND] - kt1d_real2( bulkpr_, ev_, nb_, tau, ALONG_Z,eos_table)/(tau*DZ);

    }
}


__kernel void update_bulkpr(
    __global real * d_bulkprnew,
    __global real * d_bulkpr1,
    __global real * d_bulkprstep,
    __global real * d_pistep,
    __global real4 * d_ev1,
    __global real4 * d_ev2,
    __global real * d_nb1,
    __global real * d_nb2,
    __global real4 * d_udiff,
    __global real4 * d_udx,
    __global real4 * d_udy,
    __global real4 * d_udz,
    __global real * d_Src,
    __global real * eos_table,
    const real tau,
    const int  step)
{
    int I = get_global_id(0);
    
    if(I < NX*NY*NZ ){
    real4 e_v1 = d_ev1[I];
    real4 e_v2 = d_ev2[I];
    real  nb_v1 = d_nb1[I];
    real  nb_v2 = d_nb2[I];

    real4 u_old = umu4(e_v1);
    real4 u_new = umu4(e_v2);
    real4 udt = (u_new - u_old)/DT; //step==2

    // correct with previous udiff=u_visc-u_ideal*
    //if ( step == 1 ) udt += d_udiff[I]/DT;
    if ( step == 1 ) { 
        udt = d_udiff[I]/DT;
        //u_new = u_old + d_udiff[I]; //if step==1, we predict the flow velocity.
        u_new = u_old;
    }

    real4 udx = d_udx[I];
    real4 udy = d_udy[I];
    real4 udz = d_udz[I];

    // dalpha_ux = (partial_t, partial_x, partial_y, partial_z)ux
    real4 dalpha_u[4] = {(real4)(udt.s0, udx.s0, udy.s0, udz.s0),
                         (real4)(udt.s1, udx.s1, udy.s1, udz.s1),
                         (real4)(udt.s2, udx.s2, udy.s2, udz.s2),
                         (real4)(udt.s3, udx.s3, udy.s3, udz.s3)};
    // Notice DU = u^{lambda} \partial_{lambda} u^{beta} 
    real DU[4];
    real u[4];
    real ed_step = 0.0f;
    real nb_step = 0.0f;
    if ( step == 1 ) {
        for (int i=0; i<4; i++ ){
            DU[i] = dot(u_old, dalpha_u[i]);
        }
        u[0] = u_old.s0;
        u[1] = u_old.s1;
        u[2] = u_old.s2;
        u[3] = u_old.s3;
        ed_step = e_v1.s0;
        nb_step = nb_v1;
    } else {
        for (int i=0; i<4; i++ ){
            DU[i] = dot(u_new, dalpha_u[i]);
        }
        u[0] = u_new.s0;
        u[1] = u_new.s1;
        u[2] = u_new.s2;
        u[3] = u_new.s3;
        ed_step = e_v2.s0;
        nb_step = nb_v2;
    }
    
    // theta = dtut + dxux + dyuy + dzuz where d=coviariant differential
    real theta = udt.s0 + udx.s1 + udy.s2 + udz.s3;

    real temperature = eos_T(ed_step, nb_step, eos_table);
    real sss =eos_s(ed_step, nb_step, eos_table);
    //real local_etaos = etaos(temperature);
    real local_etaos = CZETA(temperature);
    real local_pr = eos_P(ed_step, nb_step, eos_table);
    real etav = hbarc * local_etaos * (ed_step + local_pr)/(max(temperature,acu));
    // GeV*fm^{-2}

    real cs2 = eos_CS2(ed_step, nb_step, eos_table);
    real tau_pi_factor = 15.0*(1/3.0-cs2)*(1/3.0-cs2)*(ed_step+local_pr);
    real one_over_taupi = tau_pi_factor/etav;
    real one_over_3DT = 1.0/(3.0*DT);
    one_over_taupi = min(one_over_3DT,one_over_taupi);
    one_over_taupi = max(1.0f/10.0f,one_over_taupi);

    
    // fm^-1 

    
    real sigma[4][4];
    real sigma_tep=0.0;
    for (int mu=0; mu<4; mu ++){
        for (int nu= 0; nu < 4; nu++ ){
                sigma_tep = dot(gm[mu], dalpha_u[nu]) + 
                            dot(gm[nu], dalpha_u[mu]) -
                            (u[mu]*DU[nu] + u[nu]*DU[mu]) -
                            2.0f/3.0f*(gmn[mu][nu]-u[mu]*u[nu])*theta;
                 if(u[0] > 10000.0f){
                     sigma_tep=0.0f;
                  }
                 sigma[mu][nu] = sigma_tep;
    
    }
    }

    real pimn[4][4] = {{d_pistep[10*I+idx(0, 0)],d_pistep[10*I+idx(0, 1)],d_pistep[10*I+idx(0, 2)],d_pistep[10*I+idx(0, 3)]},
                       {d_pistep[10*I+idx(1, 0)],d_pistep[10*I+idx(1, 1)],d_pistep[10*I+idx(1, 2)],d_pistep[10*I+idx(1, 3)]},
                       {d_pistep[10*I+idx(2, 0)],d_pistep[10*I+idx(2, 1)],d_pistep[10*I+idx(2, 2)],d_pistep[10*I+idx(2, 3)]},
                       {d_pistep[10*I+idx(3, 0)],d_pistep[10*I+idx(3, 1)],d_pistep[10*I+idx(3, 2)],d_pistep[10*I+idx(3, 3)]}};
    
    real pisigma_coupling = 0.0;
    for(int mu = 0 ; mu<4 ; mu++){
        for(int nu = 0 ; nu<4 ; nu++){
            for(int alpha = 0 ; alpha<4 ; alpha++){
                for(int beta = 0 ; beta<4 ; beta++){
                    pisigma_coupling = pisigma_coupling+ gmn[mu][alpha]*gmn[nu][beta]*pimn[mu][nu]*sigma[alpha][beta];
                }
            }
        }
    }
   
    real trans_coeff = (1/3.0 - cs2)*4.0/5.0;


    d_Src[I] = d_Src[I] + trans_coeff*pisigma_coupling;
     
    
    real piNS = -etav*theta;
    real pi_old = d_bulkpr1[I] * u_old.s0;
    d_Src[I] = d_Src[I] + 1/3.0* d_bulkprstep[I] *theta;
    pi_old = (pi_old + (d_Src[I]+one_over_taupi*piNS)*DT/step) / (u_new.s0 + DT*one_over_taupi);
    real pi_old1 = d_bulkpr1[I] ;

    
    d_bulkprnew[I] = pi_old;

    
    //real eps_scale = 0.1;
    //real xi = 0.01;
    //real factor = 100.0*(1.0/(exp(-(ed_step - eps_scale)/xi)+1.0) 
    //- 1.0/(exp(( eps_scale)/xi)+1.0));
    //
    //
    //real bulksize = 3.*pi_old*pi_old;

    //real eq_size = ed_step*ed_step + 3.*local_pr*local_pr;

    //real rho_bulk  = sqrt(bulksize/eq_size)/factor;
    //real rho_bulk_max = 0.1;
    //if(rho_bulk > rho_bulk_max )
    //{
    //    d_bulkprnew[I] = (rho_bulk_max/rho_bulk)*pi_old;
    //}
     
    if (fabs(pi_old) > local_pr)
    {
        d_bulkprnew[I]=0;
        
    }


    
 

   

    


    }

}



