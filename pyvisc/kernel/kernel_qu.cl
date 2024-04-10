#include<helper.h>


#pragma OPENCL EXTENSION cl_amd_printf : enable

constant real gmn[4][4] = {{1.0f, 0.0f, 0.0f, 0.0f},
                           {0.0f,-1.0f, 0.0f, 0.0f},
                           {0.0f, 0.0f,-1.0f, 0.0f},
                           {0.0f, 0.0f, 0.0f,-1.0f}};


__kernel void qub_src_christoffel(
            __global real* d_Src,
            __global real* d_qb1,
            __global real4* d_ev,
            const real tau,
            const int step){

    int I = get_global_id(0);
    if(I < NX*NY*NZ) {
        if (step == 1) {
           for (int mn = 0; mn < 4; mn++){
               d_Src[idn2(I, mn)]=0.0f;
           } 
        }

        real4 e_v = d_ev[I];
        real uz = e_v.s3*gamma(e_v.s1, e_v.s2, e_v.s3);
        real u0 = gamma(e_v.s1, e_v.s2, e_v.s3);
        d_Src[idn2(I,0)] -= uz* d_qb1[idn2(I,3)]/tau;
        d_Src[idn2(I,1)] -= 0.0f;
        d_Src[idn2(I,2)] -= 0.0f;
        d_Src[idn2(I,3)] -= (uz* d_qb1[idn2(I,0)]/tau + u0*d_qb1[idn2(I,3)]/tau);
        //d_Src[idn2(I,3)] -= (u0*d_qb1[idn2(I,3)]/tau);
        //d_Src[idn2(I,3)] -= (uz* d_qb1[idn2(I,0)]/tau);
        //d_Src[idn2(I,3)] -= 0.0;

        
    
    }
}



__kernel void qub_src_alongx(
             __global real* d_Src,
             __global real* d_mubdx,
             __global real* d_qb1,
	     __global real4* d_ev,
             __global real* d_nb,
             __global real* eos_table,
	     const real tau,
	     const int step) {


    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    __local real qb[4*(NX+4)];
    __local real nb[NX+4];

    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[I+2] = d_ev[IND];
        nb[I+2] = d_nb[IND];
        for ( int mn = 0; mn < 4; mn ++ ) {
            qb[idn2(I+2, mn)] = d_qb1[idn2(IND, mn)];
        }
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
       for ( int mn = 0; mn < 4; mn ++ ) {
             qb[idn2(0, mn)] = qb[idn2(2, mn)];
             qb[idn2(1, mn)] = qb[idn2(2, mn)];
             qb[idn2(NX+3, mn)] = qb[idn2(NX+1, mn)];
             qb[idn2(NX+2, mn)] = qb[idn2(NX+1, mn)];
       }
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
        
        real mu0_im2 = eos_mu(ev_im2.s0,nb_[0],eos_table)/(max(eos_T(ev_im2.s0,nb_[0],eos_table),acu));
        real mu0_im1 = eos_mu(ev_im1.s0,nb_[1],eos_table)/(max(eos_T(ev_im1.s0,nb_[1],eos_table),acu));
        real mu0_i = eos_mu(ev_i.s0,nb_[2],eos_table)/(max(eos_T(ev_i.s0,nb_[2],eos_table),acu));
        real mu0_ip1 = eos_mu(ev_ip1.s0,nb_[3],eos_table)/(max(eos_T(ev_ip1.s0,nb_[3],eos_table),acu));
        real mu0_ip2 = eos_mu(ev_ip2.s0,nb_[4],eos_table)/(max(eos_T(ev_ip2.s0,nb_[4],eos_table),acu));
 
        d_mubdx[IND] = minmod(0.5f*(mu0_ip1-mu0_im1),
                            minmod(THETA*(mu0_i - mu0_im1),
                                    THETA*(mu0_ip1 - mu0_i)
                                    ))/DX;

        //d_mubdx[IND] = minmod3(THETA*(mu0_ip1 - mu0_i),THETA*(mu0_i - mu0_im1),0.5f*(mu0_ip1-mu0_im1))/DX;


        for ( int mn = 0; mn < 4; mn ++ ) {
            real qb_[5] = {u0_im2*qb[idn2(i-2, mn)], u0_im1*qb[idn2(i-1, mn)],u0_i*qb[idn2(i, mn)], u0_ip1*qb[idn2(i+1, mn)],u0_ip2*qb[idn2(i+2, mn)]}; 

           //d_Src[idn2(IND, mn)] = d_Src[idn2(IND, mn)] - kt1d_real(
           //   u0_im2*qb[idn2(i-2, mn)], u0_im1*qb[idn2(i-1, mn)],
           //   u0_i*qb[idn2(i, mn)], u0_ip1*qb[idn2(i+1, mn)],
           //   u0_ip2*qb[idn2(i+2, mn)],
           //   v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_X)/DX;
            
	   d_Src[idn2(IND, mn)] = d_Src[idn2(IND, mn)] - kt1d_real2(
               qb_, ev_, nb_, tau, ALONG_X,eos_table)/DX;

          

        }

    }
}


__kernel void qub_src_alongy(
             __global real * d_Src,
             __global real * d_mubdy,
             __global real * d_qb1,
             __global real4 * d_ev,
             __global real * d_nb,
             __global real * eos_table,
             const real tau,
	     const int step) {
    int I = get_global_id(0);
    int K = get_global_id(2);
    __local real4 ev[NY+4];
    __local real qb[4*(NY+4)];
    __local real nb[NY+4];
    
    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[J+2] = d_ev[IND];
        nb[J+2] = d_nb[IND];
        for ( int mn = 0; mn < 4; mn ++ ) {
            qb[idn2(J+2, mn)] = d_qb1[idn2(IND, mn)];
        }
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

        for ( int mn = 0; mn < 4; mn ++ ) {
            qb[idn2(0, mn)] = qb[idn2(2, mn)];
            qb[idn2(1, mn)] = qb[idn2(2, mn)];
            qb[idn2(NY+3, mn)] = qb[idn2(NY+1, mn)];
            qb[idn2(NY+2, mn)] = qb[idn2(NY+1, mn)];
        }
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
        
        real nb_[5]={nb[i-2],nb[i-1],nb[i],nb[i+1],nb[i+2]};
        real4 ev_[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};

        real u0_im2 = gamma_real4(ev_im2);
        real u0_im1 = gamma_real4(ev_im1);
        real u0_i = gamma_real4(ev_i);
        real u0_ip1 = gamma_real4(ev_ip1);
        real u0_ip2 = gamma_real4(ev_ip2);
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s2 + ev_i.s2);
        real v_ph = 0.5f*(ev_ip1.s2 + ev_i.s2);
        
        real nb_mh = 0.5f*(nb_[1]+nb_[2]);
        real nb_ph = 0.5f*(nb_[2]+nb_[3]);
        
        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i),nb_mh, v_mh, eos_table);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i),nb_ph, v_ph, eos_table);
        
        real mu0_im2 = eos_mu(ev_im2.s0,nb_[0],eos_table)/(max(eos_T(ev_im2.s0,nb_[0],eos_table),acu));
        real mu0_im1 = eos_mu(ev_im1.s0,nb_[1],eos_table)/(max(eos_T(ev_im1.s0,nb_[1],eos_table),acu));
        real mu0_i = eos_mu(ev_i.s0,nb_[2],eos_table)/(max(eos_T(ev_i.s0,nb_[2],eos_table),acu));
        real mu0_ip1 = eos_mu(ev_ip1.s0,nb_[3],eos_table)/(max(eos_T(ev_ip1.s0,nb_[3],eos_table),acu));
        real mu0_ip2 = eos_mu(ev_ip2.s0,nb_[4],eos_table)/(max(eos_T(ev_ip2.s0,nb_[4],eos_table),acu));


        d_mubdy[IND] = minmod(0.5f*(mu0_ip1 - mu0_im1),
                            minmod(THETA*(mu0_i - mu0_im1),
                                    THETA*(mu0_ip1 - mu0_i)
                                    ))/DY;
        //d_mubdy[IND] = minmod3(THETA*(mu0_ip1 - mu0_i),THETA*(mu0_i - mu0_im1),0.5f*(mu0_ip1-mu0_im1))/DY;
        //d_udy[IND] = 0.5f*(umu4(ev_ip1)-umu4(ev_im1))/DY;
        //d_udy[IND] = (-umu4(ev_ip2)+8.0f*umu4(ev_ip1)-8.0f*umu4(ev_im1)+umu4(ev_im2))/(12.0f*DY);


        for ( int mn = 0; mn < 4; mn ++ ) {
            real qb_[5] = {u0_im2*qb[idn2(i-2, mn)], u0_im1*qb[idn2(i-1, mn)],u0_i*qb[idn2(i, mn)], u0_ip1*qb[idn2(i+1, mn)],u0_ip2*qb[idn2(i+2, mn)]};
            //d_Src[idn2(IND,mn)] = d_Src[idn2(IND,mn)] - kt1d_real(
            //   u0_im2*qb[idn2(i-2,mn)], u0_im1*qb[idn2(i-1,mn)],
            //   u0_i*qb[idn2(i,mn)], u0_ip1*qb[idn2(i+1,mn)],
            //   u0_ip2*qb[idn2(i+2,mn)],
            //   v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_Y)/DY;

	    d_Src[idn2(IND, mn)] = d_Src[idn2(IND, mn)] - kt1d_real2(
               qb_, ev_, nb_, tau, ALONG_Y,eos_table)/DY;

        }
    }
}


__kernel void qub_src_alongz(
             __global real * d_Src,
             __global real * d_mubdz,
             __global real * d_qb1,
             __global real4 * d_ev,
             __global real * d_nb,
             __global real * eos_table,
             const real tau,
	     const int step) {

    int I = get_global_id(0);
    int J = get_global_id(1);
    __local real4 ev[NZ+4];
    __local real qb[4*(NZ+4)];
    __local real nb[NZ+4];

    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[K+2] = d_ev[IND];
        nb[K+2] = d_nb[IND];
        for ( int mn = 0; mn < 4; mn ++ ) {
            qb[idn2(K+2, mn)] = d_qb1[idn2(IND, mn)];
        }
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

        for ( int mn = 0; mn < 4; mn ++ ) {
            qb[idn2(0,mn)] = qb[idn2(2,mn)];
            qb[idn2(1,mn)] = qb[idn2(2,mn)];
            qb[idn2(NZ+3, mn)] = qb[idn2(NZ+1, mn)];
            qb[idn2(NZ+2, mn)] = qb[idn2(NZ+1, mn)];
        }
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
        
        real nb_[5] = {nb[i-2],nb[i-1],nb[i],nb[i+1],nb[i+2]};
        real4 ev_[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]}; 

        real u0_im2 = gamma_real4(ev_im2);
        real u0_im1 = gamma_real4(ev_im1);
        real u0_i = gamma_real4(ev_i);
        real u0_ip1 = gamma_real4(ev_ip1);
        real u0_ip2 = gamma_real4(ev_ip2);
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s3 + ev_i.s3);
        real v_ph = 0.5f*(ev_ip1.s3 + ev_i.s3);
        
        real nb_mh = 0.5f*(nb_[1]+nb_[2]);
        real nb_ph = 0.5f*(nb_[2]+nb_[3]);
        
        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i),nb_mh, v_mh, eos_table);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i),nb_ph, v_ph, eos_table);


        real mu0_im2 = eos_mu(ev_im2.s0,nb_[0],eos_table)/(max(eos_T(ev_im2.s0,nb_[0],eos_table),acu));
        real mu0_im1 = eos_mu(ev_im1.s0,nb_[1],eos_table)/(max(eos_T(ev_im1.s0,nb_[1],eos_table),acu));
        real mu0_i = eos_mu(ev_i.s0,nb_[2],eos_table)/(max(eos_T(ev_i.s0,nb_[2],eos_table),acu));
        real mu0_ip1 = eos_mu(ev_ip1.s0,nb_[3],eos_table)/(max(eos_T(ev_ip1.s0,nb_[3],eos_table),acu));
        real mu0_ip2 = eos_mu(ev_ip2.s0,nb_[4],eos_table)/(max(eos_T(ev_ip2.s0,nb_[4],eos_table),acu));



        real4 christoffel_term = (real4)(ev_i.s3, 0.0f, 0.0f, 1.0f)*u0_i/tau;
        
        d_mubdz[IND] = minmod(0.5f*(mu0_ip1 - mu0_im1),
                           minmod(THETA*(mu0_i - mu0_im1),
                                    THETA*(mu0_ip1 - mu0_i)
                           ))/(DZ*tau);

        
        for ( int mn = 0; mn < 4; mn ++ ) {
            real qb_[5] = {u0_im2*qb[idn2(i-2, mn)], u0_im1*qb[idn2(i-1, mn)],u0_i*qb[idn2(i, mn)], u0_ip1*qb[idn2(i+1, mn)],u0_ip2*qb[idn2(i+2, mn)]};

            //d_Src[idn2(IND, mn)] = d_Src[idn2(IND,mn)] - kt1d_real(
            //   u0_im2*qb[idn2(i-2,mn)], u0_im1*qb[idn2(i-1,mn)],
            //   u0_i*qb[idn2(i,mn)], u0_ip1*qb[idn2(i+1,mn)],
            //   u0_ip2*qb[idn2(i+2,mn)],
            //   v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_Z)/(tau*DZ);
	    d_Src[idn2(IND, mn)] = d_Src[idn2(IND, mn)] - kt1d_real2(
               qb_, ev_, nb_, tau, ALONG_Z,eos_table)/(tau*DZ);


        }
    }
}

__kernel void update_qub(
    __global real * d_qbnew,
    __global real * d_goodcell,
    __global real * d_qb1,
    __global real * d_qbstep,
    __global real4 * d_ev1,
    __global real4 * d_ev2,
    __global real * d_nb1,
    __global real * d_nb2,
    __global real4 * d_udiff,
    __global real4 * d_udx,
    __global real4 * d_udy,
    __global real4 * d_udz,
    __global real * d_mubdiff,
    __global real * d_mubdx,
    __global real * d_mubdy,
    __global real * d_mubdz,
    __global real * d_Src,
    __global real * eos_table,
    const real tau,
    const int  step)
{
    int I = get_global_id(0);
    
    if(I < NX * NY *NZ)
    { 

    real4 e_v1 = d_ev1[I];
    real4 e_v2 = d_ev2[I];
    real  nb_v1 = d_nb1[I];
    real  nb_v2 = d_nb2[I];

    real4 u_old = umu4(e_v1);
    real4 u_new = umu4(e_v2);
    real4 udt = (u_new - u_old)/DT; //step==2

    real mubdt = ( eos_mu(e_v2.s0,nb_v2,eos_table)/(max(eos_T(e_v2.s0,nb_v2,eos_table),acu))
                - eos_mu(e_v1.s0,nb_v1,eos_table)/(max(eos_T(e_v1.s0,nb_v1,eos_table),acu) ))/DT;
    
    

    // correct with previous udiff=u_visc-u_ideal*
    //if ( step == 1 ) udt += d_udiff[I]/DT;
    if ( step == 1 ) { 
        udt = d_udiff[I]/DT;
        mubdt = d_mubdiff[I]/DT;
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

    real partial_muT[4] = {mubdt, d_mubdx[I],d_mubdy[I],d_mubdz[I] };
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
   real local_CB = CB(temperature);
   real nbT_over_epP= nb_step*temperature/(ed_step+eos_P(ed_step,nb_step,eos_table));
   real coth_mubT = 1.0f/(3.0f*tanh(eos_mu(ed_step,nb_step,eos_table)/temperature)+1e-15); 
   real kappaB = nb_step*(coth_mubT-nbT_over_epP)/(temperature/hbarc);
   real kappaB1 = kappaB;
   kappaB = local_CB*kappaB;
   if (kappaB > 10000.0f )
   {
       kappaB = 0.0f;
   }



   
    real one_over_tauqu = temperature/(1.0f*max(acu,local_CB)*hbarc);
    real one_over_3DT = 1.0/(3.0*DT);
    one_over_tauqu = min(one_over_3DT,one_over_tauqu);

    real qb2[4];
    for ( int mn=0; mn < 4; mn ++ ) {
        qb2[mn] = d_qbstep[idn2(I,mn)]; //old_one
    }


    real sigma;
    real max_pimn_abs = 0.0f;
    real max_qb_abs = 0.0f;
     
    real qsigma[4];
    for (int mu=0; mu<4; mu ++){
        qsigma[mu] = 0.0f;
        for (int nu= 0; nu < 4; nu++ ){
            for (int la = 0; la<4;la++){
                sigma = dot(gm[mu], dalpha_u[nu]) + 
                            dot(gm[nu], dalpha_u[mu]) -
                            (u[mu]*DU[nu] + u[nu]*DU[mu]) -
                            2.0f/3.0f*(gmn[mu][nu]-u[mu]*u[nu])*theta;
                 if(u[0] > 10000.0f){
                     sigma=0.0f;
                  }
                 qsigma[mu] += gmn[la][nu]*qb2[la]*(3.0f/10.0f)*sigma;
    }
    }
    }
    
    real gduq = 0.0f;
    for (int la =0; la < 4 ;la++ ){
        for(int nu = 0; nu < 4;nu++){
            gduq += gmn[la][nu]*DU[nu]*qb2[la];
        }
    }
 

    for (int mu=0 ; mu<4 ; mu++)
    {
        real qubNS = 0.0f;
        for(int nu = 0 ;nu <4 ;nu++){       
            qubNS += kappaB*(gmn[mu][nu]-u[mu]*u[nu])*partial_muT[nu];
        }
       


	real src = - qsigma[mu] - gduq*u[mu] - qb2[mu]*u[0]/(tau*1.0f);
        real qb_old = d_qb1[idn2(I,mu)]*u_old.s0;
        
        d_Src[idn2(I,mu)] += src;
        
        qb_old = (qb_old + d_Src[idn2(I,mu)]*DT/step 
                + DT*one_over_tauqu*qubNS )/(u_new.s0 + DT*one_over_tauqu);
        
        //qb_old = (qb_old + (d_Src[idn2(I,mu)]+one_over_tauqu*qubNS)*DT/step)/(u_new.s0 + DT*one_over_tauqu);
        
        
        if ( fabs(qb_old) > max_qb_abs )  max_qb_abs = fabs(qb_old);
        d_qbnew[idn2(I,mu)]= qb_old;

    }
    
    d_qbnew[idn2(I,0)] = (d_qbnew[idn2(I,1)]*u_new.s1 + d_qbnew[idn2(I,2)]*u_new.s2 + d_qbnew[idn2(I,3)]*u_new.s3 )/(u_new.s0);

    if (fabs(d_qbnew[idn2(I,0)]) > max_qb_abs ) max_qb_abs = fabs(d_qbnew[idn2(I,0)]);

    real J0 = nb_step*u_new.s0;

    if (max_qb_abs > J0)
    {
       for(int mn=0; mn < 4; mn++)
       {
           d_qbnew[idn2(I,mn)] = 0.0f;
       }
    }
    
        
    
    // real eps_scale = 0.1;
    // real xi = 0.01;
    // real factor = 100.0*(1.0/(exp(-(ed_step - eps_scale)/xi)+1.0) - 1.0/(exp(( eps_scale)/xi)+1.0));

    // real d_qb_local[4];
    
    // d_qb_local[0] = d_qbnew[idn2(I,0)];
    // d_qb_local[1] = d_qbnew[idn2(I,1)];
    // d_qb_local[2] = d_qbnew[idn2(I,2)];
    // d_qb_local[3] = d_qbnew[idn2(I,3)];

    // real q_size = d_qb_local[0]*d_qb_local[0]-d_qb_local[1]*d_qb_local[1]-d_qb_local[2]*d_qb_local[2]-d_qb_local[3]*d_qb_local[3];
    // real rho_q = 0.0;
    // real rho_q_max = 0.0;
    // if(q_size > 0){
     
    //  d_qbnew[idn2(I,0)] = 0.0;
    //  d_qbnew[idn2(I,1)] = 0.0;
    //  d_qbnew[idn2(I,2)] = 0.0;
    //  d_qbnew[idn2(I,3)] = 0.0;

    // }
    // else{

    // rho_q = sqrt(-q_size/(nb_step*nb_step))/factor;
    // rho_q_max = 0.1;

    // if(rho_q > rho_q_max )
    // {
    //  d_qbnew[idn2(I,0)] = (rho_q_max/rho_q)*d_qb_local[0];
    //  d_qbnew[idn2(I,1)] = (rho_q_max/rho_q)*d_qb_local[1];
    //  d_qbnew[idn2(I,2)] = (rho_q_max/rho_q)*d_qb_local[2];
    //  d_qbnew[idn2(I,3)] = (rho_q_max/rho_q)*d_qb_local[3];
    // }
    // }



    


    }


}

