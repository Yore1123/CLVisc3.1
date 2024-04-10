#include<helper.h>
#pragma OPENCL EXTENSION cl_amd_printf : enable

constant real gmn[4][4] = {{1.0f, 0.0f, 0.0f, 0.0f},
                           {0.0f,-1.0f, 0.0f, 0.0f},
                           {0.0f, 0.0f,-1.0f, 0.0f},
                           {0.0f, 0.0f, 0.0f,-1.0f}};

__kernel void visc_initialize(
             __global real * d_qb1, 
             __global real * d_qb2, 
             __global real * d_pi1,
             __global real * d_pi2,
             __global real * d_bulkpr1,
             __global real * d_bulkpr2,
             __global real * d_goodcell,
	     __global real4 * d_udiff,
	     __global real * d_mubdiff,
	     __global real4 * d_ev,
             __global real * d_nb,
	     const real tau,
             __global real* eos_table) {
    int I = get_global_id(0);

    if ( I < NX*NY*NZ ) {
       real4 ev = d_ev[I];
       real  nb = d_nb[I];
       //real temperature = T(ev.s0, eos_table);
       //real etav = etaos(temperature) * S(ev.s0, eos_table) * hbarc;
       real temperature = eos_T(ev.s0, nb, eos_table);
       real pressure = eos_P(ev.s0, nb, eos_table);
       real etav = hbarc * CETA(temperature) * (ev.s0 + pressure)/(max(temperature,acu));
       real etav1 =  (ev.s0 + pressure)/(max(temperature,acu));
       real tmp = 2.0f/3.0f * etav / tau;
       //d_pi1[10*I+idx(1, 1)] = tmp;  //Gev*fm^-3
       //d_pi1[10*I+idx(2, 2)] = tmp;
       //d_pi1[10*I+idx(3, 3)] = -2.0f*tmp;
       
       d_pi1[10*I+idx(1, 1)] = 0.0f;
       d_pi1[10*I+idx(2, 2)] = 0.0f;
       d_pi1[10*I+idx(3, 3)] = 0.0f;
       
       d_pi2[10*I+idx(1, 1)] = 0.0f;
       d_pi2[10*I+idx(2, 2)] = 0.0f;
       d_pi2[10*I+idx(3, 3)] = 0.0f;
       d_udiff[I] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
       d_mubdiff[I] = 0.0f;
       d_goodcell[I] = 1.0f;
       d_qb1[I*4+0] = 0.0;
       d_qb1[I*4+1] = 0.0;
       d_qb1[I*4+2] = 0.0;
       d_qb1[I*4+3] = 0.0;
       d_qb2[I*4+0] = 0.0;
       d_qb2[I*4+1] = 0.0;
       d_qb2[I*4+2] = 0.0;
       d_qb2[I*4+3] = 0.0;


       d_bulkpr1[I] = 0.0;
       d_bulkpr2[I] = 0.0;
       

    }
}






__kernel void visc_src_christoffel(
             __global real * d_Src,
             __global real * d_pi1,
             __global real4 * d_ev,
             __global real * d_nb,
             const real tau,
             const int step) {
    int I = get_global_id(0);

    if ( I < NX*NY*NZ ) {
        if ( step == 1 ) {
           for ( int mn = 0; mn < 10; mn ++ ) {
               d_Src[10*I+mn] = 0.0f;
           }
        }
        
        real4 e_v = d_ev[I];
        real uz = e_v.s3 * gamma(e_v.s1, e_v.s2, e_v.s3);

        d_Src[10*I+0] -= 2.0f * uz * d_pi1[10*I+idx(0, 3)]/tau;
        d_Src[10*I+1] -= uz * d_pi1[10*I+idx(1, 3)]/tau;
        d_Src[10*I+2] -= uz * d_pi1[10*I+idx(2, 3)]/tau;
        d_Src[10*I+3] -= uz * (d_pi1[10*I+idx(0, 0)] + d_pi1[10*I+idx(3,3)])/tau;
        d_Src[10*I+6] -= uz * d_pi1[10*I+idx(0, 1)]/tau;
        d_Src[10*I+8] -= uz * d_pi1[10*I+idx(0, 2)]/tau;
        d_Src[10*I+9] -= 2.0f * uz * d_pi1[10*I+idx(0, 3)]/tau;

    }
}


// output: d_Src kt src for pimn evolution;
// output: d_udx the velocity gradient along x
// all the others are input
__kernel void visc_src_alongx(
             __global real * d_Src,
             __global real4 * d_udx,
             __global real * d_pi1,
	     __global real4 * d_ev,
             __global real* d_nb,
             __global real* eos_table,
	     const real tau) {
    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    __local real pimn[10*(NX+4)];
    __local real nb[NX+4];

    // Use num of threads = BSZ to compute src for NX elements
    // 10 pimn terms are stored continuesly
    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[I+2] = d_ev[IND];
        nb[I+2] = d_nb[IND];
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[idn(I+2, mn)] = d_pi1[idn(IND, mn)];
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
       for ( int mn = 0; mn < 10; mn ++ ) {
             pimn[idn(0, mn)] = pimn[idn(2, mn)];
             pimn[idn(1, mn)] = pimn[idn(2, mn)];
             pimn[idn(NX+3, mn)] = pimn[idn(NX+1, mn)];
             pimn[idn(NX+2, mn)] = pimn[idn(NX+1, mn)];
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
        // .s1 -> .s2 or .s3 in other directions
        real v_mh = 0.5f*(ev_im1.s1 + ev_i.s1);
        real v_ph = 0.5f*(ev_ip1.s1 + ev_i.s1);

        real nb_mh = 0.5f*(nb_[1]+nb_[2]);
        real nb_ph = 0.5f*(nb_[2]+nb_[3]);

        real lam_mh = maxPropagationSpeed(0.5f*(ev_im1+ev_i),nb_mh, v_mh, eos_table);
        real lam_ph = maxPropagationSpeed(0.5f*(ev_ip1+ev_i),nb_ph, v_ph, eos_table);
        
        d_udx[IND] = minmod4(0.5f*(umu4(ev_ip1)-umu4(ev_im1)),
                             minmod4(THETA*(umu4(ev_i) - umu4(ev_im1)),
                                     THETA*(umu4(ev_ip1) - umu4(ev_i))
                                     ))/DX;


        for ( int mn = 0; mn < 10; mn ++ ) {
	    real pimn_[5] = {u0_im2*pimn[idn(i-2, mn)], u0_im1*pimn[idn(i-1, mn)],u0_i*pimn[idn(i, mn)], u0_ip1*pimn[idn(i+1, mn)],u0_ip2*pimn[idn(i+2, mn)]};



            //d_Src[idn(IND, mn)] = d_Src[idn(IND, mn)] - kt1d_real(
            //  u0_im2*pimn[idn(i-2, mn)], u0_im1*pimn[idn(i-1, mn)],
            //   u0_i*pimn[idn(i, mn)], u0_ip1*pimn[idn(i+1, mn)],
            //   u0_ip2*pimn[idn(i+2, mn)],
            //   v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_X)/DX;
            d_Src[idn(IND, mn)] = d_Src[idn(IND, mn)] - kt1d_real2(
               pimn_, ev_, nb_, tau, ALONG_X,eos_table)/DX;



        }
    }
}

__kernel void visc_src_alongy(
             __global real * d_Src,
             __global real4 * d_udy,
             __global real * d_pi1,
             __global real4 * d_ev,
             __global real * d_nb,
             __global real * eos_table,
             const real tau) {
    int I = get_global_id(0);
    int K = get_global_id(2);
    __local real4 ev[NY+4];
    __local real pimn[10*(NY+4)];
    __local real nb[NY+4];
    
    // Use num of threads = BSZ to compute src for NX elements
    // 10 pimn terms are stored continuesly
    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[J+2] = d_ev[IND];
        nb[J+2] = d_nb[IND];
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[idn(J+2, mn)] = d_pi1[idn(IND, mn)];
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

        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[idn(0, mn)] = pimn[idn(2, mn)];
            pimn[idn(1, mn)] = pimn[idn(2, mn)];
            pimn[idn(NY+3, mn)] = pimn[idn(NY+1, mn)];
            pimn[idn(NY+2, mn)] = pimn[idn(NY+1, mn)];
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
        
        d_udy[IND] = minmod4(0.5f*(umu4(ev_ip1)-umu4(ev_im1)),
                             minmod4(THETA*(umu4(ev_i) - umu4(ev_im1)),
                                     THETA*(umu4(ev_ip1) - umu4(ev_i))
                                     ))/DY;
        //d_udy[IND] = 0.5f*(umu4(ev_ip1)-umu4(ev_im1))/DY;
        //d_udy[IND] = (-umu4(ev_ip2)+8.0f*umu4(ev_ip1)-8.0f*umu4(ev_im1)+umu4(ev_im2))/(12.0f*DY);


        for ( int mn = 0; mn < 10; mn ++ ) {

             real pimn_[5] = {u0_im2*pimn[idn(i-2, mn)], u0_im1*pimn[idn(i-1, mn)],u0_i*pimn[idn(i, mn)], u0_ip1*pimn[idn(i+1, mn)],u0_ip2*pimn[idn(i+2, mn)]};

           //d_Src[10*IND + mn] = d_Src[10*IND + mn] - kt1d_real(
           //    u0_im2*pimn[(i-2)*10+mn], u0_im1*pimn[(i-1)*10+mn],
           //    u0_i*pimn[i*10+mn], u0_ip1*pimn[(i+1)*10+mn],
           //    u0_ip2*pimn[(i+2)*10+mn],
           //    v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_Y)/DY;

           d_Src[idn(IND, mn)] = d_Src[idn(IND, mn)] - kt1d_real2(
              pimn_, ev_, nb_, tau, ALONG_Y,eos_table)/DY;

        }
    }
}




__kernel void visc_src_alongz(
             __global real * d_Src,
             __global real4 * d_udz,
             __global real * d_pi1,
             __global real4 * d_ev,
             __global real * d_nb,
             __global real * eos_table,
             const real tau) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    __local real4 ev[NZ+4];
    __local real pimn[10*(NZ+4)];
    __local real nb[NZ+4];

    // Use num of threads = BSZ to compute src for NX elements
    // 10 pimn terms are stored continuesly
    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[K+2] = d_ev[IND];
        nb[K+2] = d_nb[IND];
        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[idn(K+2, mn)] = d_pi1[idn(IND, mn)];
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

        for ( int mn = 0; mn < 10; mn ++ ) {
            pimn[mn] = pimn[20+mn];
            pimn[10+mn] = pimn[20+mn];
            pimn[idn(NZ+3, mn)] = pimn[idn(NZ+1, mn)];
            pimn[idn(NZ+2, mn)] = pimn[idn(NZ+1, mn)];
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
        
        real4 christoffel_term = (real4)(ev_i.s3, 0.0f, 0.0f, 1.0f)*u0_i/tau;
        
        d_udz[IND] = minmod4(0.5f*(umu4(ev_ip1)-umu4(ev_im1)),
                             minmod4(THETA*(umu4(ev_i) - umu4(ev_im1)),
                                     THETA*(umu4(ev_ip1) - umu4(ev_i))
                            ))/(DZ*tau) + christoffel_term;
        //d_udz[IND] = 0.5f*(umu4(ev_ip1)-umu4(ev_im1))/(tau*DZ) + christoffel_term;

        //d_udz[IND] = (-umu4(ev_ip2)+8.0f*umu4(ev_ip1)-8.0f*umu4(ev_im1)+umu4(ev_im2)
        //              )/(12.0f*DZ*tau) + christoffel_term;

        for ( int mn = 0; mn < 10; mn ++ ) {
            real pimn_[5] = {u0_im2*pimn[idn(i-2, mn)], u0_im1*pimn[idn(i-1, mn)],u0_i*pimn[idn(i, mn)], u0_ip1*pimn[idn(i+1, mn)],u0_ip2*pimn[idn(i+2, mn)]};
	
            //d_Src[10*IND + mn] = d_Src[10*IND + mn] - kt1d_real(
            //  u0_im2*pimn[(i-2)*10+mn], u0_im1*pimn[(i-1)*10+mn],
            //   u0_i*pimn[i*10+mn], u0_ip1*pimn[(i+1)*10+mn],
            //   u0_ip2*pimn[(i+2)*10+mn],
            //   v_mh, v_ph, lam_mh, lam_ph, tau, ALONG_Z)/(tau*DZ);

            d_Src[idn(IND, mn)] = d_Src[idn(IND, mn)] - kt1d_real2(
               pimn_, ev_, nb_, tau, ALONG_Z,eos_table)/(tau*DZ);
        }
    }
}



/** \pi^{< mu}_{lam} * pi^{nu> lam} self coupling term **/
inline real PiPi(int lam, int mu, int nu, __private real pimn[10], __private real u[4])
{
  /** pi_{\mu}^{lam} **/
  real4 A = (real4)(pimn[idx(0, lam)], -pimn[idx(1, lam)], -pimn[idx(2, lam)],
                    -pimn[idx(3, lam)]);

  real4 umu = (real4)(u[0], u[1], u[2], u[3]);

  real4 Delta4[] = {gm[0]-u[0]*umu, gm[1]-u[1]*umu,
                    gm[2]-u[2]*umu, gm[3]-u[3]*umu};

  // Delta^{mu alpha} Delta^{nu beta} * A_{alpha} * A_{beta}
  real firstTerm = dot(Delta4[mu], A) * dot(Delta4[nu], A);

  real4 temp = (real4)(dot(Delta4[0], A), dot(Delta4[1], A),
                       dot(Delta4[2], A), dot(Delta4[3], A));

  // -1/3.0 * Delta[mu][nu]  Delta^{alpha beta} * A_{alpha} * A_{beta}
  real secondTerm = -1.0f/3.0f*(gmn[mu][nu] - u[mu]*u[nu]) * dot(temp, A);

  return firstTerm + secondTerm;
  //return 0.0f;
}


inline real PiSigma(int lam, int mu, int nu, __private real pimn[10], __private real Sigma[10], __private real u[4])
{
  /** pi_{\mu}^{lam} **/
  real4 A = (real4)(pimn[idx(0, lam)], -pimn[idx(1, lam)], -pimn[idx(2, lam)],
                    -pimn[idx(3, lam)]);

  real4 B = (real4)(Sigma[idx(0, lam)], -Sigma[idx(1, lam)], -Sigma[idx(2, lam)],
                    -Sigma[idx(3, lam)]);

  real4 umu = (real4)(u[0], u[1], u[2], u[3]);

  real4 Delta4[] = {gm[0]-u[0]*umu, gm[1]-u[1]*umu,
                    gm[2]-u[2]*umu, gm[3]-u[3]*umu};

  // Delta^{mu alpha} Delta^{nu beta} * A_{alpha} * A_{beta}
  real firstTerm1 = dot(Delta4[mu], A) * dot(Delta4[nu], B);
  real firstTerm2 = dot(Delta4[mu], B) * dot(Delta4[nu], A);

  real4 temp = (real4)(dot(Delta4[0], B), dot(Delta4[1], B),
                       dot(Delta4[2], B), dot(Delta4[3], B));

  // -1/3.0 * Delta[mu][nu]  Delta^{alpha beta} * A_{alpha} * A_{beta}
  real secondTerm = -1.0f/3.0f*(gmn[mu][nu] - u[mu]*u[nu]) * dot(temp, A);

  real res =0.5f*(firstTerm1 + firstTerm2)  + secondTerm;
   
  res = isnan(res)? 0.0f : res;  //sometimes , 0*0 = nan
  return res;

}













/** update d_pinew with d_pi1, d_pistep and d_Src
where d_pi1 is for u0*d_pi as Q0_old
d_pistep is for src term in Runge Kutta method,
which is d_pi[1] for step==1 and d_pi[2] for step==2
d_ev[1] is the ev_visc 
d_ev[2] is ev_ideal*+correction at step==1
and ev_visc* at step==2
d_udiff is the correction from previous step
**/

__kernel void update_pimn(
    __global real * d_pinew,
    __global real * d_goodcell,
    __global real * d_pi1,
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
        u_new = u_old; //if step==1, we predict the flow velocity.
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
    //real local_etaos = etaos(temperature);
    real local_etaos = CETA(temperature);
    real local_pr = eos_P(ed_step, nb_step, eos_table);
    real local_s = eos_s(ed_step, nb_step, eos_table);
#ifdef BARYON_ON
    real etav = hbarc * CETA(temperature) * (ed_step + local_pr)/(max(temperature,acu));
#else
    real etav = hbarc * CETA(temperature) * local_s;
#endif
    // GeV*fm^{-2}
    real one_over_taupi = temperature/(5.0f*max(acu, local_etaos)*hbarc);
    real one_over_3DT = 1.0/(3.0*DT);





    one_over_taupi = min(one_over_3DT,one_over_taupi);


    //real one_over_taupi = temperature/(3.0f*max(acu, local_etaos)*hbarc);
    // fm^-1 


// the following definition is switchon if cfg.gubser_visc_test is set to True
//#define GUBSER_VISC_TEST
#ifdef GUBSER_VISC_TEST
    // for gubser visc solution test, use EOS: P=e/3, T=e**1/4, S=e**1/3
    // where e is in units of fm^{-4}
    real ETAOS = ETAOS_YMIN;
    real etavH = ETAOS * 4.0f/3.0f;
    real taupiH = LAM1*LAM1*etavH/3.0f;
    one_over_taupi = 1.0f/(taupiH*pow(ed_step, -0.25f));
    real coef_pipi = LAM1/ed_step;
    etav = ETAOS * 4.0f/3.0f * pow(ed_step, 0.75f);
    ////////////////////////////////
#endif

    real pi2[10];
    for ( int mn=0; mn < 10; mn ++ ) {
        pi2[mn] = d_pistep[10*I+mn]; //old_one
    }
    
    real sigma_nb[10];
    real sigma1;
    //calculate sigam^{\mu\nu} for sigma_pi_coupling
    for ( int mu = 0; mu < 4; mu ++ )
        for ( int nu = mu; nu < 4; nu ++ ) {
            sigma1 = dot(gm[mu], dalpha_u[nu]) + 
                            dot(gm[nu], dalpha_u[mu]) -
                            (u[mu]*DU[nu] + u[nu]*DU[mu]) -
                            2.0f/3.0f*(gmn[mu][nu]-u[mu]*u[nu])*theta;

            if ( u[0] > 10000.0f ) {
                sigma1 = 0.0f;
            }
            sigma_nb[idx(mu,nu)] = sigma1;
            
    }
     
    real sigma;
    real max_pimn_abs = 0.0f;
    for ( int mu = 0; mu < 4; mu ++ )
        for ( int nu = mu; nu < 4; nu ++ ) {
            sigma = dot(gm[mu], dalpha_u[nu]) + 
                            dot(gm[nu], dalpha_u[mu]) -
                            (u[mu]*DU[nu] + u[nu]*DU[mu]) -
                            2.0f/3.0f*(gmn[mu][nu]-u[mu]*u[nu])*theta;

            int mn = idx(mu, nu);
            //// set the sigma^{mu nu} and theta 0 when ed is too small
            if ( u[0] > 10000.0f ) {
                sigma = 0.0f;
            }
            
            real piNS = etav*sigma;

            real pi_old = d_pi1[10*I + mn] * u_old.s0;

            // /** step==1: Q' = Q0 + Src*DT
            //     step==2: Q  = Q0 + (Src(Q0)+Src(Q'))*DT/2
            // */

            real src = - pi2[mn]*theta/3.0f - pi2[mn]*u[0]/tau;
            src -= (u[mu]*pi2[idx(nu,0)] + u[nu]*pi2[idx(mu,0)])*DU[0]*gmn[0][0];
            src -= (u[mu]*pi2[idx(nu,1)] + u[nu]*pi2[idx(mu,1)])*DU[1]*gmn[1][1];
            src -= (u[mu]*pi2[idx(nu,2)] + u[nu]*pi2[idx(mu,2)])*DU[2]*gmn[2][2];
            src -= (u[mu]*pi2[idx(nu,3)] + u[nu]*pi2[idx(mu,3)])*DU[3]*gmn[3][3];
            
            // PiPi_couple
           real pr_step = eos_P(ed_step, nb_step, eos_table);
           real coef_pipi1 = (36.0f)/(70.f*(ed_step + pr_step+acu));
           src += one_over_taupi*coef_pipi1*(PiPi(0, mu, nu, pi2, u) 
               -PiPi(1, mu, nu, pi2, u)
               -PiPi(2, mu, nu, pi2, u)
               -PiPi(3, mu, nu, pi2, u));



           // Pi_Sigma_couple
                       
           real coef_pisigma1 = (5.0f)/(7.0f);
           src -= coef_pisigma1*(PiSigma(0, mu, nu, pi2, sigma_nb, u) 
               -PiSigma(1, mu, nu, pi2, sigma_nb, u)
               -PiSigma(2, mu, nu, pi2, sigma_nb, u)
               -PiSigma(3, mu, nu, pi2, sigma_nb, u));



#ifdef GUBSER_VISC_TEST
            src -= one_over_taupi*coef_pipi*(PiPi(0, mu, nu, pi2, u) 
                -PiPi(1, mu, nu, pi2, u)
                -PiPi(2, mu, nu, pi2, u)
                -PiPi(3, mu, nu, pi2, u));
#endif
            real src2 = d_Src[idn(I, mn)] ;
            d_Src[idn(I, mn)] += src;

            // use implicit method for stiff term; 
            pi_old = (pi_old + d_Src[idn(I, mn)]*DT/step +
                      DT*one_over_taupi*piNS) / (u_new.s0 + DT*one_over_taupi);
            
            //pi_old = (pi_old + (d_Src[idn(I, mn)]+one_over_taupi*piNS)*DT/step) / (u_new.s0 + DT*one_over_taupi);

            if ( fabs(pi_old) > max_pimn_abs ) max_pimn_abs = fabs(pi_old);

            d_pinew[10*I + mn] = pi_old;



          
    }

    //real pre = eos_P(e_v2.s0,nb_v2, eos_table);
    //real T00 = (e_v2.s0 + pre)*u_new.s0*u_new.s0 - pre;
    // comments from wxy, u_new and e_v2.s0 is not same time at step=1
    //real T00 = (e_v2.s0 + pre)*u_new.s0*u_new.s0;
    //real T00 = sqrt(e_v2.s0*e_v2.s0 + 3.0f*pre*pre);

    real pre = eos_P(ed_step,nb_step, eos_table);
    real T00 = (ed_step + pre)*u[0]*u[0] - pre;
    d_goodcell[I] = 1.0f;
    
    if ( max_pimn_abs > T00 ) {
       for ( int mn = 0; mn <10; mn++ ) {
           d_pinew[10*I + mn] = 0.0f;
       }
       d_goodcell[I] = 0.0f;
    }


    
    
    // real eps_scale = 0.1;
    // real xi = 0.01;
    // real factor = 100.0*(1.0/(exp(-(ed_step - eps_scale)/xi)+1.0) 
    // - 1.0/(exp(( eps_scale)/xi)+1.0));
    
    // real pi_00 = d_pinew[10*I + 0];
    // real pi_01 = d_pinew[10*I + 1];
    // real pi_02 = d_pinew[10*I + 2];
    // real pi_03 = d_pinew[10*I + 3];
    // real pi_11 = d_pinew[10*I + 4];
    // real pi_12 = d_pinew[10*I + 5];
    // real pi_13 = d_pinew[10*I + 6];
    // real pi_22 = d_pinew[10*I + 7];
    // real pi_23 = d_pinew[10*I + 8];
    // real pi_33 = d_pinew[10*I + 9];
    
    // real pisize = (pi_00*pi_00 + pi_11*pi_11 + pi_22*pi_22 + pi_33*pi_33
    //      - 2.*(pi_01*pi_01 + pi_02*pi_02 + pi_03*pi_03)
    //      + 2.*(pi_12*pi_12 + pi_13*pi_13 + pi_23*pi_23));
    

    // real eq_size = ed_step*ed_step + 3.*pre*pre;
    // real rho_shear = sqrt(fabs(pisize)/eq_size)/factor;
    // real rho_shear_max = 0.1;
    // if (rho_shear > rho_shear_max) {
    //     for (int ele = 0; ele < 10; ele++) {
    //         d_pinew[10*I + ele] = (rho_shear_max/rho_shear)*d_pinew[10*I + ele];
    //     }
    // }

   

    


    }

}



/** get the u^mu difference between u_{n} and u_{n-1}
 * the d_udiff will be used in the next prediction step */

__kernel void get_udiff(
    __global real4 * d_udiff,
    __global real * d_mubdiff,
    __global real4 * d_ev0,
    __global real4 * d_ev1,
    __global real  * d_nb0,
    __global real  * d_nb1,
    __global real  * eos_table)

{
    int I = get_global_id(0); 
    real4 e_v0 = d_ev0[I];
    real4 e_v1 = d_ev1[I];
    
    real nb_v0 = d_nb0[I];
    real nb_v1 = d_nb1[I];

    d_udiff[I] =  umu4(e_v1) - umu4(e_v0);
    d_mubdiff[I] = ( eos_mu(e_v1.s0,nb_v1,eos_table)/(max(eos_T(e_v1.s0,nb_v1,eos_table),acu))
                - eos_mu(e_v0.s0,nb_v0,eos_table)/(max(eos_T(e_v0.s0,nb_v0,eos_table),acu)));
    

}

