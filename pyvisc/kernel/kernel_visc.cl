#include<helper.h>

#define IDX_PI 11

/* d_SrcT: Src for viscous hydro dT^{tau mu}/dt terms
   d_SrcN: Src for conserved charge densities
   d_ev:   (ed, vx, vy, veta)
   d_pi:   (10 for pi^{mu nu} and 1 for PI)
*/

__kernel void kt_src_christoffel(
             __global real4 * d_SrcT,
             __global real * d_SrcJ,
             __global real4 * d_ev,
             __global real * d_nb,
             __global real * d_pi,
             __global real * d_bulkpr,
             __global real * eos_table,
             const real tau,
             const int step) {
    int I = get_global_id(0);

    if ( I < NX*NY*NZ ) {
        if ( step == 1 ) {
            d_SrcT[I] = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
            d_SrcJ[I] = 0.0f;
        }
        real4 e_v = d_ev[I];
        real ed = e_v.s0;
        real vx = e_v.s1;
        real vy = e_v.s2;
        real vz = e_v.s3;
        real nb = d_nb[I];
        real u0 = gamma(vx, vy, vz);


        real pressure = eos_P(ed,nb,eos_table) ;
        // Tzz_tilde = T^{eta eta} * tau^2; no 1/tau in vz
        real Tzz_tilde = (ed + pressure + d_bulkpr[I])*u0*u0*vz*vz + pressure+d_bulkpr[I]
                         + d_pi[idn(I, idx(3, 3))];
        real Ttz_tilde = (ed + pressure+d_bulkpr[I])*u0*u0*vz
                         + d_pi[idn(I, idx(0, 3))];
        d_SrcT[I] = d_SrcT[I] - (real4)(Tzz_tilde, 0.0f, 0.0f, Ttz_tilde);




    }
}



real4 kt1d_visc(real4 ev[5], real nb[5], real4 pim0[5], real4 pimi[5],
           real vip_half, real vim_half,
           real tau, int along, __global real * eos_table);

real4 kt1d_visc2(real4 ev[5], real nb[5], real4 pim0[5], real4 pimi[5],
           real bulkpr[5], real vip_half, real vim_half,
           real tau, int along, __global real * eos_table);


real4 kt1d_visc(real4 ev[5], real nb[5], real4 pim0[5], real4 pimi[5],
           real vip_half, real vim_half,real tau, int along, __global real *eos_table) {
   real pr[5];
   real4 Q[5];
   for ( int i = 0; i < 5; i ++ ) {
       pr[i] = eos_P(ev[i].s0, nb[i], eos_table);
       Q[i] = tau * (t0m(ev[i], pr[i]) + pim0[i]);
   }

   real4 DA0, DA1;
   int i = 2;
   DA0 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));
   i = 3;
   DA1 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));
 
   real4  AL = Q[2] + 0.5f * DA0;
   real4  AR = Q[3] - 0.5f * DA1;
   real pr_half = 0.5f*(pr[2] + pr[3]);
   real4 pim0_half = 0.5f*(pim0[2] + pim0[3]);
   real4 pimi_half = 0.5f*(pimi[2] + pimi[3]);
   // Flux Jp = (T0m + pr*g^{tau mu} - pim0)*v^x - pr*g^{x mu} + pimi
   real4 Jp = (AR + pr_half*tau*gm[0] - tau*pim0_half)*vip_half - pr_half*tau*gm[along+1] + tau*pimi_half;
   real4 Jm = (AL + pr_half*tau*gm[0] - tau*pim0_half)*vip_half - pr_half*tau*gm[along+1] + tau*pimi_half;

   real4 ev_half = 0.5f*(ev[2]+ev[3]);
   real nb_half = 0.5f*(nb[2]+nb[3]);
   // maximum local propagation speed at i+1/2
   real lam = maxPropagationSpeed(ev_half,nb_half, vip_half, eos_table);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real4 src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   DA1 = DA0;  // reuse the previous calculate value
   i = 1;
   DA0 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   AL = Q[1]   + 0.5f * DA0;
   AR = Q[2] - 0.5f * DA1;

   // pr_half = tau*pr(i+1/2)
   pr_half = 0.5f*(pr[1] + pr[2]);
   pim0_half = 0.5f*(pim0[1] + pim0[2]);
   pimi_half = 0.5f*(pimi[1] + pimi[2]);

   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
   Jp = (AR + pr_half*tau*gm[0] - tau*pim0_half)*vim_half - pr_half*tau*gm[along+1] + tau*pimi_half;
   Jm = (AL + pr_half*tau*gm[0] - tau*pim0_half)*vim_half - pr_half*tau*gm[along+1] + tau*pimi_half;

   // maximum local propagation speed at i-1/2
   ev_half = 0.5f*(ev[2] + ev[1]);
   nb_half = 0.5f*(nb[2] + nb[1]);
   lam = maxPropagationSpeed(ev_half,nb_half, vim_half, eos_table);
   // second part of kt1d; final results = src[i] - src[i-1]
   
   
   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);


   return src;
}

real4 kt1d_visc2(real4 ev[5], real nb[5], real4 pim0[5], real4 pimi[5], real bulkpr[5],
           real vip_half, real vim_half,real tau, int along, __global real *eos_table) {
   real pr[5];
   real4 Q[5];
   for ( int i = 0; i < 5; i ++ ) {
       pr[i] = eos_P(ev[i].s0, nb[i], eos_table);
       Q[i] = tau * (t0m(ev[i], pr[i]+bulkpr[i]) + pim0[i]);
   }
   
   real4 DA0, DA1;
   real4 DA0_ev,DA1_ev;
   real DA0_nb,DA1_nb;
   real DA0_bulkpr,DA1_bulkpr;
   real4 DA0_pim0,DA1_pim0;
   real4 DA0_pimi,DA1_pimi;

   int i = 2;
   DA0 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   DA0_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));

   DA0_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));
    
   DA0_bulkpr = minmod(0.5f*(bulkpr[i+1] - bulkpr[i-1]),
           minmod(THETA*(bulkpr[i+1] - bulkpr[i]), THETA*(bulkpr[i] - bulkpr[i-1])));

   DA0_pim0 = minmod4(0.5f*(pim0[i+1] - pim0[i-1]),
            minmod4(THETA*(pim0[i+1] - pim0[i]), THETA*(pim0[i] - pim0[i-1])));

   DA0_pimi = minmod4(0.5f*(pimi[i+1] - pimi[i-1]),
            minmod4(THETA*(pimi[i+1] - pimi[i]), THETA*(pimi[i] - pimi[i-1])));

   i = 3;
   DA1 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   DA1_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));

   DA1_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));

   DA1_bulkpr = minmod(0.5f*(bulkpr[i+1] - bulkpr[i-1]),
           minmod(THETA*(bulkpr[i+1] - bulkpr[i]), THETA*(bulkpr[i] - bulkpr[i-1])));

   DA1_pim0 = minmod4(0.5f*(pim0[i+1] - pim0[i-1]),
            minmod4(THETA*(pim0[i+1] - pim0[i]), THETA*(pim0[i] - pim0[i-1])));
   DA1_pimi = minmod4(0.5f*(pimi[i+1] - pimi[i-1]),
            minmod4(THETA*(pimi[i+1] - pimi[i]), THETA*(pimi[i] - pimi[i-1])));


   real4 evL = ev[2] + 0.5f*DA0_ev;
   real4 evR = ev[3] - 0.5f*DA1_ev;
  
   real vL[3] = {evL.s1, evL.s2, evL.s3};
   real vR[3] = {evR.s1, evR.s2, evR.s3};
   
   real vv1 = vL[0]*vL[0] + vL[1]*vL[1] +vL[2]*vL[2]; 
   real vv2 = vR[0]*vR[0] + vR[1]*vR[1] +vR[2]*vR[2]; 
   real vv  = ev[2].s1*ev[2].s1 + ev[2].s2*ev[2].s2 + ev[2].s3*ev[2].s3;
   real vv3  = ev[1].s1*ev[1].s1 + ev[1].s2*ev[1].s2 + ev[1].s3*ev[1].s3;
   real vv4  = ev[3].s1*ev[3].s1 + ev[3].s2*ev[3].s2 + ev[3].s3*ev[3].s3;

   real nbL = nb[2] + 0.5f*DA0_nb;
   real nbR = nb[3] - 0.5f*DA1_nb;

   real prL = eos_P(evL.s0,nbL,eos_table);
   real prR = eos_P(evR.s0,nbR,eos_table);

   real bulkprL = 0.5f*(bulkpr[2] + bulkpr[3]);
   real bulkprR = 0.5f*(bulkpr[2] + bulkpr[3]);

   //real4 pim0L = pim0[2] + 0.5f*DA0_pim0;
   //real4 pim0R = pim0[3] - 0.5f*DA1_pim0;

   real4 pim0L = 0.5f*(pim0[2] + pim0[3]);
   real4 pim0R = 0.5f*(pim0[2] + pim0[3]);
   
   

   //real4 pimiL = pimi[2] + 0.5f*DA0_pimi;
   //real4 pimiR = pimi[3] - 0.5f*DA1_pimi;

   real4 pimiL = 0.5f*(pimi[2] + pimi[3]);
   real4 pimiR = 0.5f*(pimi[2] + pimi[3]);

   real4  AL = Q[2] + 0.5f * DA0;
   real4  AR = Q[3] - 0.5f * DA1;

   // Flux Jp = (T0m + pr*g^{tau mu} - pim0)*v^x - pr*g^{x mu} + pimi
   real4 Jp = (AR + (prR+bulkprR)*tau*gm[0] - tau*pim0R)*vR[along] - (prR+bulkprR)*tau*gm[along+1] + tau*pimiR;
   real4 Jm = (AL + (prL+bulkprL)*tau*gm[0] - tau*pim0L)*vL[along] - (prL+bulkprL)*tau*gm[along+1] + tau*pimiL;

   // maximum local propagation speed at i+1/2
   real lamL = maxPropagationSpeed(evL,nbL, vL[along], eos_table);
   real lamR = maxPropagationSpeed(evR,nbR, vR[along], eos_table);
   
   real lam = max(lamL,lamR);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real4 src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);
   real4 src1 = src;

   DA1 = DA0;  // reuse the previous calculate value
   DA1_ev = DA0_ev;
   DA1_nb = DA0_nb;
   DA1_bulkpr = DA0_bulkpr;
   DA1_pim0 = DA0_pim0;
   DA1_pimi = DA0_pimi;

   i=1;
   DA0 = minmod4(0.5f*(Q[i+1] - Q[i-1]),
           minmod4(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));
   DA0_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));
   DA0_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));
   DA0_bulkpr = minmod(0.5f*(bulkpr[i+1] - bulkpr[i-1]),
           minmod(THETA*(bulkpr[i+1] - bulkpr[i]), THETA*(bulkpr[i] - bulkpr[i-1])));

   DA0_pim0 = minmod4(0.5f*(pim0[i+1] - pim0[i-1]),
            minmod4(THETA*(pim0[i+1] - pim0[i]), THETA*(pim0[i] - pim0[i-1])));

   DA0_pimi = minmod4(0.5f*(pimi[i+1] - pimi[i-1]),
            minmod4(THETA*(pimi[i+1] - pimi[i]), THETA*(pimi[i] - pimi[i-1])));

   evL = ev[1] + 0.5f*DA0_ev;
   evR = ev[2] - 0.5f*DA1_ev;

   nbL = nb[1] + 0.5f*DA0_nb;
   nbR = nb[2] - 0.5f*DA1_nb;

   vL[0] = evL.s1;
   vL[1] = evL.s2;
   vL[2] = evL.s3;

   vR[0] = evR.s1;
   vR[1] = evR.s2;
   vR[2] = evR.s3;

//    pim0L = pim0[1] + 0.5f*DA0_pim0;
//    pim0R = pim0[2] - 0.5f*DA1_pim0;

//    pimiL = pimi[1] + 0.5f*DA0_pimi;
//    pimiR = pimi[2] - 0.5f*DA1_pimi;

   pimiL = 0.5f*(pimi[1] + pimi[2]);
    pimiR = 0.5f*(pimi[1] + pimi[2]);

   pimiL = 0.5f*(pimi[1] + pimi[2]);
    pimiR = 0.5f*(pimi[1] + pimi[2]);

   prL = eos_P(evL.s0,nbL,eos_table);
   prR = eos_P(evR.s0,nbR,eos_table);

   bulkprL = 0.5f*(bulkpr[1] + bulkpr[2]);
   bulkprR = 0.5f*(bulkpr[1] + bulkpr[2]);


   AL = Q[1] + 0.5f * DA0;
   AR = Q[2] - 0.5f * DA1;



   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
   Jp = (AR + (prR+bulkprR)*tau*gm[0] - tau*pim0R)*vR[along] - (prR+bulkprR)*tau*gm[along+1] + tau*pimiR;
   Jm = (AL + (prL+bulkprL)*tau*gm[0] - tau*pim0L)*vL[along] - (prL+bulkprL)*tau*gm[along+1] + tau*pimiL;

   // maximum local propagation speed at i-1/2
   lamL = maxPropagationSpeed(evL,nbL, vL[along], eos_table);
   lamR = maxPropagationSpeed(evR,nbR, vR[along], eos_table);
   lam = min(lamL,lamR);

   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   return src;
}


// output: d_Src; all the others are input
__kernel void kt_src_alongx(
             __global real4 * d_Src,
             __global real4 * d_ev,
             __global real  * d_nb,
             __global real  * d_pi,
             __global real  * d_bulkpr,
             __global real  * eos_table,
             const real tau) {

    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    __local real4 pim0[NX+4];
    __local real4 pimi[NX+4];
    __local real  nb[NX+4]; 
    __local real  bulkpr[NX+4];

    // Use num of threads = BSZ to compute src for NX elements
    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[I+2] = d_ev[IND];
        nb[I+2] = d_nb[IND];
        bulkpr[I+2] = d_bulkpr[IND];
        pim0[I+2] = (real4)(d_pi[idn(IND, 0)],
                            d_pi[idn(IND, 1)],
                            d_pi[idn(IND, 2)],
                            d_pi[idn(IND, 3)]);
        pimi[I+2] = (real4)(d_pi[idn(IND, idx(1, 0))],
                            d_pi[idn(IND, idx(1, 1))],
                            d_pi[idn(IND, idx(1, 2))],
                            d_pi[idn(IND, idx(1, 3))]);
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


       pim0[0] = pim0[2];
       pim0[1] = pim0[2];
       pim0[NX+3] = pim0[NX+1];
       pim0[NX+2] = pim0[NX+1];

       pimi[0] = pimi[2];
       pimi[1] = pimi[2];
       pimi[NX+3] = pimi[NX+1];
       pimi[NX+2] = pimi[NX+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    // load the following data from local to private memory
    for ( int I = get_global_id(0); I < NX; I = I + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = I + 2;
        real4 ev_[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};
        real nb_[5] ={nb[i-2], nb[i-1], nb[i], nb[i+1], nb[i+2]};

        real4 pim0_[5] ={pim0[i-2], pim0[i-1], pim0[i], pim0[i+1], pim0[i+2]};
        real4 pimi_[5] ={pimi[i-2], pimi[i-1], pimi[i], pimi[i+1], pimi[i+2]};

        real bulkpr_[5] ={bulkpr[i-2], bulkpr[i-1], bulkpr[i], bulkpr[i+1], bulkpr[i+2]};
        real vip_half = 0.5f*(ev_[2].s1 + ev_[3].s1);
        real vim_half = 0.5f*(ev_[1].s1 + ev_[2].s1);
        real4 src1 = d_Src[IND];



     real4 ev_half = 0.5f*(ev_[2]+ev_[3]);
        real lam1 = maxPropagationSpeed(ev_half,0.0f, vip_half, eos_table);
         real cs2 = eos_CS2(ev_half.s0,0.0f, eos_table);

       // d_Src[IND] = d_Src[IND] - kt1d_visc(ev_,nb_, pim0_, pimi_, vip_half,
       //                     vim_half, tau,ALONG_X, eos_table)/DX;

        d_Src[IND] = d_Src[IND] - kt1d_visc2(ev_,nb_, pim0_, pimi_,bulkpr_, vip_half,
                              vim_half, tau,ALONG_X, eos_table)/DX;






    }
}


// output: d_Src; all the others are input
__kernel void kt_src_alongy(
             __global real4 * d_Src,
             __global real4 * d_ev,
             __global real  * d_nb,
             __global real  * d_pi,
             __global real  * d_bulkpr,
             __global real  * eos_table,
             const real tau) {
    int I = get_global_id(0);
    int K = get_global_id(2);
    __local real4 ev[NY+4];
    __local real4 pim0[NY+4];
    __local real4 pimi[NY+4];
    __local real  nb[NY+4];
    __local real  bulkpr[NY+4];


    // Use num of threads = BSZ to compute src for NX elements
    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[J+2] = d_ev[IND];
        nb[J+2] = d_nb[IND];
        bulkpr[J+2] = d_bulkpr[IND];
        pim0[J+2] = (real4)(d_pi[idn(IND, 0)],
                            d_pi[idn(IND, 1)],
                            d_pi[idn(IND, 2)],
                            d_pi[idn(IND, 3)]);
        pimi[J+2] = (real4)(d_pi[idn(IND, idx(2, 0))],
                            d_pi[idn(IND, idx(2, 1))],
                            d_pi[idn(IND, idx(2, 2))],
                            d_pi[idn(IND, idx(2, 3))]);
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

       pim0[0] = pim0[2];
       pim0[1] = pim0[2];
       pim0[NY+3] = pim0[NY+1];
       pim0[NY+2] = pim0[NY+1];

       pimi[0] = pimi[2];
       pimi[1] = pimi[2];
       pimi[NY+3] = pimi[NY+1];
       pimi[NY+2] = pimi[NY+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for ( int J = get_global_id(1); J < NY; J = J + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = J + 2;

        real4 ev_[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};
        real  nb_[5] ={nb[i-2], nb[i-1], nb[i], nb[i+1], nb[i+2]};
        real  bulkpr_[5] ={bulkpr[i-2], bulkpr[i-1], bulkpr[i], bulkpr[i+1], bulkpr[i+2]};
        real4 pim0_[5] ={pim0[i-2], pim0[i-1], pim0[i], pim0[i+1], pim0[i+2]};
        real4 pimi_[5] ={pimi[i-2], pimi[i-1], pimi[i], pimi[i+1], pimi[i+2]};

        real vip_half = 0.5f*(ev_[2].s2 + ev_[3].s2);
        real vim_half = 0.5f*(ev_[1].s2 + ev_[2].s2);
        //d_Src[IND] = d_Src[IND] - kt1d_visc(ev_, nb_,pim0_, pimi_, vip_half,
        //                        vim_half, tau, ALONG_Y, eos_table)/DY;
       
        //real4 src1 = d_Src[IND]; 
        d_Src[IND] = d_Src[IND] - kt1d_visc2(ev_, nb_,pim0_, pimi_,bulkpr_, vip_half,
                                  vim_half, tau, ALONG_Y, eos_table)/DY;



    }
}

// output: d_Src; all the others are input
__kernel void kt_src_alongz(
             __global real4 * d_Src,
	     __global real4 * d_ev,
             __global real  * d_nb,
             __global real  * d_pi,
             __global real  * d_bulkpr,
             __global real  * eos_table,
	     const real tau) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    __local real4 ev[NZ+4];
    __local real4 pim0[NZ+4];
    __local real4 pimi[NZ+4];
    __local real  nb[NZ+4];
    __local real  bulkpr[NZ+4];

    // Use num of threads = BSZ to compute src for NX elements
    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        ev[K+2] = d_ev[IND];
        nb[K+2] = d_nb[IND];
        bulkpr[K+2] = d_bulkpr[IND];

        pim0[K+2] = (real4)(d_pi[idn(IND, 0)],
                            d_pi[idn(IND, 1)],
                            d_pi[idn(IND, 2)],
                            d_pi[idn(IND, 3)]);
        pimi[K+2] = (real4)(d_pi[idn(IND, idx(3, 0))],
                            d_pi[idn(IND, idx(3, 1))],
                            d_pi[idn(IND, idx(3, 2))],
                            d_pi[idn(IND, idx(3, 3))]);
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

       pim0[0] = pim0[2];
       pim0[1] = pim0[2];
       pim0[NZ+3] = pim0[NZ+1];
       pim0[NZ+2] = pim0[NZ+1];

       pimi[0] = pimi[2];
       pimi[1] = pimi[2];
       pimi[NZ+3] = pimi[NZ+1];
       pimi[NZ+2] = pimi[NZ+1];
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for ( int K = get_global_id(2); K < NZ; K = K + BSZ ) {
        int IND = I*NY*NZ + J*NZ + K;
        int i = K + 2;
        // load from local to private memory
        real4 ev_[5] ={ev[i-2], ev[i-1], ev[i], ev[i+1], ev[i+2]};
        real4 pim0_[5] ={pim0[i-2], pim0[i-1], pim0[i], pim0[i+1], pim0[i+2]};
        real4 pimi_[5] ={pimi[i-2], pimi[i-1], pimi[i], pimi[i+1], pimi[i+2]};
        real nb_[5] ={nb[i-2], nb[i-1], nb[i], nb[i+1], nb[i+2]};
        real bulkpr_[5] ={bulkpr[i-2], bulkpr[i-1], bulkpr[i], bulkpr[i+1], bulkpr[i+2]};

        real vip_half = 0.5f*(ev_[2].s3 + ev_[3].s3);
        real vim_half = 0.5f*(ev_[1].s3 + ev_[2].s3);
        //d_Src[IND] = d_Src[IND] - kt1d_visc(ev_,nb_, pim0_, pimi_, vip_half,
        //                vim_half, tau,ALONG_Z, eos_table)/(tau*DZ);
        d_Src[IND] = d_Src[IND] - kt1d_visc2(ev_,nb_, pim0_, pimi_,bulkpr_, vip_half,
                          vim_half, tau,ALONG_Z, eos_table)/(tau*DZ);

    }
}



