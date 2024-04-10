#include<helper.h>
#pragma OPENCL EXTENSION cl_amd_printf : enable


__kernel void kt_src_christoffel(
              __global real* d_Src,
              const int step){

    int I = get_global_id(0);
    if(I < NX*NY*NZ){
        if( step ==1 ){
            d_Src[I]=0.0f;
        }

    }

}



real kt1d_visc(real4 ev[5], real nb[5],real qb0[5],real qbi[5],real vip_half, real vim_half,real tau, __global real * eos_table);
real kt1d_visc2(real4 ev[5], real nb[5],real qb0[5],real qbi[5],real vip_half, real vim_half,real tau,int along, __global real * eos_table);


real kt1d_visc(real4 ev[5], real nb[5],real qb0[5],real qbi[5],real vip_half, real vim_half,real tau, __global real * eos_table) {
   real Q[5];
   for ( int i = 0; i < 5; i ++ ) {
       Q[i] = tau*(nb[i]*gamma(ev[i].s1, ev[i].s2, ev[i].s3)+qb0[i]);
   }

   real DA0, DA1;
   int i = 2;
   DA0 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));
   i = 3;
   DA1 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));
  
   real  AL = Q[2] + 0.5f * DA0;
   real  AR = Q[3] - 0.5f * DA1;

   real qbi_half = 0.5f*(qbi[2]+qbi[3]);
   real qb0_half = 0.5f*(qb0[2]+qb0[3]);

   
   real Jp = (AR - tau*qb0_half)*vip_half + qbi_half*tau;
   real Jm = (AL - tau*qb0_half)*vip_half + qbi_half*tau;
   
 
   real4 ev_half = 0.5f*(ev[2]+ev[3]);
   real nb_half = 0.5f*(nb[2]+nb[3]);
   real lam = maxPropagationSpeed(ev_half,nb_half, vip_half, eos_table);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);
   DA1 = DA0;  // reuse the previous calculate value
   i = 1;
   DA0 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   AL = Q[1] + 0.5f * DA0;
   AR = Q[2] - 0.5f * DA1;

   qbi_half = 0.5f*(qbi[1]+qbi[2]);
   qb0_half = 0.5f*(qb0[1]+qb0[2]);

   Jp = (AR - tau*qb0_half)*vim_half + qbi_half*tau;
   Jm = (AL - tau*qb0_half)*vim_half + qbi_half*tau;


   ev_half = 0.5f*(ev[2] + ev[1]);
   nb_half = 0.5f*(nb[2] + nb[1]);
   lam = maxPropagationSpeed(ev_half,nb_half, vim_half, eos_table);

   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   return src;
}



real kt1d_visc2(real4 ev[5], real nb[5],real qb0[5],real qbi[5],real vip_half, real vim_half,real tau, int along, __global real * eos_table) {
   real Q[5];
   for ( int i = 0; i < 5; i ++ ) {
       Q[i] = tau*(nb[i]*gamma(ev[i].s1, ev[i].s2, ev[i].s3)+qb0[i]);
   }

   real DA0, DA1;
   real4 DA0_ev,DA1_ev;
   real DA0_nb,DA1_nb;
   real DA0_qb0,DA1_qb0;
   real DA0_qbi,DA1_qbi;

   int i = 2;
   DA0 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   DA0_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));

   DA0_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));

   DA0_qb0 = minmod(0.5f*(qb0[i+1] - qb0[i-1]),
            minmod(THETA*(qb0[i+1] - qb0[i]), THETA*(qb0[i] - qb0[i-1])));

   DA0_qbi = minmod(0.5f*(qbi[i+1] - qbi[i-1]),
            minmod(THETA*(qbi[i+1] - qbi[i]), THETA*(qbi[i] - qbi[i-1])));

   i = 3;
   DA1 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   DA1_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));

   DA1_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));

   DA1_qb0 = minmod(0.5f*(qb0[i+1] - qb0[i-1]),
            minmod(THETA*(qb0[i+1] - qb0[i]), THETA*(qb0[i] - qb0[i-1])));

   DA1_qbi = minmod(0.5f*(qbi[i+1] - qbi[i-1]),
            minmod(THETA*(qbi[i+1] - qbi[i]), THETA*(qbi[i] - qbi[i-1])));
 
   real4 evL = ev[2] + 0.5f*DA0_ev;
   real4 evR = ev[3] - 0.5f*DA1_ev;

   real vL[3] = {evL.s1, evL.s2, evL.s3};
   real vR[3] = {evR.s1, evR.s2, evR.s3};

   real nbL = nb[2] + 0.5f*DA0_nb;
   real nbR = nb[3] - 0.5f*DA1_nb;

   real qb0L = qb0[2] + 0.5f*DA0_qb0;
   real qb0R = qb0[3] - 0.5f*DA1_qb0;

   real qbiL = qbi[2] + 0.5f*DA0_qbi;
   real qbiR = qbi[3] - 0.5f*DA1_qbi;

   real AL = Q[2] + 0.5f * DA0;
   real AR = Q[3] - 0.5f * DA1;

   real Jp = (AR - tau*qb0R)*vR[along] + qbiR*tau;
   real Jm = (AL - tau*qb0L)*vL[along] + qbiL*tau;
   
   real lamL = maxPropagationSpeed(evL,nbL, vL[along], eos_table);
   real lamR = maxPropagationSpeed(evR,nbR, vR[along], eos_table);

   real lam = max(lamL,lamR);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   
   DA1 = DA0;  // reuse the previous calculate value
   DA1_ev = DA0_ev;
   DA1_nb = DA0_nb;
   DA1_qb0 = DA0_qb0;
   DA1_qbi = DA0_qbi;

   i = 1;
   DA0 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   DA0_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));

   DA0_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));

   DA0_qb0 = minmod(0.5f*(qb0[i+1] - qb0[i-1]),
            minmod(THETA*(qb0[i+1] - qb0[i]), THETA*(qb0[i] - qb0[i-1])));

   DA0_qbi = minmod(0.5f*(qbi[i+1] - qbi[i-1]),
            minmod(THETA*(qbi[i+1] - qbi[i]), THETA*(qbi[i] - qbi[i-1])));

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

   qb0L = qb0[1] + 0.5f*DA0_qb0;
   qb0R = qb0[2] - 0.5f*DA1_qb0;

   qbiL = qbi[1] + 0.5f*DA0_qbi;
   qbiR = qbi[2] - 0.5f*DA1_qbi;
 

   AL = Q[1] + 0.5f * DA0;
   AR = Q[2] - 0.5f * DA1;

   Jp = (AR - tau*qb0R)*vR[along] + qbiR*tau;
   Jm = (AL - tau*qb0L)*vL[along] + qbiL*tau;
   lamL = maxPropagationSpeed(evL,nbL, vL[along], eos_table);
   lamR = maxPropagationSpeed(evR,nbR, vR[along], eos_table);
   lam = max(lamL,lamR);
   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   return src;
}



real kt1d_visc3(real4 ev[5], real nb[5],real qb0[5],real qbi[5],real vip_half, real vim_half,real tau, int along, __global real * eos_table) {
   real Q[5];
   
   for ( int i = 0; i < 5; i ++ ) {
        Q[i] = tau*(nb[i]*gamma(ev[i].s1, ev[i].s2, ev[i].s3)+qb0[i]);
        
   }

   real DA0, DA1;
   real4 DA0_ev,DA1_ev;
   real DA0_nb,DA1_nb;
   real DA0_qb0,DA1_qb0;
   real DA0_qbi,DA1_qbi;

   int i = 2;
   DA0 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   DA0_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));

   DA0_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));


   i = 3;
   DA1 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   DA1_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));

   DA1_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));


   real qbi_half = 0.5f*(qbi[2]+qbi[3]);
   real qb0_half = 0.5f*(qb0[2]+qb0[3]);


   real4 evL = ev[2] + 0.5f*DA0_ev;
   real4 evR = ev[3] - 0.5f*DA1_ev;

   real vL[3] = {evL.s1, evL.s2, evL.s3};
   real vR[3] = {evR.s1, evR.s2, evR.s3};

   real nbL = nb[2] + 0.5f*DA0_nb;
   real nbR = nb[3] - 0.5f*DA1_nb;



   real AL = Q[2] + 0.5f * DA0 + tau*qb0_half;
   real AR = Q[3] - 0.5f * DA1 + tau*qb0_half;

   real Jp = (AR- tau*qb0_half)*vR[along] + qbi_half*tau;
   real Jm = (AL- tau*qb0_half)*vL[along] + qbi_half*tau;


   
   real lamL = maxPropagationSpeed(evL,nbL, vL[along], eos_table);
   real lamR = maxPropagationSpeed(evR,nbR, vR[along], eos_table);

   real lam = max(lamL,lamR);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   
   DA1 = DA0;  // reuse the previous calculate value
   DA1_ev = DA0_ev;
   DA1_nb = DA0_nb;


   i = 1;
   DA0 = minmod(0.5f*(Q[i+1] - Q[i-1]),
           minmod(THETA*(Q[i+1] - Q[i]), THETA*(Q[i] - Q[i-1])));

   DA0_ev = minmod4(0.5f*(ev[i+1] - ev[i-1]),
           minmod4(THETA*(ev[i+1] - ev[i]), THETA*(ev[i] - ev[i-1])));

   DA0_nb = minmod(0.5f*(nb[i+1] - nb[i-1]),
           minmod(THETA*(nb[i+1] - nb[i]), THETA*(nb[i] - nb[i-1])));


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

 
   qbi_half = 0.5f*(qbi[1]+qbi[2]);
   qb0_half = 0.5f*(qb0[1]+qb0[2]);

   AL = Q[1] + 0.5f * DA0+tau*qb0_half;
   AR = Q[2] - 0.5f * DA1+tau*qb0_half;

  Jp = (AR - tau*qb0_half)*vR[along] + qbi_half*tau;
  Jm = (AL - tau*qb0_half)*vL[along] + qbi_half*tau;




   lamL = maxPropagationSpeed(evL,nbL, vL[along], eos_table);
   lamR = maxPropagationSpeed(evR,nbR, vR[along], eos_table);
   lam = max(lamL,lamR);
   real dqdx = 0.5*( qbi[1]+qbi[2] ) - 0.5*( qbi[2]+qbi[3] ) ; 
   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);

   return src;
}


__kernel void kt_src_alongx(
              __global real* d_Src,
              __global real*  d_nb,
              __global real*  d_qb,
              __global real4* d_ev,
              const real tau,
              __global real* eos_table,
	      const int step){



    int J = get_global_id(1);
    int K = get_global_id(2);
    __local real4 ev[NX+4];
    __local real  nb[NX+4];
    __local real  qb0[NX+4];
    __local real  qbi[NX+4];


    for(int I = get_global_id(0);I < NX; I = I+BSZ){
        int IND = I*NY*NZ+J*NZ+K ;
        ev[I+2] = d_ev[IND];
        nb[I+2] = d_nb[IND];
        qb0[I+2] = d_qb[idn2(IND,0)];
        qbi[I+2] = d_qb[idn2(IND,1)];


    }

    barrier(CLK_LOCAL_MEM_FENCE);

   if(get_local_id(0) == 0){
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NX+3] = ev[NX+1];
       ev[NX+2] = ev[NX+1];

       nb[0] = nb[2];
       nb[1] = nb[2];
       nb[NX+3] = nb[NX+1];
       nb[NX+2] = nb[NX+1];

       qb0[0] = qb0[2];
       qb0[1] = qb0[2];
       qb0[NX+3] = qb0[NX+1];
       qb0[NX+2] = qb0[NX+1];

       qbi[0] = qbi[2];
       qbi[1] = qbi[2];
       qbi[NX+3] = qbi[NX+1];
       qbi[NX+2] = qbi[NX+1];
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   for(int I = get_global_id(0);I < NX; I = I + BSZ){

      int IND = I*NY*NZ + J*NZ +K;
      int i = I+2;
      real4 ev_[5] = {ev[i-2],ev[i-1],ev[i],ev[i+1],ev[i+2]};
      real  nb_[5] = {nb[i-2],nb[i-1],nb[i],nb[i+1],nb[i+2]};
      real  qb0_[5] = {qb0[i-2],qb0[i-1],qb0[i],qb0[i+1],qb0[i+2]};
      real  qbi_[5] = {qbi[i-2],qbi[i-1],qbi[i],qbi[i+1],qbi[i+2]};


      real vip_half = 0.5f*(ev_[2].s1+ev_[3].s1);
      real vim_half = 0.5f*(ev_[1].s1+ev_[2].s1);
   
      d_Src[IND] = d_Src[IND] -kt1d_visc3(ev_,nb_,qb0_,qbi_,vip_half,vim_half,tau,ALONG_X,eos_table)/DX;
      

   }


}


__kernel void kt_src_alongy(
              __global real* d_Src,
              __global real*  d_nb,
              __global real*  d_qb,
              __global real4* d_ev,
              const real tau,
              __global real* eos_table,
	      const int step){

    int I = get_global_id(0);
    int K = get_global_id(2);
    __local real4 ev[NY+4];
    __local real  nb[NY+4];
    __local real  qb0[NY+4];
    __local real  qbi[NY+4];


    for(int  J = get_global_id(1);J < NY; J = J+BSZ){
        int IND = I*NY*NZ+J*NZ+K ;
        ev[J+2] = d_ev[IND];
        nb[J+2] = d_nb[IND];
        qb0[J+2] = d_qb[idn2(IND,0)];
        qbi[J+2] = d_qb[idn2(IND,2)];


    }

    barrier(CLK_LOCAL_MEM_FENCE);

   if(get_local_id(1) == 0){
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NY+3] = ev[NY+1];
       ev[NY+2] = ev[NY+1];

       nb[0] = nb[2];
       nb[1] = nb[2];
       nb[NY+3] = nb[NY+1];
       nb[NY+2] = nb[NY+1];

       qb0[0] = qb0[2];
       qb0[1] = qb0[2];
       qb0[NY+3] = qb0[NY+1];
       qb0[NY+2] = qb0[NY+1];

       qbi[0] = qbi[2];
       qbi[1] = qbi[2];
       qbi[NY+3] = qbi[NY+1];
       qbi[NY+2] = qbi[NY+1];
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   for(int J = get_global_id(1);J < NY; J = J + BSZ){

      int IND = I*NY*NZ + J*NZ +K;
      int i = J+2;
      real4 ev_[5] = {ev[i-2],ev[i-1],ev[i],ev[i+1],ev[i+2]};
      real  nb_[5] = {nb[i-2],nb[i-1],nb[i],nb[i+1],nb[i+2]};
      real  qb0_[5] = {qb0[i-2],qb0[i-1],qb0[i],qb0[i+1],qb0[i+2]};
      real  qbi_[5] = {qbi[i-2],qbi[i-1],qbi[i],qbi[i+1],qbi[i+2]};


      real vip_half = 0.5f*(ev_[2].s2+ev_[3].s2);
      real vim_half = 0.5f*(ev_[1].s2+ev_[2].s2);
      d_Src[IND] = d_Src[IND] -kt1d_visc3(ev_,nb_,qb0_,qbi_,vip_half,vim_half,tau,ALONG_Y,eos_table)/DY; 


   }


}


__kernel void kt_src_alongz(
              __global real* d_Src,
              __global real*  d_nb,
              __global real*  d_qb,
              __global real4* d_ev,
              const real tau,
              __global real* eos_table,
	      const int step){

    int I = get_global_id(0);
    int J = get_global_id(1);
    __local real4 ev[NZ+4];
    __local real  nb[NZ+4];
    __local real  qb0[NZ+4];
    __local real  qbi[NZ+4];


    for(int  K = get_global_id(2);K < NZ; K = K+BSZ){
        int IND = I*NY*NZ+J*NZ+K ;
        ev[K+2] = d_ev[IND];
        nb[K+2] = d_nb[IND];
        qb0[K+2] = d_qb[idn2(IND,0)];
        qbi[K+2] = d_qb[idn2(IND,3)];


    }

    barrier(CLK_LOCAL_MEM_FENCE);

   if(get_local_id(2) == 0){
       ev[0] = ev[2];
       ev[1] = ev[2];
       ev[NZ+3] = ev[NZ+1];
       ev[NZ+2] = ev[NZ+1];

       nb[0] = nb[2];
       nb[1] = nb[2];
       nb[NZ+3] = nb[NZ+1];
       nb[NZ+2] = nb[NZ+1];

       qb0[0] = qb0[2];
       qb0[1] = qb0[2];
       qb0[NZ+3] = qb0[NZ+1];
       qb0[NZ+2] = qb0[NZ+1];

       qbi[0] = qbi[2];
       qbi[1] = qbi[2];
       qbi[NZ+3] = qbi[NZ+1];
       qbi[NZ+2] = qbi[NZ+1];
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   for(int K = get_global_id(2);K < NZ; K = K + BSZ){

      int IND = I*NY*NZ + J*NZ +K;
      int i = K+2;
      real4 ev_[5] = {ev[i-2],ev[i-1],ev[i],ev[i+1],ev[i+2]};
      real  nb_[5] = {nb[i-2],nb[i-1],nb[i],nb[i+1],nb[i+2]};
      real  qb0_[5] = {qb0[i-2],qb0[i-1],qb0[i],qb0[i+1],qb0[i+2]};
      real  qbi_[5] = {qbi[i-2],qbi[i-1],qbi[i],qbi[i+1],qbi[i+2]};


      real vip_half = 0.5f*(ev_[2].s3+ev_[3].s3);
      real vim_half = 0.5f*(ev_[1].s3+ev_[2].s3);
      d_Src[IND] = d_Src[IND] -kt1d_visc3(ev_,nb_,qb0_,qbi_,vip_half,vim_half,tau,ALONG_Z,eos_table)/(tau*DZ); 
   }


}



__kernel void update_evn(
	__global real4 * d_evnew,
	__global real4 * d_ev1,
	 __global real4 * d_ev2,
	__global real * d_nbnew,
	__global real * d_nb1,
        __global real * d_pi1,
        __global real * d_pi2,
        __global real * d_qb1,
        __global real * d_qb2,
        __global real * d_bulkpr1,
        __global real * d_bulkpr2,
	__global real4 * d_udiff,
	__global real4 * d_SrcT,
        __global real  * d_SrcJ,
        __global real  * eos_table,
	const real tau,
	const int  step,
	const real nbmax)
{
    int I = get_global_id(0);
    if ( I < NX*NY*NZ ) {
    real4 e_v = d_ev1[I];
    real nb = d_nb1[I];
    real ed = e_v.s0;
    real vx = e_v.s1;
    real vy = e_v.s2;
    real vz = e_v.s3;
    real pressure = eos_P(ed,nb, eos_table);
    real bulkpr = d_bulkpr1[I];
    real u0 = gamma(vx, vy, vz);
    real4 umu = u0*(real4)(1.0f, vx, vy, vz);
    real4 pim0 = (real4)(d_pi1[idn(I, 0)],
                         d_pi1[idn(I, 1)],
                         d_pi1[idn(I, 2)],
                         d_pi1[idn(I, 3)]);
   
    real4 u_new = umu4(d_ev2[I]);
    real4 u_old = umu;
    if ( step == 1 ) {
        //u_new = umu + d_udiff[I]; 
        //u_old = umu ;
        u_old = umu - d_udiff[I];
        u_new = umu;
    }


    real qb0 = d_qb1[idn2(I,0)];
    // when step=2, tau=(n+1)*DT, while T0m need tau=n*DT

    real old_time = tau - (step-1)*DT;
    real new_time = tau + (2-step)*DT;
    
    real4 deltapi = d_bulkpr2[I]*(gm[0]-u_new*u_new.s0)*new_time - d_bulkpr1[I]*(gm[0] - u_old*u_old.s0 )*old_time;

    
    real4 T0m = ((ed + pressure)*u0*umu - (pressure)*gm[0] + pim0)
                * old_time;
    real  J0 = (nb*u0+qb0)*old_time; 

    pim0 = (real4)(d_pi2[idn(I, 0)], d_pi2[idn(I, 1)],
                         d_pi2[idn(I, 2)], d_pi2[idn(I, 3)]);
    qb0 = d_qb2[idn2(I,0)];
    real qb1 = d_qb2[idn2(I,1)];
    real qb2 = d_qb2[idn2(I,2)];
    real qb3 = d_qb2[idn2(I,3)];

   
    real4 T0m1;
    T0m1 = T0m;
    real T001 = T0m1.s0;
    T0m = T0m + d_SrcT[I]*DT/step - pim0*new_time - deltapi;
    
    J0 = J0+ d_SrcJ[I]*DT/step - qb0*new_time;


    real ed_min=1e-7;
    //real T00 = max(acu, T0m.s0)/tau;
    //real T01 = (fabs(T0m.s1) < acu) ? 0.0f : T0m.s1/tau;
    //real T02 = (fabs(T0m.s2) < acu) ? 0.0f : T0m.s2/tau;
    //real T03 = (fabs(T0m.s3) < acu) ? 0.0f : T0m.s3/tau;

    real T00 = max(ed_min, T0m.s0)/new_time;
    real T01 = (fabs(T0m.s1) < acu) ? 0.0f : T0m.s1/new_time;
    real T02 = (fabs(T0m.s2) < acu) ? 0.0f : T0m.s2/new_time;
    real T03 = (fabs(T0m.s3) < acu) ? 0.0f : T0m.s3/new_time;

    real M = sqrt(T01*T01 + T02*T02 + T03*T03);
  
    real M1 = M;
    real SCALE_COEF = 0.999f;
    if ( M > T00 ) {
	    T01 *= SCALE_COEF * T00 / M;
	    T02 *= SCALE_COEF * T00 / M;
	    T03 *= SCALE_COEF * T00 / M;
            M = SCALE_COEF * T00;
    }
    
    //real Jid0 = (fabs(J0)<ed_min) ? 0.0f: J0/new_time;
    //real Jid0 = (fabs(J0)<ed_min) ? 0.0f: J0/tau;
    //real Jid0 = max(J0,acu)/tau;
    real Jid0 = J0/new_time;
    //real Jid0 = J0/tau;

    real ed_find = 0.0f;
    real nb_find = 0.0f;
    real vv = sqrt(vx*vx + vy*vy + vz*vz);    
    int state = 0; 
    rootFinding_newton(&ed_find, &nb_find,&state, T00, M, Jid0, eos_table);
    


   // set the fluid velocity to 0 if the energy density is too small
   if ( ed_find < acu ) {
       T01 = 0.0f;
       T02 = 0.0f;
       T03 = 0.0f;
   }

   //ed_find = max(0.0f, ed_find);
   //if ( ed_find < ed_min ) {
   //    T01 = 0.0f;
   //    T02 = 0.0f;
   //    T03 = 0.0f;
   //}


   ed_find = max(ed_min, ed_find);
   real pr = eos_P(ed_find,nb_find, eos_table);

   // vi = T0i/(T00+pr) = (e+p)u0*u0*vi/((e+p)*u0*u0)
   real epv = max(acu, T00 + pr );
   d_evnew[I] = (real4)(ed_find, T01/epv, T02/epv, T03/epv);
   d_nbnew[I] =nb_find;
   
   real vsq = d_evnew[I].s1*d_evnew[I].s1+d_evnew[I].s2*d_evnew[I].s2
           + d_evnew[I].s3*d_evnew[I].s3;
   
	   
	   
	   
   if ( vsq -1.0 > ed_min)
   {
     real scale = sqrt(0.999/vsq);
     d_evnew[I].s1 = d_evnew[I].s1*scale;
     d_evnew[I].s2 = d_evnew[I].s2*scale;
     d_evnew[I].s3 = d_evnew[I].s3*scale;

   }

   if ( state == 100 ) {
       d_evnew[I] = (real4)(ed, vx, vy, vz);
       d_nbnew[I] = nb;
     
   }



   

   
   real ed_find1 = max(ed_min, T00);

   if(T00 < 0.0f )
   {

        real ed_find1 = max(ed_min, T00);
        d_evnew[I] = (real4) (ed_find1, 0.0f, 0.0f, 0.0f);
	d_nbnew[I] = 0.0f;
   }
   real u0_tst = gamma(d_evnew[I].s1,d_evnew[I].s2, d_evnew[I].s3);

    








    }
}

__kernel void get_tpsmu(
	__global real4 * d_tpsmu,
	__global real  * eos_table,
	__global real4 * d_ev,
	__global real  * d_nb){


    int I = get_global_id(0);
    if (I < NX*NY*NZ)
    {
        real ed = d_ev[I].s0;
	real nb = d_nb[I];
	real pr = eos_P(ed, nb, eos_table);
	real T  = eos_T(ed, nb, eos_table);
	real mu = eos_mu(ed,nb, eos_table);
	real s  = eos_s(ed, nb, eos_table);
        
	d_tpsmu[I] = (real4) (T, pr, s, mu);
        

    }



}



__kernel void get_tmn(
        __global real4 * d_ev1,
	__global real  * d_nb1,
	__global real4 * d_Tmn,
	__global real *  d_J,
	__global real* eos_table){

    int I = get_global_id(0);
    if ( I < NX*NY*NZ ) {
    
    real4 e_v = d_ev1[I];
    real nb = d_nb1[I];
    real ed = e_v.s0;
    real vx = e_v.s1;
    real vy = e_v.s2;
    real vz = e_v.s3;
    real pressure = eos_P(ed,nb, eos_table);

    real u0 = gamma(vx, vy, vz);
    real4 umu = u0*(real4)(1.0f, vx, vy, vz);
    real4 T0m = (ed + pressure)*u0*umu - pressure*gm[0]; 
    real  J0 =  nb*u0; 
    real T00 = max(acu, T0m.s0);
    real T01 = (fabs(T0m.s1) < acu) ? 0.0f : T0m.s1;
    real T02 = (fabs(T0m.s2) < acu) ? 0.0f : T0m.s2;
    real T03 = (fabs(T0m.s3) < acu) ? 0.0f : T0m.s3;

    real M = sqrt(T01*T01 + T02*T02 + T03*T03);
    real SCALE_COEF = 0.999f;

    if ( M > T00 ) {
	    T01 *= SCALE_COEF * T00 / M;
	    T02 *= SCALE_COEF * T00 / M;
	    T03 *= SCALE_COEF * T00 / M;
            M = SCALE_COEF * T00;
    }

    real Jid0 = J0;
    
    d_Tmn[I] = (real4)(T00,T01,T02,T03);
    d_J[I] = J0;

    }

}

