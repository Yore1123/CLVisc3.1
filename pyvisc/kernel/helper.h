#ifndef __HELPERH__
#define __HELPERH__

#include"real_type.h"
#include"eos_table.h"

#define THETA 1.8f
#define hbarc 0.19733f

// represent 16 components of pi^{mu nu} in 10 elements
#define idx(i,j) (((i)<(j))?((7*(i)+2*(j)-(i)*(i))/2):((7*(j)+2*(i)-(j)*(j))/2))
#define idx3(i,j) (i)*4+j
// idn() return the idx of pi^{mu nu}_{i,j,k}in global mem
// I = i*NY*NZ + j*NZ + k
#define idn(I, mn) (I)*10+mn
#define idn2(I,mn) (I)*4+mn

#define ALONG_X 0
#define ALONG_Y 1
#define ALONG_Z 2


inline real CB(real temperature) {
    return CB_XMIN; 
}


// parametrization of temperature dependent eta/s
inline real CETA(real temperature) {
    return (temperature < CETA_XMIN) ? (CETA_LEFT_SLOP * (temperature - CETA_XMIN) + CETA_YMIN)
                                      : (CETA_RIGHT_SLOP * (temperature - CETA_XMIN) + CETA_YMIN);
}


inline real CZETA(real temperature) {
    real Tc = 0.18;
    real x = temperature/Tc;

    real A1=-13.77, A2=27.55, A3=13.45;
    real lambda1=0.9, lambda2=0.25, lambda3=0.9, lambda4=0.22;
    real sigma1=0.025, sigma2=0.13, sigma3=0.0025, sigma4=0.022;

    real CZETA1 = A1*x*x + A2*x - A3;
    if (temperature < 0.995*Tc) {
        CZETA1 = (lambda3*exp((x-1)/sigma3) + lambda4*exp((x-1)/sigma4) + 0.03);
    }
    if (temperature > 1.05*Tc) {
        CZETA1 = (lambda1*exp(-(x-1)/sigma1) + lambda2*exp(-(x-1)/sigma2) + 0.001);
    }
#ifdef CZETA_DUKE
    return 1.1*CZETA1; 
#else
    return CZETA_CONST;
#endif

}


//inline real CZETA1(real temperature) {
//    real Tc = 0.155;
//    real x = temperature/Tc;
//    real A0= -13.45;
//    real A1= 27.55;
//    real A2=-13.77;
//    real lambda1=0.9;
//    real lambda2=0.25;
//    real lambda3=0.9;
//    real lambda4=0.22;
//    real sigma1=0.025;
//    real sigma2=0.13;
//    real sigma3=0.0025;
//    real sigma4=0.022;
//
//    real CZETA1 = A2*x*x + A1*x + A0;
//    if (temperature < 0.995*Tc)
//        { CZETA1 = (lambda3*exp((x-1)/sigma3) + lambda4*exp((x-1)/sigma4) + 0.03); }
//    if (temperature > 1.05*Tc)
//        { CZETA1 = (lambda1*exp(-(x-1)/sigma1) + lambda2*exp(-(x-1)/sigma2) + 0.001);}
//    return CZETA1; 
//    
//}


//real4 kt1d_nb(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1, real4 ev_ip2,real nb[5],
//           real tau, int along, __global real* eos_table);

// kt1d to calc H(i+1/2)-H(i-1/2), along=0,1,2 for x, y, z
real4 kt1d(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1, real4 ev_ip2,
           real tau, int along, __global real* eos_table);

// kt1d_real for each component of shear_pi and bulk_pi;
//real kt1d_real(
//       real Q_im2, real Q_im1, real Q_i, real Q_ip1, real Q_ip2,
//      real v_mh, real v_ph, real lam_mh, real lam_ph,
//       real tau, int along);

real kt1d_real2(real Q[5],real4 ev[5], real nb[5],real tau, int along, __global real* eos_table);

// g^{tau mu}, g^{x mu}, g^{y mu}, g^{eta mu} without tau*tau
constant real4 gm[4] = 
{(real4)(1.0f, 0.0f, 0.0f, 0.0f),
(real4)(0.0f, -1.0f, 0.0f, 0.0f),
(real4)(0.0f, 0.0f, -1.0f, 0.0f),
(real4)(0.0f, 0.0f, 0.0f, -1.0f)};
////////////////////////////////////////////////////////////
///*!Cacl gamma from vx, vy, vz where vz=veta in Milne space */
inline real gamma(real vx, real vy, real vz){
    return 1.0f/sqrt(max(1.0f-vx*vx-vy*vy-vz*vz, acu));
}
//
///*!Cacl gamma from (real4)(ed, vx, vy, vz) where vz=veta in Milne space */
inline real gamma_real4(real4 ev){
    return 1.0f/sqrt(max(1.0f-ev.s1*ev.s1-ev.s2*ev.s2-ev.s3*ev.s3, acu));
}
//
///** get (real4) umu4 from (real4) ev */
inline real4 umu4(real4 ev){
    return gamma_real4(ev)*(real4)(1.0f, ev.s123);
}
//
///** 1D linear interpolation */
inline real lin_int( real x1, real x2, real y1, real y2, real x )
{
    real r = ( x - x1 )/ (x2-x1 );
    return y1*(1.0f-r) + y2*r ;
}

inline real poly3_int( real x1, real x2, real x3, real y1, real y2, real y3, real x )
{
  return y1*(x-x2)*(x-x3)/((x1-x2)*(x1-x3)) + y2*(x-x1)*(x-x3)/((x2-x1)*(x2-x3)) + y3*(x-x1)*(x-x2)/((x3-x1)*(x3-x2)) ;
}


///** Flux limit for scalar and real4*/
inline real minmod(real x, real y) {
    real res = min(fabs(x), fabs(y));
    return res*(sign(x)+sign(y))*0.5f;
}


inline real4 minmod4(real4 x, real4 y) {
    real4 res = min(fabs(x), fabs(y));
    return res*(sign(x)+sign(y))*0.5f;
}

inline real minmod3(real x, real y,real z) {
    real a=0.0;
    real b=1.0;
    if (fabs(x)<1e-7)
    {
        return 0.0;
    }
    else
    {
        return x*max(a,min(b,min(y/x,z/x)));
    }
    
}


//// Calc du/dt, du/dx or du/dy or du/dz where du/dz = du/(tau deta)
inline real4 dudw(real4 ul, real4 ur, real dw) {
    return (ur - ul)/dw;
}


///** Calc maximum propagation speed along k direction
// * The maximum lam can not be bigger than 1.0f, in relativity
// * if fluid velocity is v=1, maximum cs2=1/3, the signal speed
// * in computing frame is cs2' = (cs2+v)/(1+cs2*v) = 1.0 */
inline real maxPropagationSpeed(real4 edv, real nb, real vk, __global real* eos_table){
    real ut = gamma(edv.s1, edv.s2, edv.s3);
    real uk = fabs(ut*vk);
    real ut2 = ut*ut;
    real uk2 = uk*uk;
    //real cs2 = pr/max(edv.s0, acu);
    //real cs2 = P(edv.s0, eos_table)/max(edv.s0, acu);
    real cs2 = eos_CS2(edv.s0,nb, eos_table);
    real lam = ((ut*uk*(1.0f-cs2))+sqrt((ut2-uk2-(ut2-uk2-1.0f)*cs2)*cs2))
       /(ut2 - (ut2-1.0f)*cs2+1e-15);
     

    return min(max(lam,fabs(vk)),1.0f);
    //return cs2;
    //return min(lam,1.0f);
}



/** solve energy density from T00 and K=sqrt(T01**2 + T02**2 + T03**2)
 * 0.001% fail if absolute error < 0.01
 * What happens to these testing events?
 * */
// inline void rootFinding(real* EdFind, real T00, real M, read_only image2d_t eos_table){
//     real E0, E1;
//     E1 = T00;   /*predict */
//     real K2 = M*M;
//     int i = 0;
//     while ( true ) {
//         E0 = E1;
//         E1 = T00 - K2/(T00 + eos_P(E1, eos_table)) ; 
//         i ++ ;
//         if( i>100 || fabs(E1-E0)/max(fabs(E1), (real)acu)<acu ) break;
//     }

//     * EdFind = E1;
// }

inline void rootFinding_newton(real* ed_find,real* nb_find,int* state, real T00, real M,
                                  real J0, __global real* eos_table){
             
    real vl = 0.0f;
    real vh = 1.0f;
    real v = 0.5f*(vl+vh);
    real ed = T00 - M*v;
    real nb = (J0 )*sqrt(1-v*v);
    real pr = eos_P(ed, nb, eos_table);
    //real dpe = eos_CS2(ed,nb,eos_table);
    real dpe = 1.0/3.0;
    real f = (T00 + pr)*v - M;
    real df = (T00+pr) - M*v*dpe;
    real dvold = vh-vl;
    real dv = dvold;
    int i = 0;
    while ( true ) {
        if ((f + df * (vh - v)) * (f + df * (vl - v)) > 0.0f ||
            fabs(2.f * f) > fabs(dvold * df)) {  // bisection
          dvold = dv;
          dv = 0.5f * (vh - vl);
          v = vl + dv;
        } else {  // Newton
          dvold = dv;
          dv = f / df;
          v -= dv;
        }
        v = max(0.0f,min(v,1.0f));
        i ++;
        if ( fabs(dv) < 0.00001f || i > 100 ) break;

        ed = T00 - M*v;
        nb = (J0)*sqrt(1-v*v);
        pr = eos_P(ed, nb, eos_table);
        f = (T00 + pr)*v - M;
        //dpe = eos_CS2(ed,nb,eos_table);
        dpe = 1.0/3.0;
        df = (T00+pr) - M*v*dpe;
        if ( f > 0.0f ) {
             vh = v;
        } else { 
             vl = v;
        }
    }
    // accelerate the speed of iteration by combining with bisection(low slope) and newton method(high slope)
    * ed_find = T00 - M*v;
    * nb_find = (J0 )*sqrt(1-v*v);
    * state = i;
}

//inline void rootFinding_newton2(real* ed_find,real* nb_find, real T00, real M,
//                                  real J0,  __global real* eos_table){
//             
//    real vl = 0.0f;
//    real vh = 1.0f;
//    //real dpe = 1.0f/3.0f;
//    real v = 0.5f*(vl+vh);
//    real ed = T00 - M*v;
//    real nb = (J0 )*sqrt(1-v*v);
//    real pr = eos_P(ed, nb, eos_table);
//    //real dpe = CS2_nb(ed,nb,eos_table);
//    real dpde = eos_dpde(ed,nb,eos_table);
//    real dpdn = eos_dpdn(ed,nb,eos_table);
//    real dpdv = -(M*dpde + dpdn*J0*v/sqrt(1-v*v));
//    real f = (T00 + pr)*v - M;
//    real df = (T00+pr) + dpdv*v;
//    real dvold = vh-vl;
//    real dv = dvold;
//    int i = 0;
//    while ( true ) {
//        if ((f + df * (vh - v)) * (f + df * (vl - v)) > 0.0f ||
//            fabs(2.f * f) > fabs(dvold * df)) {  // bisection
//          dvold = dv;
//          dv = 0.5f * (vh - vl);
//          v = vl + dv;
//        } else {  // Newton
//          dvold = dv;
//          dv = f / df;
//          v -= dv;
//        }
//        i ++;
//        if ( fabs(dv) < 0.00001f || i > 100 ) break;
//
//        ed = T00 - M*v;
//        nb = (J0)*sqrt(1-v*v);
//        pr = eos_P(ed, nb, eos_table);
//        f = (T00 + pr)*v - M;
//        
//	dpde = eos_dpde(ed,nb,eos_table);
//        dpdn = eos_dpdn(ed,nb,eos_table);
//        dpdv = -(M*dpde + dpdn*J0*v/sqrt(1-v*v));
//        df = (T00+pr) + dpdv*v;
//        if ( f > 0.0f ) {
//             vh = v;
//        } else { 
//             vl = v;
//        }
//    }
//    // accelerate the speed of iteration by combining with bisection(low slope) and newton method(high slope)
//    * ed_find = T00 - M*v;
//    * nb_find = (J0 )*sqrt(1-v*v);
//}



/** construct T^{tau mu} 4 vector*/
inline real4 t0m(real4 edv, real pr) {
    real u0 = gamma(edv.s1, edv.s2, edv.s3);
    real ed = edv.s0;
    real4 u4 = (real4)(1.0f, edv.s1, edv.s2, edv.s3)*u0;
    return (ed+pr)*u0*u4 - gm[0]*pr;
}

//real4 kt1d(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1,
//           real4 ev_ip2, real tau, int along,
//           read_only image2d_t eos_table) {
//   real pr_im1 = eos_P(ev_im1.s0, eos_table);
//   real pr_i = P(ev_i.s0, eos_table);
//   real pr_ip1 = P(ev_ip1.s0, eos_table);
//   real pr_ip2 = P(ev_ip2.s0, eos_table);
//
//   real4 T0m_im1 = t0m(ev_im1, pr_im1);
//   real4 T0m_i = t0m(ev_i, pr_i);
//   real4 T0m_ip1 = t0m(ev_ip1, pr_ip1);
//   real4 T0m_ip2 = t0m(ev_ip2, pr_ip2);
//
//   real4 DA0, DA1;
//   DA0 = minmod4(0.5f*(T0m_ip1-T0m_im1),
//           minmod4(THETA*(T0m_ip1-T0m_i), THETA*(T0m_i-T0m_im1)));
//
//   DA1 = minmod4(0.5f*(T0m_ip2-T0m_i),
//         minmod4(THETA*(T0m_ip2-T0m_ip1), THETA*(T0m_ip1-T0m_i)));
//  
//   real vim1[3] = {ev_im1.s1, ev_im1.s2, ev_im1.s3};
//   real vi[3] = {ev_i.s1, ev_i.s2, ev_i.s3};
//   real vip1[3] = {ev_ip1.s1, ev_ip1.s2, ev_ip1.s3};
//
//   real4  AL = T0m_i   + 0.5f * DA0;
//   real4  AR = T0m_ip1 - 0.5f * DA1;
//
//   real pr_half = 0.5f*(pr_ip1 + pr_i);
//   real vi_half = 0.5f*(vi[along] + vip1[along]);
//   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
//   real4 Jp = (AR + pr_half*gm[0])*vi_half - pr_half*gm[along+1];
//   real4 Jm = (AL + pr_half*gm[0])*vi_half - pr_half*gm[along+1];
//
//   real4 ev_half = 0.5f*(ev_i+ev_ip1);
//   // maximum local propagation speed at i+1/2
//   real lam = maxPropagationSpeed(ev_half, vi_half, eos_table);
//
//   // first part of kt1d; the final results = src[i]-src[i-1]
//   real4 src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);
//
//   real pr_im2 = P(ev_im2.s0, eos_table);
//   real4 T0m_im2 = t0m(ev_im2, pr_im2);
//   DA1 = DA0;  // reuse the previous calculate value
//   DA0 = minmod4(0.5f*(T0m_i-T0m_im2),
//           minmod4(THETA*(T0m_i-T0m_im1), THETA*(T0m_im1-T0m_im2)));
//
//   AL = T0m_im1 + 0.5f * DA0;
//   AR = T0m_i - 0.5f * DA1;
//
//   // pr_half = pr(i+1/2)
//   pr_half = 0.5f*(pr_im1 + pr_i);
//   vi_half = 0.5f*(vim1[along] + vi[along]);
//   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
//   Jp = (AR + pr_half*gm[0])*vi_half - pr_half*gm[along+1];
//   Jm = (AL + pr_half*gm[0])*vi_half - pr_half*gm[along+1];
//
//   // maximum local propagation speed at i-1/2
//   ev_half = 0.5f*(ev_i+ev_im1);
//   lam = maxPropagationSpeed(ev_half, vi_half, eos_table);
//   // second part of kt1d; final results = src[i] - src[i-1]
//   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);
//
//   return src;
//}



//real4 kt1d_nb(real4 ev_im2, real4 ev_im1, real4 ev_i, real4 ev_ip1,
//           real4 ev_ip2, real nb[5], real tau, int along,
//           __global real* eos_table) {
//
//   real pr_im1 = eos_P(ev_im1.s0,nb[1], eos_table);
//   real pr_i = P_nb(ev_i.s0,nb[2], eos_table);
//   real pr_ip1 = P_nb(ev_ip1.s0,nb[3], eos_table);
//   real pr_ip2 = P_nb(ev_ip2.s0,nb[4], eos_table);
//
//   real4 T0m_im1 = t0m(ev_im1, pr_im1);
//   real4 T0m_i = t0m(ev_i, pr_i);
//   real4 T0m_ip1 = t0m(ev_ip1, pr_ip1);
//   real4 T0m_ip2 = t0m(ev_ip2, pr_ip2);
//
//
//   real4 DA0, DA1;
//   DA0 = minmod4(0.5f*(T0m_ip1-T0m_im1),
//           minmod4(THETA*(T0m_ip1-T0m_i), THETA*(T0m_i-T0m_im1)));
//
//   DA1 = minmod4(0.5f*(T0m_ip2-T0m_i),
//         minmod4(THETA*(T0m_ip2-T0m_ip1), THETA*(T0m_ip1-T0m_i)));
//  
//   real vim1[3] = {ev_im1.s1, ev_im1.s2, ev_im1.s3};
//   real vi[3] = {ev_i.s1, ev_i.s2, ev_i.s3};
//   real vip1[3] = {ev_ip1.s1, ev_ip1.s2, ev_ip1.s3};
//
//   real4  AL = T0m_i   + 0.5f * DA0;
//   real4  AR = T0m_ip1 - 0.5f * DA1;
//
//   real pr_half = 0.5f*(pr_ip1 + pr_i);
//   real vi_half = 0.5f*(vi[along] + vip1[along]);
//   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
//   real4 Jp = (AR + pr_half*gm[0])*vi_half - pr_half*gm[along+1];
//   real4 Jm = (AL + pr_half*gm[0])*vi_half - pr_half*gm[along+1];
//
//   real4 ev_half = 0.5f*(ev_i+ev_ip1);
//   real  nb_half = 0.5f*(nb[2]+nb[3]);
//   // maximum local propagation speed at i+1/2
//   real lam = maxPropagationSpeed_nb(ev_half,nb_half,vi_half, eos_table);
//
//
//   // first part of kt1d; the final results = src[i]-src[i-1]
//   real4 src = 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);
//
//   real pr_im2 = P_nb(ev_im2.s0,nb[0], eos_table);
//   real4 T0m_im2 = t0m(ev_im2, pr_im2);
//   DA1 = DA0;  // reuse the previous calculate value
//   DA0 = minmod4(0.5f*(T0m_i-T0m_im2),
//           minmod4(THETA*(T0m_i-T0m_im1), THETA*(T0m_im1-T0m_im2)));
//
//   AL = T0m_im1 + 0.5f * DA0;
//   AR = T0m_i - 0.5f * DA1;
//
//   // pr_half = pr(i+1/2)
//   pr_half = 0.5f*(pr_im1 + pr_i);
//   vi_half = 0.5f*(vim1[along] + vi[along]);
//   // Flux Jp = (T0m + pr*g^{tau mu})*v^x - pr*g^{x mu}
//   Jp = (AR + pr_half*gm[0])*vi_half - pr_half*gm[along+1];
//   Jm = (AL + pr_half*gm[0])*vi_half - pr_half*gm[along+1];
//
//   // maximum local propagation speed at i-1/2
//   ev_half = 0.5f*(ev_i+ev_im1);
//   nb_half = 0.5f*(nb[1]+nb[2]);
//   lam = maxPropagationSpeed_nb(ev_half,nb_half, vi_half, eos_table);
//   // second part of kt1d; final results = src[i] - src[i-1]
//   src -= 0.5f*(Jp+Jm) - 0.5f*lam*(AR-AL);
//
//   return src;
//}




// use pimn and ev at  i-2, i-1, i, i+1, i+2 to calc src term from flux
// pr_mh = pr_{i-1/2} and pr_ph = pr_{i+1/2}
// these mh, ph terms are calcualted 1 time and used 10 times by pimn

real kt1d_real(
       real Q_im2, real Q_im1, real Q_i, real Q_ip1, real Q_ip2,
       real v_mh, real v_ph, real lam_mh, real lam_ph,
       real tau, int along)
{
   real DA0, DA1;
   DA0 = minmod(0.5f*(Q_ip1-Q_im1),
           minmod(THETA*(Q_ip1-Q_i), THETA*(Q_i-Q_im1)));

   DA1 = minmod(0.5f*(Q_ip2-Q_i),
         minmod(THETA*(Q_ip2-Q_ip1), THETA*(Q_ip1-Q_i)));

   real  AL = Q_i   + 0.5f * DA0;
   real  AR = Q_ip1 - 0.5f * DA1;

   // Flux Jp = (Q + pr*g^{tau mu})*v^x - pr*g^{x mu}
   real Jp = AR * v_ph;
   real Jm = AL * v_ph;

   // first part of kt1d; the final results = src[i]-src[i-1]
   real src = 0.5f*(Jp+Jm) - 0.5f*lam_ph*(AR-AL);

   DA1 = DA0;  // reuse the previous calculate value
   DA0 = minmod(0.5f*(Q_i-Q_im2),
           minmod(THETA*(Q_i-Q_im1), THETA*(Q_im1-Q_im2)));

   AL = Q_im1 + 0.5f * DA0;
   AR = Q_i - 0.5f * DA1;

   Jp = AR*v_mh;
   Jm = AL*v_mh;

   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam_mh*(AR-AL);

   return src;
}

real kt1d_real2(real Q[5],real4 ev[5], real nb[5],real tau, int along,__global real* eos_table)
{
   real DA0, DA1;
   real4 DA0_ev,DA1_ev;
   real DA0_nb,DA1_nb;

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

   real4 evL = ev[2] + 0.5f*DA0_ev;
   real4 evR = ev[3] - 0.5f*DA1_ev;

   real vL[3] = {evL.s1, evL.s2, evL.s3};
   real vR[3] = {evR.s1, evR.s2, evR.s3};

   real nbL = nb[2] + 0.5f*DA0_nb;
   real nbR = nb[3] - 0.5f*DA1_nb;


   real  AL = Q[2] + 0.5f * DA0;
   real  AR = Q[3] - 0.5f * DA1;

   // Flux Jp = (Q + pr*g^{tau mu})*v^x - pr*g^{x mu}
   real Jp = AR * vR[along];
   real Jm = AL * vL[along];

   real lamL = maxPropagationSpeed(evL,nbL, vL[along], eos_table);
   real lamR = maxPropagationSpeed(evR,nbR, vR[along], eos_table);

   real lam_ph = max(lamL,lamR);

   // first part of kt1d; the final results = src[i]-src[i-1]
   real src = 0.5f*(Jp+Jm) - 0.5f*lam_ph*(AR-AL);

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

   AL = Q[1] + 0.5f * DA0;
   AR = Q[2] - 0.5f * DA1;

   Jp = AR*vR[along];
   Jm = AL*vL[along];

   lamL = maxPropagationSpeed(evL,nbL, vL[along], eos_table);
   lamR = maxPropagationSpeed(evR,nbR, vR[along], eos_table);
   real lam_mh = max(lamL,lamR);

   // second part of kt1d; final results = src[i] - src[i-1]
   src -= 0.5f*(Jp+Jm) - 0.5f*lam_mh*(AR-AL);

   return src;
}









#endif
