#include<helper.h>

inline real van_Leer(real r){
    
    return sign(r)* max((real)0.0f,(min((real) 2.0*fabs(r),(real)1.0),min(fabs(r),(real)2.0)));
} 


inline void check_slope4(real4* slope){
    if(fabs((*slope).s0) > 1000.0)
    {  
         (*slope).s0=0.0;//van_Leer((*slope).s0);
        
    } 
    if(fabs((*slope).s1) > 1000.0)
    {
        (*slope).s1=0.0;//van_Leer((*slope).s1);
        
    } 
    if(fabs((*slope).s2) > 1000.0)
    {
        (*slope).s2=0.0;//van_Leer((*slope).s2);
        
    } 
    if(fabs((*slope).s3) > 1000.0)
    {
        (*slope).s3=0.0;//van_Leer((*slope).s3);
        
    } 

}

inline void check_slope(real* slope){
    if(fabs(*slope) > 1000.0)
    {  
         (*slope)=0.0;//van_Leer((*slope).s0);
        
    } 
    

}

inline real4 minmod14(real4 x, real4 y) {
    real4 res = min(fabs(x), fabs(y));
    return res*(sign(x)+sign(y))*0.5f;
}
// invariant vorticity vector 
// 2 * omega^{mu nu} = epsilon^{mu nu a b} d_a u_b
// 2 * omega^{tau x} = dyuz - dzuy
// 2 * omega^{tau y} = -(dxuz - dzux)
// 2 * omega^{tau z} = dxuy - dyux
// 2 * omega^{x y} = dtuz - dzut
// 2 * omega^{x z} = -(dtuy - dyut)
// 2 * omega^{y z} = dtux - dxut

// Covariant derivatives == normal derivatives for omega^{mu nu}


// calc beta*u_mu from (ed, vx, vy, tau^2*veta) float4 vector
// (u_t, u_x, u_y, u_eta) where u_eta = - gamma*v_eta

inline real4 ubeta(real4 ev, real nb , __global real * eos_table)
{
    real4 gmn = (real4)(1.0f, -1.0f, -1.0f, -1.0f);
    return gmn*umu4(ev)/eos_T(ev.s0,nb, eos_table);

}



inline real4 ukt4(real4 ev)
{
    real4 gmn = (real4)(1.0f, -1.0f, -1.0f, -1.0f);
    return gmn*umu4(ev);
}



// wrapper for address index
inline int address(int i, int j, int k)
{
    return i*NY*NZ + j*NZ + k;
}

__kernel void omega(
  __global real4 * d_ev1,
  __global real4 * d_ev2,
  __global real * d_nb1,
  __global real * d_nb2,

  __global real  * d_omega,
  __global real * eos_table,
     const real tau)
{
  int I = get_global_id(0);
  int J = get_global_id(1);
  int K = get_global_id(2);


  real4 uold = ubeta(d_ev1[address(I, J, K)], d_nb1[address(I, J, K)], eos_table);
  real4 unew = ubeta(d_ev2[address(I, J, K)], d_nb2[address(I, J, K)], eos_table);




  //   nabla_{t} u_{mu}
  real4 dudt = (unew - uold)/DT;
  real4 dudx = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
  real4 xx0= (real4)(0.0f, 0.0f, 0.0f, 0.0f);
  real4 xx1= (real4)(0.0f, 0.0f, 0.0f, 0.0f);

  
  if ( I != 0 && I != NX-1 ) {
      dudx = (ubeta(d_ev2[address(I+1, J, K)],d_nb2[address(I+1,J,K)], eos_table)
            - ubeta(d_ev2[address(I-1, J, K)],d_nb2[address(I-1,J,K)], eos_table)) / (2.0f*DX);
  } else if ( I == 0 ) { 
      dudx = (ubeta(d_ev2[address(I+1, J, K)],d_nb2[address(I+1, J, K)], eos_table) - unew) / DX;
  } else if ( I == NX-1 ) {
      dudx = (unew - ubeta(d_ev2[address(I-1, J, K)],d_nb2[address(I-1, J, K)], eos_table)) / DX;
  }

  real4 dudy = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
  if ( J != 0 && J != NY-1 ) {
      dudy = (ubeta(d_ev2[address(I, J+1, K)],d_nb2[address(I, J+1, K)], eos_table)
            - ubeta(d_ev2[address(I, J-1, K)],d_nb2[address(I, J-1, K)], eos_table)) / (2.0f*DY);
  } else if ( J == 0 ) { 
      dudy = (ubeta(d_ev2[address(I, J+1, K)],d_nb2[address(I, J+1, K)],  eos_table) - unew) / DY;
  } else if ( J == NY-1 ) {
      dudy = (unew - ubeta(d_ev2[address(I, J-1, K)],d_nb2[address(I, J-1, K)],  eos_table)) / DY;
  }

  // do not use Christoffel symbols, dudz = 1/tau * partial_eta u_{mu}
  // u_{eta} = - gamma*v_eta, has no dimension here

  // real4 dudz = (real4)(unew.s3, 0.0f, 0.0f, -unew.s0)/tau;
  // nabla_{tau} u_{eta} - (1/tau)nabla_{eta}u_{tau} = partial_{tau}u_{eta} - (1/tau)partial_{eta}u_{tau}

  real temp = eos_T(d_ev2[address(I, J, K)].s0, d_nb2[address(I, J, K)], eos_table);
  real4 Christoffel = (real4) (- unew.s3/(tau), 0.0f,0.0f, - unew.s0/(tau) );


  real4 dudz = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
  if ( K != 0 && K != NZ-1 ) {
      dudz += (ubeta(d_ev2[address(I, J, K+1)], d_nb2[address(I, J, K+1)],  eos_table)
            - ubeta(d_ev2[address(I, J, K-1)],  d_nb2[address(I, J, K-1)],  eos_table)) / (2.0f*DZ*tau)+Christoffel;
  } else if ( K == 0 ) { 
      dudz += (ubeta(d_ev2[address(I, J, K+1)], d_nb2[address(I, J, K+1)], eos_table) - unew) / (DZ*tau)+Christoffel;
  } else if ( K == NZ-1 ) {
      dudz += (unew - ubeta(d_ev2[address(I, J, K-1)], d_nb2[address(I, J, K-1)], eos_table)) / (DZ*tau)+Christoffel;
  }

  // hbarc convers 1/(GeV*fm) to dimensionless

  check_slope4(&dudt);
  check_slope4(&dudx);
  check_slope4(&dudy);
  check_slope4(&dudz);

  d_omega[6*address(I,J,K)+0] = 0.5f * hbarc*(dudy.s3 - dudz.s2); //tx
  d_omega[6*address(I,J,K)+1] = 0.5f * hbarc*(dudz.s1 - dudx.s3); //ty
  d_omega[6*address(I,J,K)+2] = 0.5f * hbarc*(dudx.s2 - dudy.s1); //tz
  d_omega[6*address(I,J,K)+3] = 0.5f * hbarc*(dudt.s3 - dudz.s0); //xy
  d_omega[6*address(I,J,K)+4] = 0.5f * hbarc*(dudy.s0 - dudt.s2); //xz
  d_omega[6*address(I,J,K)+5] = 0.5f * hbarc*(dudt.s1 - dudx.s0); //yz





}






__kernel void omega_shear(
   __global real4 * d_ev1,
   __global real4 * d_ev2,
   __global real * d_nb1,
   __global real * d_nb2,
   __global real  * d_omega1,
   __global real  * d_omega2,
   __global real * eos_table,
   const real tau)
{
   int I = get_global_id(0);
   int J = get_global_id(1);
   int K = get_global_id(2);

   real4 uold = ukt4(d_ev1[address(I, J, K)]);
   real4 unew = ukt4(d_ev2[address(I, J, K)]);


   real4 uold_mu = umu4(d_ev1[address(I, J, K)]);
   real4 unew_mu = umu4(d_ev2[address(I, J, K)]);


   //   nabla_{t} u_{mu}
   real4 dudt = (unew - uold)/DT;
   real4 dudx = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
   if ( I != 0 && I != NX-1 ) {
       dudx = (ukt4(d_ev2[address(I+1, J, K)])
             - ukt4(d_ev2[address(I-1, J, K)])) / (2.0f*DX);
   } else if ( I == 0 ) {
       dudx = (ukt4(d_ev2[address(I+1, J, K)]) - unew) / DX;
   } else if ( I == NX-1 ) {
       dudx = (unew - ukt4(d_ev2[address(I-1, J, K)])) / DX;
   }

   real4 dudy = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
   if ( J != 0 && J != NY-1 ) {
       dudy = (ukt4(d_ev2[address(I, J+1, K)])
             - ukt4(d_ev2[address(I, J-1, K)])) / (2.0f*DY);
   } else if ( J == 0 ) {
       dudy = (ukt4(d_ev2[address(I, J+1, K)]) - unew) / DY;
   } else if ( J == NY-1 ) {
       dudy = (unew - ukt4(d_ev2[address(I, J-1, K)])) / DY;
   }

   // do not use Christoffel symbols, dudz = 1/tau * partial_eta u_{mu}
   // u_{eta} = - gamma*v_eta, has no dimension here

   // real4 dudz = (real4)(unew.s3, 0.0f, 0.0f, -unew.s0)/tau;
   // nabla_{tau} u_{eta} - (1/tau)nabla_{eta}u_{tau} = partial_{tau}u_{eta} - (1/tau)partial_{eta}u_{tau}

   real4 dudz = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
   real4 Christoffel = (real4) (- unew.s3/tau, 0.0f,0.0f, - unew.s0/tau );
   if ( K != 0 && K != NZ-1 ) {
       dudz += (ukt4(d_ev2[address(I, J, K+1)])
             - ukt4(d_ev2[address(I, J, K-1)])) / (2.0f*DZ*tau) + Christoffel;
   } else if ( K == 0 ) {
       dudz += (ukt4(d_ev2[address(I, J, K+1)]) - unew) / (DZ*tau) + Christoffel;
   } else if ( K == NZ-1 ) {
       dudz += (unew - ukt4(d_ev2[address(I, J, K-1)])) / (DZ*tau) + Christoffel;
   }

   check_slope4(&dudt);
   check_slope4(&dudx);
   check_slope4(&dudy);
   check_slope4(&dudz);



   // hbarc convers 1/(GeV*fm) to dimensionless
   d_omega1[16*address(I,J,K)+0]  = 0.5*dudt.s0;
   d_omega1[16*address(I,J,K)+1]  = 0.5*dudt.s1;
   d_omega1[16*address(I,J,K)+2]  = 0.5*dudt.s2;
   d_omega1[16*address(I,J,K)+3]  = 0.5*dudt.s3;
   d_omega1[16*address(I,J,K)+4]  = 0.5*dudx.s0;
   d_omega1[16*address(I,J,K)+5]  = 0.5*dudx.s1;
   d_omega1[16*address(I,J,K)+6]  = 0.5*dudx.s2;
   d_omega1[16*address(I,J,K)+7]  = 0.5*dudx.s3;
   d_omega1[16*address(I,J,K)+8]  = 0.5*dudy.s0;
   d_omega1[16*address(I,J,K)+9]  = 0.5*dudy.s1;
   d_omega1[16*address(I,J,K)+10] = 0.5*dudy.s2;
   d_omega1[16*address(I,J,K)+11] = 0.5*dudy.s3;
   d_omega1[16*address(I,J,K)+12] = 0.5*dudz.s0;
   d_omega1[16*address(I,J,K)+13] = 0.5*dudz.s1;
   d_omega1[16*address(I,J,K)+14] = 0.5*dudz.s2;
   d_omega1[16*address(I,J,K)+15] = 0.5*dudz.s3;

   d_omega2[4*address(I,J,K)+0] = 0.5*(unew_mu.s0*dudt.s0 + unew_mu.s1*dudx.s0 + unew_mu.s2*dudy.s0 + unew_mu.s3*dudz.s0);
   d_omega2[4*address(I,J,K)+1] = 0.5*(unew_mu.s0*dudt.s1 + unew_mu.s1*dudx.s1 + unew_mu.s2*dudy.s1 + unew_mu.s3*dudz.s1);
   d_omega2[4*address(I,J,K)+2] = 0.5*(unew_mu.s0*dudt.s2 + unew_mu.s1*dudx.s2 + unew_mu.s2*dudy.s2 + unew_mu.s3*dudz.s2);
   d_omega2[4*address(I,J,K)+3] = 0.5*(unew_mu.s0*dudt.s3 + unew_mu.s1*dudx.s3 + unew_mu.s2*dudy.s3 + unew_mu.s3*dudz.s3);
}



__kernel void omega_accT(
   __global real4 * d_ev1,
   __global real4 * d_ev2,
   __global real * d_nb1,
   __global real * d_nb2,
   __global real  * d_omega,
   __global real * eos_table,
   const real tau)
{
   int I = get_global_id(0);
   int J = get_global_id(1);
   int K = get_global_id(2);

   real edold = d_ev1[address(I,J,K)].s0;
   real ednew = d_ev2[address(I,J,K)].s0;

   



   real nbold = d_nb1[address(I,J,K)];
   real nbnew = d_nb2[address(I,J,K)];

   real temp = eos_T(d_ev2[address(I, J, K)].s0, d_nb2[address(I, J, K)], eos_table);
   real dTdt = (eos_T(ednew,nbnew, eos_table) - eos_T(edold,nbold, eos_table))/DT;

   real dTdx = 0.0f;
   if ( I != 0 && I != NX-1 ) {
       dTdx = (eos_T(d_ev2[address(I+1, J, K)].s0, d_nb2[address(I+1, J, K)], eos_table)
             - eos_T(d_ev2[address(I-1, J, K)].s0, d_nb2[address(I-1, J, K)], eos_table)) / (2.0f*DX);
   } else if ( I == 0 ) {
       dTdx = (eos_T(d_ev2[address(I+1, J, K)].s0, d_nb2[address(I+1, J, K)], eos_table) - eos_T(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table) ) / DX;
   } else if ( I == NX-1 ) {
       dTdx = (eos_T(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table) - eos_T(d_ev2[address(I-1, J, K)].s0, d_nb2[address(I-1, J, K)], eos_table)) / DX;
   }

   real dTdy = 0.0f;
   if ( J != 0 && J != NY-1 ) {
       dTdy = (eos_T(d_ev2[address(I, J+1, K)].s0, d_nb2[address(I, J+1, K)],  eos_table)
             - eos_T(d_ev2[address(I, J-1, K)].s0, d_nb2[address(I, J-1, K)], eos_table)) / (2.0f*DY);
   } else if ( J == 0 ) {
       dTdy = (eos_T(d_ev2[address(I, J+1, K)].s0,d_nb2[address(I, J+1, K)],  eos_table) - eos_T(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table) ) / DY;
   } else if ( J == NY-1 ) {
       dTdy = (eos_T(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table) - eos_T(d_ev2[address(I, J-1, K)].s0,d_nb2[address(I, J-1, K)], eos_table)) / DY;
   }


   real dTdz = 0.0f;
   if ( K != 0 && K != NZ-1 ) {
       dTdz += (eos_T(d_ev2[address(I, J, K+1)].s0, d_nb2[address(I, J, K+1)],  eos_table)
             -  eos_T(d_ev2[address(I, J, K-1)].s0, d_nb2[address(I, J, K-1)],  eos_table)) / (2.0f*DZ*tau);
   } else if ( K == 0 ) {
       dTdz += (eos_T(d_ev2[address(I, J, K+1)].s0, d_nb2[address(I, J, K+1)], eos_table) 
	       - eos_T(d_ev2[address(I, J, K)].s0,   d_nb2[address(I, J, K)],eos_table) ) / (DZ*tau);
   } else if ( K == NZ-1 ) {
       dTdz += (eos_T(d_ev2[address(I, J, K)].s0,  d_nb2[address(I, J, K)],  eos_table) 
	       - eos_T(d_ev2[address(I, J, K-1)].s0,d_nb2[address(I, J, K-1)], eos_table)) / (DZ*tau);
   }
   




   real4 uold = ukt4(d_ev1[address(I, J, K)]);
   real4 unew = ukt4(d_ev2[address(I, J, K)]);

   real4 uold_mu = umu4(d_ev1[address(I, J, K)]);
   real4 unew_mu = umu4(d_ev2[address(I, J, K)]);

   //   nabla_{t} u_{mu}
   real4 dudt = (unew - uold)/DT;

   real4 dudx = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
   if ( I != 0 && I != NX-1 ) {
       dudx = (ukt4(d_ev2[address(I+1, J, K)])
             - ukt4(d_ev2[address(I-1, J, K)])) / (2.0f*DX);
   } else if ( I == 0 ) {
       dudx = (ukt4(d_ev2[address(I+1, J, K)]) - unew) / DX;
   } else if ( I == NX-1 ) {
       dudx = (unew - ukt4(d_ev2[address(I-1, J, K)])) / DX;
   }

   real4 dudy = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
   if ( J != 0 && J != NY-1 ) {
       dudy = (ukt4(d_ev2[address(I, J+1, K)])
             - ukt4(d_ev2[address(I, J-1, K)])) / (2.0f*DY);
   } else if ( J == 0 ) {
       dudy = (ukt4(d_ev2[address(I, J+1, K)]) - unew) / DY;
   } else if ( J == NY-1 ) {
       dudy = (unew - ukt4(d_ev2[address(I, J-1, K)])) / DY;
   }


   real4 dudz = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
   real4 Christoffel = (real4) (- unew.s3/tau, 0.0f,0.0f, - unew.s0/tau );
   if ( K != 0 && K != NZ-1 ) {
       dudz += (ukt4(d_ev2[address(I, J, K+1)])
             - ukt4(d_ev2[address(I, J, K-1)])) / (2.0f*DZ*tau)+ Christoffel;
   } else if ( K == 0 ) {
       dudz += (ukt4(d_ev2[address(I, J, K+1)]) - unew) / (DZ*tau)+ Christoffel;
   } else if ( K == NZ-1 ) {
       dudz += (unew - ukt4(d_ev2[address(I, J, K-1)])) / (DZ*tau)+ Christoffel;
   }
   ////
   check_slope(&dTdt);
   check_slope(&dTdx);
   check_slope(&dTdy);
   check_slope(&dTdz);

   check_slope4(&dudt);
   check_slope4(&dudx);
   check_slope4(&dudy);
   check_slope4(&dudz);



   real4 DU = (real4){0.0f , 0.0f, 0.0f, 0.0f};
   DU.s0 = unew_mu.s0*dudt.s0 + unew_mu.s1*dudx.s0 + unew_mu.s2*dudy.s0 + unew_mu.s3*dudz.s0;
   DU.s1 = unew_mu.s0*dudt.s1 + unew_mu.s1*dudx.s1 + unew_mu.s2*dudy.s1 + unew_mu.s3*dudz.s1;
   DU.s2 = unew_mu.s0*dudt.s2 + unew_mu.s1*dudx.s2 + unew_mu.s2*dudy.s2 + unew_mu.s3*dudz.s2;
   DU.s3 = unew_mu.s0*dudt.s3 + unew_mu.s1*dudx.s3 + unew_mu.s2*dudy.s3 + unew_mu.s3*dudz.s3;

   d_omega[6*address(I,J,K)+0] = 0.5*(unew.s2*(DU.s3 - dTdz/temp) - unew.s3*(DU.s2 -dTdy/temp)); //tx
   d_omega[6*address(I,J,K)+1] = 0.5*(-unew.s1*(DU.s3 - dTdz/temp) + unew.s3*(DU.s1 -dTdx/temp)); //ty
   d_omega[6*address(I,J,K)+2] = 0.5*( unew.s1*(DU.s2 - dTdy/temp) - unew.s2*(DU.s1 -dTdx/temp)); //tz
   d_omega[6*address(I,J,K)+3] = 0.5*(-unew.s3*(DU.s0 - dTdt/temp) + unew.s0*(DU.s3 -dTdz/temp)); //xy
   d_omega[6*address(I,J,K)+4] = 0.5*( unew.s2*(DU.s0 - dTdt/temp) - unew.s0*(DU.s2 -dTdy/temp)); //xz
   d_omega[6*address(I,J,K)+5] = 0.5*(-unew.s1*(DU.s0 - dTdt/temp) + unew.s0*(DU.s1 -dTdx/temp)); //yz
}




__kernel void omega_chemical(
   __global real4 * d_ev1,
   __global real4 * d_ev2,
   __global real * d_nb1,
   __global real * d_nb2,
   __global real  * d_omega,
   __global real * eos_table,
   const real tau)
{
   int I = get_global_id(0);
   int J = get_global_id(1);
   int K = get_global_id(2);


   real edold = d_ev1[address(I,J,K)].s0;
   real ednew = d_ev2[address(I,J,K)].s0;

   real nbold = d_nb1[address(I,J,K)];
   real nbnew = d_nb2[address(I,J,K)];

   real mub = eos_mu(d_ev2[address(I, J, K)].s0, d_nb2[address(I, J, K)], eos_table);
   real dmudt = (eos_mu(ednew,nbnew, eos_table)/eos_T(ednew,nbnew, eos_table) - eos_mu(edold,nbold, eos_table)/eos_T(edold,nbold, eos_table))/DT;

   real dmudx = 0.0f;
   if ( I != 0 && I != NX-1 ) {
       dmudx = (eos_mu(d_ev2[address(I+1, J, K)].s0, d_nb2[address(I+1, J, K)], eos_table)/eos_T(d_ev2[address(I+1, J, K)].s0, d_nb2[address(I+1, J, K)], eos_table)
             - eos_mu(d_ev2[address(I-1, J, K)].s0, d_nb2[address(I-1, J, K)], eos_table)/eos_T(d_ev2[address(I-1, J, K)].s0, d_nb2[address(I-1, J, K)], eos_table)) / (2.0f*DX);
   } else if ( I == 0 ) {
       dmudx = (eos_mu(d_ev2[address(I+1, J, K)].s0, d_nb2[address(I+1, J, K)], eos_table)/eos_T(d_ev2[address(I+1, J, K)].s0, d_nb2[address(I+1, J, K)], eos_table) 
	       - eos_mu(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table)/eos_T(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table)) / DX;
   } else if ( I == NX-1 ) {
       dmudx = (eos_mu(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table)/eos_T(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table)
	      - eos_mu(d_ev2[address(I-1, J, K)].s0, d_nb2[address(I-1, J, K)], eos_table)/eos_T(d_ev2[address(I-1, J, K)].s0, d_nb2[address(I-1, J, K)], eos_table)) / DX;
   }

   real dmudy = 0.0f;
   if ( J != 0 && J != NY-1 ) {
       dmudy = (eos_mu(d_ev2[address(I, J+1, K)].s0, d_nb2[address(I, J+1, K)],  eos_table)/eos_T(d_ev2[address(I, J+1, K)].s0, d_nb2[address(I, J+1, K)],  eos_table)
             - eos_mu(d_ev2[address(I, J-1, K)].s0, d_nb2[address(I, J-1, K)], eos_table)/eos_T(d_ev2[address(I, J-1, K)].s0, d_nb2[address(I, J-1, K)], eos_table) ) / (2.0f*DY);
   } else if ( J == 0 ) {
       dmudy = (eos_mu(d_ev2[address(I, J+1, K)].s0,d_nb2[address(I, J+1, K)],  eos_table)/eos_T(d_ev2[address(I, J+1, K)].s0,d_nb2[address(I, J+1, K)],  eos_table)
	      - eos_mu(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table)/eos_T(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table) ) / DY;
   } else if ( J == NY-1 ) {
       dmudy = (eos_mu(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table)/eos_T(d_ev2[address(I, J, K)].s0,d_nb2[address(I, J, K)], eos_table)
	      - eos_mu(d_ev2[address(I, J-1, K)].s0,d_nb2[address(I, J-1, K)], eos_table)/eos_T(d_ev2[address(I, J-1, K)].s0,d_nb2[address(I, J-1, K)], eos_table)) / DY;
   }


   real dmudz = 0.0f;
   if ( K != 0 && K != NZ-1 ) {
       dmudz = (eos_mu(d_ev2[address(I, J, K+1)].s0, d_nb2[address(I, J, K+1)],  eos_table)/eos_T(d_ev2[address(I, J, K+1)].s0, d_nb2[address(I, J, K+1)],  eos_table)
             -  eos_mu(d_ev2[address(I, J, K-1)].s0, d_nb2[address(I, J, K-1)],  eos_table)/eos_T(d_ev2[address(I, J, K-1)].s0, d_nb2[address(I, J, K-1)],  eos_table)) / (2.0f*DZ*tau);
   } else if ( K == 0 ) {
       dmudz = (eos_mu(d_ev2[address(I, J, K+1)].s0, d_nb2[address(I, J, K+1)], eos_table)/eos_T(d_ev2[address(I, J, K+1)].s0, d_nb2[address(I, J, K+1)], eos_table)  
	       - eos_mu(d_ev2[address(I, J, K)].s0,   d_nb2[address(I, J, K)],eos_table)/eos_T(d_ev2[address(I, J, K)].s0,   d_nb2[address(I, J, K)],eos_table)) / (DZ*tau);
   } else if ( K == NZ-1 ) {
       dmudz = (eos_mu(d_ev2[address(I, J, K)].s0,  d_nb2[address(I, J, K)],  eos_table)/eos_T(d_ev2[address(I, J, K)].s0,  d_nb2[address(I, J, K)],  eos_table)  
	        - eos_mu(d_ev2[address(I, J, K-1)].s0,d_nb2[address(I, J, K-1)], eos_table)/eos_T(d_ev2[address(I, J, K-1)].s0,d_nb2[address(I, J, K-1)], eos_table)) / (DZ*tau);
   }
   
   //check_slope(&dmudt);
   //check_slope(&dmudx);
   //check_slope(&dmudy);
   //check_slope(&dmudz);

   
   real4 uold = ukt4(d_ev1[address(I, J, K)]);
   real4 unew = ukt4(d_ev2[address(I, J, K)]);

   d_omega[6*address(I,J,K)+0] = -unew.s3*dmudy + unew.s2*dmudz; //tx
   d_omega[6*address(I,J,K)+1] =  unew.s3*dmudx - unew.s1*dmudz; //ty
   d_omega[6*address(I,J,K)+2] = -unew.s2*dmudx + unew.s1*dmudy; //tz
   d_omega[6*address(I,J,K)+3] = -unew.s3*dmudt + unew.s0*dmudz; //xy
   d_omega[6*address(I,J,K)+4] =  unew.s2*dmudt - unew.s0*dmudy; //xz
   d_omega[6*address(I,J,K)+5] = -unew.s1*dmudt + unew.s0*dmudx; //yz
}
