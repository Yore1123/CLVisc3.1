#include<helper.h>

#define one_o_2sigr2  1.0/(2.0*SIGR*SIGR)
#define one_o_2sigz2  1.0/(2.0*SIGZ*SIGZ)


#define w1  one_o_2sigr2/M_PI_F
#define w2  sqrt(one_o_2sigz2/M_PI_F)/TAU0

#define NPARTONS_PER_GROUP 512




inline void AtomicAdd(volatile __global real *source, const real operand) {  
    union {  
        unsigned int intVal;  
        float floatVal;  
    } newVal;  
    union {  
        unsigned int intVal;  
        float floatVal;  
    } prevVal;  
    do {  
        prevVal.floatVal = *source;  
        newVal.floatVal = prevVal.floatVal + operand;  
    } while (atomic_cmpxchg((volatile __global unsigned int *)source,   
                             prevVal.intVal, newVal.intVal)   
                             != prevVal.intVal);  
} 



__kernel void smearing_baryon_norm(
    __global real * d_p4x4, \
    const int nparton, \
    const int size,
    volatile __global real * norm,
    volatile __global real * norm2)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  if ( i < NX && j < NY && k < NZ ) {

  real4 grid = (real4)(TAU0, (i-NX/2)*DX, (j-NY/2)*DY, (k-NZ/2)*DZ);
  int IND = get_global_id(0)*NY*NZ + get_global_id(1)*NZ + get_global_id(2);

  __local real8 local_p4x4[NPARTONS_PER_GROUP];
  __local real local_mass[NPARTONS_PER_GROUP];
  __local real local_baryon[NPARTONS_PER_GROUP];

  real delta, etasi; 
  real nb_delta;
  real4 position;
  real4 momentum;
  real mass;
  real baryon;


  int li = get_local_id(0);
  int lj = get_local_id(1);
  int lk = get_local_id(2);

  // local id in the workgroup
  int tid = li*5*5 + lj*5 + lk;


  int npartons_in_this_group;
  const int num_of_groups = nparton / NPARTONS_PER_GROUP;


  const int block_size = get_local_size(0) * get_local_size(1) * \
                         get_local_size(2);

  for ( int n = 0; n < nparton; n = n + NPARTONS_PER_GROUP ) {


    // The last workgroup has different number of partons
    if ( n < num_of_groups * NPARTONS_PER_GROUP ) {
        npartons_in_this_group = NPARTONS_PER_GROUP;
    } else {
        npartons_in_this_group = nparton - num_of_groups * NPARTONS_PER_GROUP;
    }


    
    if(tid == 0 ){

        for ( int m = 0; m < npartons_in_this_group; m++  ) {
        if ( (n+m) < nparton ) {
#ifdef SMASH
          local_p4x4[m].s0 = d_p4x4[(n+m)*10+0];
          local_p4x4[m].s1 = d_p4x4[(n+m)*10+1];
          local_p4x4[m].s2 = d_p4x4[(n+m)*10+2];
          local_p4x4[m].s3 = d_p4x4[(n+m)*10+3];
          
          local_p4x4[m].s4 = d_p4x4[(n+m)*10+5];
          local_p4x4[m].s5 = d_p4x4[(n+m)*10+6];
          local_p4x4[m].s6 = d_p4x4[(n+m)*10+7];
          local_p4x4[m].s7 = d_p4x4[(n+m)*10+8];

          local_mass[m] = d_p4x4[(n+m)*10+4];
          local_baryon[m] = d_p4x4[(n+m)*10+9];
#endif

#ifdef AMPT
          local_p4x4[m].s0 = d_p4x4[(n+m)*9+4];
          local_p4x4[m].s1 = d_p4x4[(n+m)*9+5];
          local_p4x4[m].s2 = d_p4x4[(n+m)*9+6];
          local_p4x4[m].s3 = d_p4x4[(n+m)*9+7];
          
          local_p4x4[m].s4 = d_p4x4[(n+m)*9+0];
          local_p4x4[m].s5 = d_p4x4[(n+m)*9+1];
          local_p4x4[m].s6 = d_p4x4[(n+m)*9+2];
          local_p4x4[m].s7 = d_p4x4[(n+m)*9+3];

          local_mass[m] = 0.0f;
          local_baryon[m] = d_p4x4[(n+m)*9+8];
#endif

        }
        }
    }





    barrier(CLK_LOCAL_MEM_FENCE);

    for (int m=0; m < npartons_in_this_group; m++) {
      position = local_p4x4[m].s0123;
      momentum = local_p4x4[m].s4567;
      mass = local_mass[m];
      baryon = local_baryon[m];

      etasi = 0.5f*log(max(position.s0+position.s3, acu)) \
       - 0.5f*log(max(position.s0-position.s3, acu));
      
      position.s3 = etasi;
      real4 d = grid - position;

      
      
      real Yi = 0.5f * (log(max(momentum.s0+momentum.s3, acu)) \
                      - log(max(momentum.s0-momentum.s3, acu)));
      
     
      real gammaz = cosh(Yi - etasi);
      
      real nb_one_o_2sigr2 = 1.0/(2.0*NSIGR*NSIGR); 
      real nb_one_o_2sigz2 = 1.0/(2.0*NSIGZ*NSIGZ);
      
      
      //real distance_sqr = one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2) + one_o_2sigz2*(d.s3*d.s3*gammaz*gammaz*TAU0*TAU0);
      //real nb_distance_sqr = nb_one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2) + nb_one_o_2sigz2*(d.s3*d.s3*gammaz*gammaz*TAU0*TAU0);



      real distance_sqr = one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2) + one_o_2sigz2*(d.s3*d.s3);
      real nb_distance_sqr = nb_one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2) + nb_one_o_2sigz2*(d.s3*d.s3);
      

      // only do smearing for distance < 10*sigma
      //if ( distance_sqr < 50 ) {
      real maxedgex = max(10*SIGR,DX);
      real maxedgey = max(10*SIGR,DY);
      real maxedgez = max(10*SIGZ,DZ);

      if ( fabs(d.s3) < 10*SIGZ && fabs(d.s1) < 10*SIGR && fabs(d.s2)< 10*SIGR ) 
     {
          delta = exp(- distance_sqr);
          nb_delta = exp(- nb_distance_sqr);
#ifdef SMASH
          real mt = sqrt(momentum.s1 * momentum.s1 + momentum.s2 * momentum.s2+mass*mass);
#endif

#ifdef AMPT
	  real mt = sqrt(max(momentum.s0 * momentum.s0 - momentum.s3 * momentum.s3,
                 momentum.s1 * momentum.s1 + momentum.s2 * momentum.s2));
#endif
	  real Y = 0.5f * (log(max(momentum.s0+momentum.s3, acu)) \
                       - log(max(momentum.s0-momentum.s3, acu)));

          // grid.s3 = space-time rapidity of the hydrodynamic cell
          real4 momentum_miln = (real4)(1.0f, 1.0f,1.0f, 1.0f);
          real delta1 = delta*TAU0*DX*DY*DZ;
          real delta2 = nb_delta*TAU0*DX*DY*DZ;
          
	  AtomicAdd(&norm[n+m],delta1);
          AtomicAdd(&norm2[n+m],delta2);
          int index = n+m; 

      }

    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }


  }
}



__kernel void smearing_baryon(
    __global real4  * d_EdV, \
    __global real * d_nb, \
    __global real * d_p4x4, \
    __global real *  eos_table,\
    const int nparton, \
    const int size,\
    __global real * d_norm, 
    __global real * d_norm2 )
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  if ( i < NX && j < NY && k < NZ ) {

  real4 grid = (real4)(TAU0, (i-NX/2)*DX, (j-NY/2)*DY, (k-NZ/2)*DZ);

  __local real8 local_p4x4[NPARTONS_PER_GROUP];
  __local real local_mass[NPARTONS_PER_GROUP];
  __local real local_baryon[NPARTONS_PER_GROUP];
  __local real local_norm[NPARTONS_PER_GROUP];
  __local real local_norm2[NPARTONS_PER_GROUP];

   int IND = get_global_id(0)*NY*NZ + get_global_id(1)*NZ + get_global_id(2);

  real delta, etasi; 
  real nb_delta;
  real4 position;
  real4 momentum;
  real mass;
  real baryon;
  real norm=0.0f;
  real norm2 = 0.0f;
  real4 Tm0 = (real4)(0.0f, 0.0f, 0.0f, 0.0f);
  real  J0 = 0.0f;
   


  int li = get_local_id(0);
  int lj = get_local_id(1);
  int lk = get_local_id(2);

  // local id in the workgroup
  int tid = li*5*5 + lj*5 + lk;


  int npartons_in_this_group;
  const int num_of_groups = nparton / NPARTONS_PER_GROUP;

  const int block_size = get_local_size(0) * get_local_size(1) * \
                         get_local_size(2);

  for ( int n = 0; n < nparton; n = n + NPARTONS_PER_GROUP ) {




    // The last workgroup has different number of partons
    if ( n < num_of_groups * NPARTONS_PER_GROUP ) {
        npartons_in_this_group = NPARTONS_PER_GROUP;
    } else {
        npartons_in_this_group = nparton - num_of_groups * NPARTONS_PER_GROUP;
    }



    if(tid == 0 ){

        for ( int m = 0; m < npartons_in_this_group; m += 1 ) {
        if ( (n+m) < nparton ) {
#ifdef SMASH
          local_p4x4[m].s0 = d_p4x4[(n+m)*10+0];
          local_p4x4[m].s1 = d_p4x4[(n+m)*10+1];
          local_p4x4[m].s2 = d_p4x4[(n+m)*10+2];
          local_p4x4[m].s3 = d_p4x4[(n+m)*10+3];
          
          local_p4x4[m].s4 = d_p4x4[(n+m)*10+5];
          local_p4x4[m].s5 = d_p4x4[(n+m)*10+6];
          local_p4x4[m].s6 = d_p4x4[(n+m)*10+7];
          local_p4x4[m].s7 = d_p4x4[(n+m)*10+8];

          local_mass[m] = d_p4x4[(n+m)*10+4];
          local_baryon[m] = d_p4x4[(n+m)*10+9];
#endif

#ifdef AMPT
          local_p4x4[m].s0 = d_p4x4[(n+m)*9+4];
          local_p4x4[m].s1 = d_p4x4[(n+m)*9+5];
          local_p4x4[m].s2 = d_p4x4[(n+m)*9+6];
          local_p4x4[m].s3 = d_p4x4[(n+m)*9+7];
          
          local_p4x4[m].s4 = d_p4x4[(n+m)*9+0];
          local_p4x4[m].s5 = d_p4x4[(n+m)*9+1];
          local_p4x4[m].s6 = d_p4x4[(n+m)*9+2];
          local_p4x4[m].s7 = d_p4x4[(n+m)*9+3];

          local_mass[m] = 0.0f;
          local_baryon[m] = d_p4x4[(n+m)*9+8];
#endif

          local_norm[m] = d_norm[(n+m)];
          local_norm2[m] = d_norm2[(n+m)];
          
        }
        }
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    for (int m=0; m < npartons_in_this_group; m++) {
      position = local_p4x4[m].s0123;
      momentum = local_p4x4[m].s4567;
      mass = local_mass[m];
      baryon = local_baryon[m];
      norm = local_norm[m];
      norm2 = local_norm2[m];

      etasi = 0.5f*log(max(position.s0+position.s3, acu)) \
       - 0.5f*log(max(position.s0-position.s3, acu));

      position.s3 = etasi;

      real4 d = grid - position;
      real Yi = 0.5f * (log(max(momentum.s0+momentum.s3, acu)) \
                     - log(max(momentum.s0-momentum.s3, acu)));
       
     
      real gammaz = cosh( Yi - etasi);


      real nb_one_o_2sigr2 = 1.0/(2.0*NSIGR*NSIGR); 
      real nb_one_o_2sigz2 = 1.0/(2.0*NSIGZ*NSIGZ);
      
      //real distance_sqr = one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2) + one_o_2sigz2*(d.s3*d.s3*gammaz*gammaz*TAU0*TAU0);
      //real nb_distance_sqr = nb_one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2) + nb_one_o_2sigz2*(d.s3*d.s3*gammaz*gammaz*TAU0*TAU0);


      real distance_sqr = one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2) + one_o_2sigz2*(d.s3*d.s3);
      real nb_distance_sqr = nb_one_o_2sigr2*(d.s1*d.s1+d.s2*d.s2) + nb_one_o_2sigz2*(d.s3*d.s3);





      // only do smearing for distance < 10*sigma
      //if ( distance_sqr < 50 ) {
      real maxedgex = max(10*SIGR,DX);
      real maxedgey = max(10*SIGR,DY);
      real maxedgez = max(10*SIGZ,DZ);

      if ( fabs(d.s3) < 10*SIGZ && fabs(d.s1) < 10*SIGR && fabs(d.s2)< 10*SIGR ) {
          delta = exp(- distance_sqr);
          nb_delta = exp(- nb_distance_sqr);

#ifdef SMASH
          real mt = sqrt(momentum.s1 * momentum.s1 + momentum.s2 * momentum.s2+mass*mass);
#endif

#ifdef AMPT
	  real mt = sqrt(max(momentum.s0 * momentum.s0 - momentum.s3 * momentum.s3,
                 momentum.s1 * momentum.s1 + momentum.s2 * momentum.s2));
#endif

	  real Y = 0.5f * (log(max(momentum.s0+momentum.s3, acu)) \
                       - log(max(momentum.s0-momentum.s3, acu)));
          

          // grid.s3 = space-time rapidity of the hydrodynamic cell
          real4 momentum_miln = (real4)(mt*cosh(Y-grid.s3), momentum.s1,
                                        momentum.s2, mt*sinh(Y-grid.s3));
          norm = norm+1e-15; 
          norm2 = norm2+1e-15;
          real number_of_event = NEVENT*1.0f;
          Tm0 += delta * momentum_miln/(norm*number_of_event);
          J0  +=  baryon*nb_delta/(norm2*number_of_event);
          

      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /** KFACTOR=1.6 for LHC energy and 1.45 for RHIC energy */
  Tm0 = KFACTOR * Tm0;
  //J0 = KFACTOR * J0;

  real K2 = Tm0.s1*Tm0.s1 + Tm0.s2*Tm0.s2 + Tm0.s3*Tm0.s3;

  real K0 = Tm0.s0;
  if ( K0 < acu ) K0 = acu;

  real Ed = 0.0f;
  real Nb = 0.0f;
  real M = sqrt(K2);


#ifndef BARYON_ON
  J0 = 0.0;
#endif
  int state;
  rootFinding_newton(&Ed, &Nb,&state, K0, M, J0, eos_table);
  real acu1 = 1e-7;
   
  Ed = max(acu1, Ed);


  real EPV = max(acu, K0+eos_P(Ed, Nb, eos_table));
  
  d_EdV[get_global_id(0)*NY*NZ + get_global_id(1)*NZ + get_global_id(2)] = \
       (real4){Ed, Tm0.s1/EPV, Tm0.s2/EPV, Tm0.s3/EPV};
    
 
  d_nb[get_global_id(0)*NY*NZ + get_global_id(1)*NZ + get_global_id(2)] = Nb;

  }
}

