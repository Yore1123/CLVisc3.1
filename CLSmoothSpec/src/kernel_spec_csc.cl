#include<helper.h>

inline real get_deltaf_qmn(real T, real mu,__global real *d_deltaf_qmu){
    int idx_T =  (T - T0)/T_STEP ;
    int idx_mu = (mu - MU0)/MU_STEP ;

    real y = ( T - (idx_T*T_STEP+T0) )/T_STEP;
    real x = ( mu - (idx_mu*MU_STEP+MU0) )/MU_STEP;

    if (idx_T > TLENGTH - 2){
        return 1e20;
    }


    if (idx_mu > MULENGTH - 2){
        return 1e20;
    }

    if (idx_mu < 0){
        return 1e20;
    }
    if(idx_T < 0 ){
        return 1e20;
    }

    real ll = d_deltaf_qmu[idx_mu*TLENGTH + idx_T];
    real lh = d_deltaf_qmu[idx_mu*TLENGTH + idx_T+1];
    real hl = d_deltaf_qmu[(idx_mu+1)*TLENGTH + idx_T];
    real hh = d_deltaf_qmu[(idx_mu+1)*TLENGTH + idx_T+1];

    return (1-x)*(1-y)*ll + (1-x)*y*lh + x*(1-y)*hl + x*y*hh;

}

inline real4 LorentzBoost(real4 volocity, real4 ele){

     real velocity_squared  = volocity.s1*volocity.s1+volocity.s2*volocity.s2+volocity.s3*volocity.s3;
     real gamma = velocity_squared < 1.0f ? 1.0f/sqrt(1 - velocity_squared ) : 0.0f ;
     real xprime_0 = gamma * (ele.s0 - ele.s1*volocity.s1 - ele.s2*volocity.s2 - ele.s3*volocity.s3 );
     real constantpart = gamma/ ( gamma + 1.0f ) * (xprime_0 + ele.s0);

     real elestar0 = xprime_0;
     real elestar1 = ele.s1 - volocity.s1*constantpart;
     real elestar2 = ele.s2 - volocity.s2*constantpart;
     real elestar3 = ele.s3 - volocity.s3*constantpart;

     real4 res = (real4){elestar0,elestar1,elestar2,elestar3};
     return res;

}



/** First stage: calc sub spec for each particle */
__kernel void get_sub_dNdYPtdPtdPhi(  
            __global real8  * d_SF,            
            __global real  * d_qb,
            __global real  * d_deltaf_qmu,
            __global real  * d_pi, 
            __global real4  * d_nmtp,            
            __global real  * d_SubSpec,            
            __constant real * d_HadronInfo,           
            __constant real * d_Y ,           
            __constant real * d_Pt,
            __constant real * d_CPhi,
            __constant real * d_SPhi,
            const int pid,
            const int id_Y_PT_PHI)
{
    int I = get_global_id(0);
    
    int tid = get_local_id(0) ;
    
    __local real subspec[ BSZ ];
    real acu=1e-7; 
    
    real mass  = d_HadronInfo[ pid*5+0 ];
    real gspin = d_HadronInfo[ pid*5+1 ];
    real fermi_boson =  d_HadronInfo[ pid*5+2 ];
    real baryon = d_HadronInfo[ pid*5+3 ];
    real muB = d_HadronInfo[pid*5+4];

    real dof = gspin / pown(2.0f*M_PI_F, 3);

    real pimn[10];
    real qb[4];



    int k = id_Y_PT_PHI / NPT;
    int l = id_Y_PT_PHI - k*NPT;
    real rapidity = d_Y[k];
    real pt = d_Pt[l];
    real mt = sqrt(mass*mass + pt*pt); 

    real dNdYPtdPtdPhi[NPHI];

    for ( int m = 0; m < NPHI; m++ ) {
        dNdYPtdPtdPhi[m] = 0.0;
    }
    
    while ( I < SizeSF ) {
        real8 SF = d_SF[I];

        real4  nmtp = d_nmtp[I];
	real   nb = nmtp.s0;

#ifndef FLAG_MU_PCE
        muB = nmtp.s1*baryon;
#endif
	real   tfrz= nmtp.s2;
	real   prplused  = nmtp.s3;

    for ( int id = 0; id < 10; id ++ ) {
        pimn[id] = d_pi[10*I + id];
    }
        

    for ( int id = 0; id < 4; id ++ ) {
	    qb[id] = d_qb[4*I+id];
    }

    real4 qbmu = (real4)(qb[0], qb[1], qb[2], qb[3]);

    real4 umu = (real4)(1.0f, SF.s4, SF.s5, SF.s6) * \
        1.0f/sqrt(max((real)1.0E-15f, \
        (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
    real4 dsigma = SF.s0123;
        
    real mtcosh = mt*cosh(rapidity-SF.s7);
    real mtsinh = mt*sinh(rapidity-SF.s7);
        
        for(int m=0; m<NPHI; m++){
            real4 pmu = (real4)(mtcosh, -pt*d_CPhi[m], -pt*d_SPhi[m], -mtsinh);
            double feq = 1.0f/( exp((real)((dot(pmu, umu)-muB)/tfrz)) + fermi_boson );
            double feq1 = feq;


            // TCOEFF = 1.0f/(2T^2 (e+P)) on freeze out hyper sf from compile options
	    real tcoeff =1.0f/(2*tfrz*tfrz*(prplused));

            real p2pi_o_T2ep = tcoeff*(pmu.s0*pmu.s0*pimn[0] + pmu.s3*pmu.s3*pimn[9] +
                               2.0f*pmu.s0*(pmu.s1*pimn[1] + pmu.s2*pimn[2] + pmu.s3*pimn[3]) +
                               (pmu.s1*pmu.s1*pimn[4] + 2*pmu.s1*pmu.s2*pimn[5] + pmu.s2*pmu.s2*pimn[7]) +
                               2.0f*pmu.s3*(pmu.s1*pimn[3] + pmu.s2*pimn[6]));

            double df = feq1*(1.0f - fermi_boson*feq1)*p2pi_o_T2ep;

            double df_nb = 0.0;
	    real nb_o_ep = nb/(prplused);
	    real b_o_e = baryon/(dot(pmu,umu));
            real kappa_hat = get_deltaf_qmn(tfrz, nmtp.s1,d_deltaf_qmu); //nb*(1.0f/(tanh(muB/tfrz)*3.0f) - nb_o_ep*tfrz );
            real4 pmu_u = (real4)(mtcosh, pt*d_CPhi[m], pt*d_SPhi[m], mtsinh);
	    real  pq = dot(pmu,qbmu);
          
	    df_nb = feq1*(1.0f - fermi_boson*feq1)*(nb_o_ep - b_o_e)*pq/(kappa_hat+acu);
            
            

            double df_total = df + df_nb;

            feq += fabs(df_total) > feq1 ? df_total*feq1/fabs(df_total) : df_total;

            real4 velocity = (real4)(1.0,SF.s4, SF.s5, SF.s6);
            real4 sigma_lrf = LorentzBoost(velocity, dsigma);
            if(sigma_lrf.s0 < 0.0 ) continue; 
            
            dNdYPtdPtdPhi[m] += dof * dot(pmu, dsigma) * feq;
	    
        }
        
        /** in units of GeV.fm^3 */
        I += get_global_size(0);
    }
    
    for ( int m = 0; m < NPHI; m++ ) {
        subspec[tid] = dNdYPtdPtdPhi[m];
        barrier(CLK_LOCAL_MEM_FENCE);
    
        //// do reduction in shared mem
        for ( unsigned int s = get_local_size(0) >> 1; s > 0; s >>= 1 ) {
            if (tid < s) {
                subspec[tid] += subspec[tid + s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if(tid == 0) d_SubSpec[k*NPT*NPHI*NBlocks + l*NPHI*NBlocks \
               + m*NBlocks + get_group_id(0) ] = subspec[0];
    }
}


/** The second stage on sum reduction 
*   For each Y,Pt,Phi we need to sum NBlocks dNdYptdptdphi from the first stage */
__kernel void get_dNdYPtdPtdPhi(  
                                  __global real  * d_SubSpec,            
                                  __global real  * d_Spec,
                                  const int pid)
{
    int I = get_global_id(0);
    int tid = get_local_id(0);
    int localSize = get_local_size(0);
    real acu=1e-7; 
    
    __local real spec[ NBlocks ];
    
    spec[ tid ] = d_SubSpec[ I ];
    
    barrier( CLK_LOCAL_MEM_FENCE );
    
    for( int s = localSize>>1; s>0; s >>= 1 ){
         if( tid < s ){
             spec[ tid ] += spec[ s + tid ];
         }
         barrier( CLK_LOCAL_MEM_FENCE );
    }

    /** unroll the last warp because they are synced automatically */
    /** \bug unroll gives out wrong results */
    //if( NBlocks >= 512 ){ if( tid < 256 ) spec[tid] += spec[ 256 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };
    //if( NBlocks >= 256 ){ if( tid < 128 ) spec[tid] += spec[ 128 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };
    //if( NBlocks >= 128 ){ if( tid < 64 )  spec[tid] += spec[ 64 + tid ]; barrier( CLK_LOCAL_MEM_FENCE ); };

    //if( tid < 32 ){
    //   if( NBlocks >= 64 ) spec[ tid ] += spec[ 32 + tid ];
    //   if( NBlocks >= 32 ) spec[ tid ] += spec[ 16 + tid ];
    //   if( NBlocks >= 16 ) spec[ tid ] += spec[ 8 + tid ];
    //   if( NBlocks >= 8 )  spec[ tid ] += spec[ 4 + tid ];
    //   if( NBlocks >= 4 )  spec[ tid ] += spec[ 2 + tid ];
    //   if( NBlocks >= 2 )  spec[ tid ] += spec[ 1 + tid ];
    //}
    real acu1 = 1e-11;
    if( tid == 0 ) d_Spec[ pid*NY*NPT*NPHI + get_group_id(0) ] = max(spec[ tid ], acu1);
}
