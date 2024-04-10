#include<Int.h>
#include<distribution.h>
#include "tyche_i.cl"
#include "cl_spinvector.h"


real get_equilibrium_density(real mass,
                             real tfrz,
                             real muB,
                             real fermi_boson,
                             real gspin){
    real sum=0.0;

    for (int ii = 0; ii <48; ii++){
            real momentum_radius = Gausslegp48[ii]*50.0*tfrz + 50.0*tfrz;
            sum = sum + momentum_radius*momentum_radius*juttner_distribution_func(
            momentum_radius,mass,tfrz,muB,fermi_boson)*Gausslegw48[ii]; 
    }

    sum = sum * 50.0*tfrz;

    return gspin*prefactor*sum;
    

}

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

inline real get_density( __global real* d_HadronInfo, __global real4* d_nmtp, int ID , int pid){
        

        real mass  = d_HadronInfo[ 5*pid + 0 ];
        real gspin = d_HadronInfo[ 5*pid + 1 ];
        real fermi_boson = d_HadronInfo[ 5*pid + 2 ];
        real baryon = d_HadronInfo[ 5*pid + 3 ];
        real4 nmtp = d_nmtp[ID];

#ifdef FLAG_MU_PCE
        real muB = d_HadronInfo[ 5*pid + 4 ];
#else
        real muB = baryon * nmtp.s1;
#endif
        real tfrz = nmtp.s2;
           
        real density = get_equilibrium_density(mass,tfrz,muB,fermi_boson,gspin);
        return density;

}

inline int get_type(tyche_i_state* state,  __global real* d_HadronInfo,__global real4* d_nmtp,int I,real dntot_ ){

        
        real sum1 ;
        int index = 0;
        real ratio;
        real ratio1[SizePID];
        real rand2 = tyche_i_float((*state));
 
        ratio1[0] = 0.0;
        sum1= 0.0;
        for(int i = 1 ; i < SizePID ; i ++ ){
            sum1 = sum1 + get_density(d_HadronInfo,d_nmtp,I,i);
            ratio1[i] =sum1;
        }
         for(int i = 1 ; i < SizePID;i ++ )
          {
	     real ratio3 = ratio1[i]/sum1;
              if(rand2 < ratio3){
                  index = i;
                  break;
              }
          }

        return index;
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

// inline int poisson(tyche_i_state* state,real dNtot ){
//     real sum = 0.0f;
//     real rand2 = tyche_i_float((*state));

    

//     int index;
//     for(int i = 0  ; i < 19 ; i++){
//         sum = 0.0f;
//         for(int j = 0; j < i+1; j++){
//                 sum = sum+poisson_distribution_func(dNtot, j);
//         }

//         if (rand2 < sum) {
//             return i;
//         }


//     }

    

//     return index;
// }


inline int poisson(tyche_i_state* state,real dNtot )
{
    int x = -1;
    real u;  
    real log1, log2;
    log1 = 0;
    log2 = -dNtot;
    do
    {
        u =  tyche_i_float((*state));;   
        log1 += log(u);
        x++;   
    }while(log1 >= log2);
    return x;   
}

inline real4 update_expansion_rate(real temperature, real mass, real muB){
    real sum_prob = 0.0f;
    real prob1[5];
    for ( int i = 1; i< 6 ; i ++ ){
        real T_expansion = temperature/(i*1.0f);
        prob1[i-1] = get_equilibrium_density(mass,T_expansion,muB,0.0f,1.0f ); 
        sum_prob = sum_prob + prob1[i-1]; 
    }

    

    real4 prob;
    prob.s0 = prob1[0]/(sum_prob*1.0f);
    prob.s1 = prob1[1]/(sum_prob*1.0f);
    prob.s2 = prob1[2]/(sum_prob*1.0f);
    prob.s3 = prob1[3]/(sum_prob*1.0f);
    return prob;



}

inline real get_momentum1(tyche_i_state* state, real temperature,
                          real mass ){
    real momentum_radial, energy;
    real r0, r1, r2, r3, a, b, c;
    real K, I1, I2, I3, Itot;
    

    
    do {
        r1 = tyche_i_float((*state));
        r2 = tyche_i_float((*state));
        r3 = tyche_i_float((*state));
        a = -log(r1);
        b = -log(r2);
        c = -log(r3);
        momentum_radial = temperature * ( a + b + c );
        energy = sqrt(momentum_radial * momentum_radial + mass * mass);
            
        }while(  tyche_i_float((*state)) - exp((momentum_radial-energy)/ temperature)  > 1e-6);
        

        return momentum_radial;

}


inline real get_momentum2(tyche_i_state* state, real temperature,
                         real mass ){
    real momentum_radial, energy;
    real r0, r1, r2, r3, a, b, c;
    real K, I1, I2, I3, Itot;
    
       do {
            r0 = tyche_i_float((*state));
            I1 = mass*mass;
            I2 = 2.0*mass*temperature;
            I3 = 2.0*temperature*temperature;
            Itot = I1 + I2 + I3;
            if ( r0 < I1/Itot ){
                r1 = tyche_i_float((*state));
                K = -temperature*log(r1);
            }
            else if ( r0 < (I1+I2)/Itot ){
                r1 = tyche_i_float((*state));
                r2 = tyche_i_float((*state));
                K = -temperature*log(r1*r2);
            }
            else {
                r1 = tyche_i_float((*state));
                r2 = tyche_i_float((*state));
                r3 = tyche_i_float((*state));
                K = -temperature*log(r1*r2*r3);
            }
            energy = K + mass;
            momentum_radial = sqrt((energy + mass) * (energy - mass));
            r0 = tyche_i_float((*state));
        
        }
        while (  r0*energy - momentum_radial > 1e-6 ) ;
    
    return momentum_radial;

}



inline real update_tem2(tyche_i_state* state, real temperature,real4 prob  ){
     
        real newT;
        real rand = tyche_i_float((*state));
        if(rand < prob.s0){
            newT = temperature;
        }
        else if( rand < (prob.s0+prob.s1) ){
            newT = temperature*0.5;
        }

        else if( rand < (prob.s0+prob.s1+prob.s2) ){
            newT = temperature*0.333;
        }

        else if( rand < (prob.s0+prob.s1+prob.s2+prob.s3) ){
            newT = temperature*0.25;
        }
        else{
            newT = temperature*0.2;
        }
    


    return newT;




}





__kernel void get_dntot(
        __global real8 * d_SF,
        __global real4 * d_nmtp,
        __global real4 * d_txyz,
       __global real * d_HadronInfo,
       __global ulong* d_seed,
       __global real* d_dntot){


    int II = get_global_id(0);
    

    ulong seed1 = abs(d_seed[II]);
    tyche_i_state state;
    tyche_i_seed(&state,seed1);
    int length = SizeSF/( get_global_size(0) );

    
    for(int I = get_global_id(0);I <SizeSF; I = I + get_global_size(0) ){

        real dntot_= 0.0f;
        for(int pid = 1; pid< SizePID ; pid ++ ){
            real density = get_density(d_HadronInfo,d_nmtp,I,pid);
            dntot_ = dntot_ + density; 
               
        }
        real8 SF = d_SF[I];
        real4 dsigma = SF.s0123;
        real4 flowvelocity = (real4)(1.0f, SF.s4, SF.s5, SF.s6) ;
        real4 sigma_lrf = LorentzBoost(flowvelocity, dsigma);
        
        real sigma_lrf0 = sigma_lrf.s0;
        if(sigma_lrf0 < 0.0) continue;
        real dNtot = dntot_ * sigma_lrf0;

        //int Ni = poisson(&state, dNtot);
        d_dntot[I] = dNtot;

  
    }



     
                 

}


__kernel void get_poisson(__global int2* d_poisson,
                          __global ulong* d_seed,
                          __global real* d_dntot,
                          __global  int * d_Npoisson){

    int II = get_global_id(0);
    

    ulong seed1 = abs(d_seed[II]);
    tyche_i_state state;
    tyche_i_seed(&state,seed1);

    for(int I = get_global_id(0);I <SizeSF; I = I + get_global_size(0) ){
        int Ni = poisson(&state,d_dntot[I]);
        //int Ni = 1;
        //int Ni2 = 2;
        //printf("%d %d \n",Ni,I);
        //real rand2 = tyche_i_float((state));
       
        if(Ni > 0){
        int id_ = atomic_inc(d_Npoisson);
        
        d_poisson[id_] = (int2){Ni,I};
        }
        
    
    }

}



__kernel void classify_ptc(
        __global real8 * d_SF,
        __global real4 * d_nmtp,
        __global real4 * d_txyz,
       __global real * d_HadronInfo,
       __global ulong* d_seed,
       __global  real4 * d_hptc,
       __global  real4 * d_lptc,
       __global  int * Nhptc,
       __global  int * Nlptc,
       __global  int2 * d_poisson,
       __global  real * d_dntot,
       const  int Ncell){


    int II = get_global_id(0);
    

    ulong seed1 = abs(d_seed[II]);
    tyche_i_state state;//SizeSF
    tyche_i_seed(&state,seed1);

     int length = SizeSF/( get_global_size(0) );

    int size1 = get_global_size(0);
    
    for(int I = get_global_id(0);I <Ncell; I = I + get_global_size(0) ){
        
        int2 poisson  = d_poisson[I];
        int ID = poisson.s1;
        int Ni = poisson.s0;
        real dNtot= d_dntot[ID];


        real8 SF = d_SF[ID];
        real4 dsigma = SF.s0123;
        real4 flowvelocity = (real4)(1.0f, SF.s4, SF.s5, SF.s6) ;
        real4 sigma_lrf = LorentzBoost(flowvelocity, dsigma);
        
        real sigma_lrf0 = sigma_lrf.s0;
        if(sigma_lrf0 < 0.0) continue;
        real dntot_ = dNtot/ (sigma_lrf0*1.0);
    
        for ( int sampled_hadrons = 0; sampled_hadrons < Ni ; sampled_hadrons++  )
        {
            
            int type = get_type(&state,d_HadronInfo,d_nmtp,ID,dntot_); 
            real  mass = d_HadronInfo[5*type + 0 ];
            real gspin = d_HadronInfo[5*type + 1 ];
            real fermi_boson = d_HadronInfo[5*type + 2 ];
            real baryon = d_HadronInfo[5*type + 3 ];
    

            real4 nmtp = d_nmtp[ID];
#ifdef FLAG_MU_PCE
            real muB = d_HadronInfo[5*type + 4 ];
#else
            real muB = baryon * nmtp.s1;
#endif
            real tfrz = nmtp.s2;
            

          
            real4 prob = (real4) {0.0f, 0.0f, 0.0f, 0.0f };
            
            real newT = tfrz;
           
            if(fabs(baryon) < 0.0001 ){
                prob = update_expansion_rate(tfrz, mass, muB);
                newT = update_tem2(&state, tfrz, prob );
            }

            
            if(newT > 0.6f * mass){
                int flag = 0 ;
                int id_ = atomic_inc(Nlptc);
                real4 ptc = (real4){ ID ,type, newT, flag }; 
                d_lptc[id_] = ptc; 
            }
            else{
                int flag = 1 ;
                int id_ = atomic_inc(Nhptc);
                real4 ptc = (real4){ ID ,type, newT, flag }; 
                d_hptc[id_] = ptc; 
            }

         
            
        }
        

  
        }



     
                 

}

__kernel void sample_lighthadron(
        __global real8 * d_SF,
        __global real4 * d_nmtp,
        __global real4 * d_txyz,
        __global real * d_HadronInfo,
            __global real  * d_pi,            
            __global real  * d_qb, 
            __global real  * d_deltaf_qmu,           
        __global ulong* d_seed,
        __global  real4 * d_lptc,
        __global  int * Nlptc,
        __global  int * Nhptc,
        __global real8* d_ptc){


    int II = get_global_id(0);
    ulong seed1 = abs(d_seed[II]);
    tyche_i_state state;//SizeSF
    tyche_i_seed(&state,seed1);
    int nptc = (*Nlptc);
    int Nh = (*Nhptc);
    for(int I = get_global_id(0); I < nptc; I = I + get_global_size(0) ){
        real4 ptc = d_lptc[I];
        int type = (int) ptc.s1;
        int ID  = (int) ptc.s0;
        real newT = ptc.s2;
        real mass = d_HadronInfo[ 5*type+0 ];
        real gspin = d_HadronInfo[ 5*type+1 ];
        real fermi_boson = d_HadronInfo[ 5*type+2 ];
        real baryon = d_HadronInfo[ 5*type+3 ];

        real pmag = get_momentum1(&state, newT,  mass );
        
        real8 SF = d_SF[ID];
        real4 dsigma = SF.s0123;
        real4 txyz = d_txyz[ID];
        real4 flowvelocity = (real4)(1.0f, SF.s4, SF.s5, SF.s6) ;
        real4 sigma_lrf = LorentzBoost(flowvelocity, dsigma);
        real sigma_lrf0 = sigma_lrf.s0;
        real sigma_max = sigma_lrf0 + sqrt( sigma_lrf.s1*sigma_lrf.s1 
                       + sigma_lrf.s2*sigma_lrf.s2+sigma_lrf.s3*sigma_lrf.s3 );
        real etas = SF.s7;
        
        real4 umu = (real4)(1.0f, SF.s4, SF.s5, SF.s6) * \
            1.0f/sqrt(max((real)1.0E-15f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        real4 nmtp = d_nmtp[ID];     
#ifdef FLAG_MU_PCE
        real muB = d_HadronInfo[ 5*type+4 ];	
        real chem = muB;
#else
        real muB = nmtp.s1;
        real chem = muB*baryon;
#endif
       
        real tfrz = nmtp.s2;
        real eplusp = nmtp.s3;
        real nb = nmtp.s0;

        real pimn[10];

        real pimnmax=0.0f;
        real one_over_2TsqrEplusP = 1.0f/(2.0f*tfrz*tfrz*eplusp );
        for(int ii = 0 ; ii < 10 ; ii = ii+1 )
        {
            pimn[ii] = d_pi[ID*10 + ii];
            pimnmax = pimnmax + pimn[ii]*pimn[ii];
        }

        pimnmax = sqrt(pimnmax);

        real4 qb1;
        qb1 = (real4) {d_qb[ID*4 + 0],d_qb[ID*4 + 1],d_qb[ID*4 + 2],d_qb[ID*4 + 3] };
        real4 qb = LorentzBoost(flowvelocity, qb1);
        real qb_max = qb.s0 + sqrt(qb.s1*qb.s1 + qb.s2*qb.s2 + qb.s3*qb.s3 );



        int NLOOP = 0;
        while(true){
           //real pi1 = M_PI_F;
           real phi = tyche_i_float(state)*M_PI_F*2.0f;
           real costheta = tyche_i_float(state)*2.0f - 1.0f;
           real sintheta = sqrt(1.0f - costheta*costheta);

           real px_lrf = pmag*sintheta*cos(phi);
           real py_lrf = pmag*sintheta*sin(phi);
           real pz_lrf = pmag*costheta;
           real energy_lrf = sqrt(pmag*pmag + mass*mass);
           real4 momentum_in_lrf = (real4) {energy_lrf,px_lrf,py_lrf,pz_lrf};
           real pdotsigma = momentum_in_lrf.s0*sigma_lrf.s0
                          - momentum_in_lrf.s1*sigma_lrf.s1
                          - momentum_in_lrf.s2*sigma_lrf.s2
                          - momentum_in_lrf.s3*sigma_lrf.s3;


           real weight_visc = pdotsigma/(energy_lrf*sigma_max);

           real pmu_pnu_pimn = momentum_in_lrf.s0*momentum_in_lrf.s0*pimn[0]
                               + momentum_in_lrf.s1*momentum_in_lrf.s1*pimn[4]
                               + momentum_in_lrf.s2*momentum_in_lrf.s2*pimn[7]
                               + momentum_in_lrf.s3*momentum_in_lrf.s3*pimn[9]
                               - 2.0*momentum_in_lrf.s0*momentum_in_lrf.s1*pimn[1]
                               - 2.0*momentum_in_lrf.s0*momentum_in_lrf.s2*pimn[2]
                               - 2.0*momentum_in_lrf.s0*momentum_in_lrf.s3*pimn[3]
                               + 2.0*momentum_in_lrf.s1*momentum_in_lrf.s2*pimn[5]
                               + 2.0*momentum_in_lrf.s1*momentum_in_lrf.s3*pimn[6]
                               + 2.0*momentum_in_lrf.s2*momentum_in_lrf.s3*pimn[8];
           real f0 = juttner_distribution_func( pmag,mass,tfrz,chem,fermi_boson);
           real weight_shear = pmu_pnu_pimn*one_over_2TsqrEplusP ; 
           real weight_shear_max = energy_lrf*energy_lrf*pimnmax*one_over_2TsqrEplusP;
           
           real nb_o_ep = nb/eplusp;

           real b_o_e = baryon/energy_lrf;
           real kappa_hat =get_deltaf_qmn(tfrz, muB,d_deltaf_qmu);// max(nb*(1.0f/(tanh(muB/tfrz)*3.0f+1e-6) - nb_o_ep*tfrz ),1e-6);
           real pdotq = momentum_in_lrf.s0*qb.s0
                      - momentum_in_lrf.s1*qb.s1
                      - momentum_in_lrf.s2*qb.s2
                      - momentum_in_lrf.s3*qb.s3;
           real weigth_nb = (nb_o_ep - b_o_e)*pdotq/kappa_hat;
           real weigth_nbmax = fabs(nb_o_ep - b_o_e)*energy_lrf*qb_max/kappa_hat;

           weight_visc = weight_visc*(1.0f + (1.0f - fermi_boson*f0)*(weight_shear + weigth_nb  ));
           weight_visc = weight_visc/(1.0f + fabs(1.0f - fermi_boson*f0)*(weight_shear_max+weigth_nbmax));
          
           
           NLOOP = NLOOP + 1;
           if(NLOOP > 10000 && weight_visc < 0.0f)
           {
               real8 ptc1 = (real8) {NAN,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
               d_ptc[I] = ptc1;
                break;
           }

           real rand1 = tyche_i_float(state);
           if(rand1 <= weight_visc){
	       

               real4 momentum = LorentzBoost(-flowvelocity, momentum_in_lrf);
               real mt = sqrt(momentum.s0*momentum.s0 - momentum.s3*momentum.s3 );
               real rapidity = atanh(momentum.s3/(momentum.s0)) + etas;
               real4 momentum1 = (real4){mt*cosh(rapidity), momentum.s1, momentum.s2,mt*sinh(rapidity)};
               

              

	           real tiny_move = tyche_i_float(state)*1e-6;
               real4 position = (real4){txyz.s0,txyz.s1+tiny_move,txyz.s2+tiny_move,txyz.s3+tiny_move};
               real8 ptc1 = (real8) {momentum1.s0,momentum1.s1,momentum1.s2,momentum1.s3,
                                     position.s0,position.s1,position.s2,position.s3};
               d_ptc[I+Nh] = ptc1;
               break;
           }

        }
        
        

    
    
    }






}


__kernel void sample_heavyhadron(
        __global real8 * d_SF,
        __global real4 * d_nmtp,
        __global real4 * d_txyz,
        __global real * d_HadronInfo,
        __global real  * d_pi,            
        __global real  * d_qb, 
        __global real  * d_deltaf_qmu,         
        __global ulong* d_seed,
        __global  real4 * d_hptc,
        __global  int * Nhptc,
        __global real8* d_ptc){


    int II = get_global_id(0);
    ulong seed1 = abs(d_seed[II]);
    tyche_i_state state;//SizeSF
    tyche_i_seed(&state,seed1);
    int nptc = (*Nhptc);
    for(int I = get_global_id(0); I < nptc; I = I + get_global_size(0) ){
        real4 ptc = d_hptc[I];
        int type = (int) ptc.s1;
        int ID  = (int) ptc.s0;
        real newT = ptc.s2;
        real mass =  d_HadronInfo[ 5*type+0 ];
        real gspin = d_HadronInfo[ 5*type+1 ];
        real fermi_boson = d_HadronInfo[ 5*type+2 ];
        real baryon = d_HadronInfo[ 5*type+3 ];
        
        real pmag = get_momentum2(&state, newT,  mass );

        real8 SF = d_SF[ID];
        real4 dsigma = SF.s0123;
        real4 txyz = d_txyz[ID];
        real4 flowvelocity = (real4)(1.0f, SF.s4, SF.s5, SF.s6) ;
        real4 sigma_lrf = LorentzBoost(flowvelocity, dsigma);
        real sigma_lrf0 = sigma_lrf.s0;
        real sigma_max = sigma_lrf0 + sqrt( sigma_lrf.s1*sigma_lrf.s1 
                       + sigma_lrf.s2*sigma_lrf.s2+sigma_lrf.s3*sigma_lrf.s3 );
        real etas = SF.s7;

        real4 umu = (real4)(1.0f, SF.s4, SF.s5, SF.s6) * \
            1.0f/sqrt(max((real)1.0E-15f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        
        real4 nmtp = d_nmtp[ID];

#ifdef FLAG_MU_PCE
        real muB = d_HadronInfo[ 5*type+4 ];	
        real chem = muB;
#else
        real muB = nmtp.s1;
        real chem = muB*baryon;
#endif
        real tfrz = nmtp.s2;
        real eplusp = nmtp.s3;
        real nb = nmtp.s0;

        real pimn[10];
        real pimnmax=0.0f;
        real one_over_2TsqrEplusP = 1.0f/(2.0f*tfrz*tfrz*eplusp );
        for(int ii = 0 ; ii < 10 ; ii = ii+1 )
        {
            pimn[ii] = d_pi[ID*10 + ii];
            pimnmax = pimnmax + pimn[ii]*pimn[ii];
        }

        pimnmax = sqrt(pimnmax);


        real4 qb1;
        qb1 = (real4) {d_qb[ID*4 + 0],d_qb[ID*4 + 1],d_qb[ID*4 + 2],d_qb[ID*4 + 3] };
        real4 qb = LorentzBoost(flowvelocity, qb1);
        real qb_max = qb.s0 + sqrt(qb.s1*qb.s1 + qb.s2*qb.s2 + qb.s3*qb.s3 );

        int NLOOP = 0;
        while(true){
            real phi = tyche_i_float(state)*M_PI_F*2.0f;
            real costheta = tyche_i_float(state)*2.0f - 1.0f;
            real sintheta = sqrt(1.0f - costheta*costheta);

            real px_lrf = pmag*sintheta*cos(phi);
            real py_lrf = pmag*sintheta*sin(phi);
            real pz_lrf = pmag*costheta;
            real energy_lrf = sqrt(pmag*pmag + mass*mass);
            real4 momentum_in_lrf = (real4) {energy_lrf,px_lrf,py_lrf,pz_lrf};
            real pdotsigma = momentum_in_lrf.s0*sigma_lrf.s0
                          - momentum_in_lrf.s1*sigma_lrf.s1
                          - momentum_in_lrf.s2*sigma_lrf.s2
                          - momentum_in_lrf.s3*sigma_lrf.s3;
	   
			  
           real weight_visc = pdotsigma/(energy_lrf*sigma_max);

           real pmu_pnu_pimn = momentum_in_lrf.s0*momentum_in_lrf.s0*pimn[0]
                               + momentum_in_lrf.s1*momentum_in_lrf.s1*pimn[4]
                               + momentum_in_lrf.s2*momentum_in_lrf.s2*pimn[7]
                               + momentum_in_lrf.s3*momentum_in_lrf.s3*pimn[9]
                               - 2.0*momentum_in_lrf.s0*momentum_in_lrf.s1*pimn[1]
                               - 2.0*momentum_in_lrf.s0*momentum_in_lrf.s2*pimn[2]
                               - 2.0*momentum_in_lrf.s0*momentum_in_lrf.s3*pimn[3]
                               + 2.0*momentum_in_lrf.s1*momentum_in_lrf.s2*pimn[5]
                               + 2.0*momentum_in_lrf.s1*momentum_in_lrf.s3*pimn[6]
                               + 2.0*momentum_in_lrf.s2*momentum_in_lrf.s3*pimn[8];
           real f0 = juttner_distribution_func( pmag,mass,tfrz,chem,fermi_boson);

           real weight_shear = pmu_pnu_pimn*one_over_2TsqrEplusP ; 
           real weight_shear_max = energy_lrf*energy_lrf*pimnmax*one_over_2TsqrEplusP;
           
           real nb_o_ep = nb/eplusp;
           real b_o_e = baryon/energy_lrf;
           real kappa_hat = get_deltaf_qmn(tfrz, muB,d_deltaf_qmu);//max(nb*(1.0f/(tanh(muB/tfrz)*3.0f+1e-6) - nb_o_ep*tfrz ),1e-6);
           real pdotq = momentum_in_lrf.s0*qb.s0
                      - momentum_in_lrf.s1*qb.s1
                      - momentum_in_lrf.s2*qb.s2
                      - momentum_in_lrf.s3*qb.s3;
           real weight_nb = (nb_o_ep - b_o_e)*pdotq/kappa_hat;
           real weight_nbmax = fabs(nb_o_ep - b_o_e)*energy_lrf*qb_max/kappa_hat;

           weight_visc = weight_visc*(1.0f + (1.0f - fermi_boson*f0)*(weight_shear + weight_nb  ));
           weight_visc = weight_visc/(1.0f + fabs(1.0f - fermi_boson*f0)*(weight_shear_max+weight_nbmax));
           

           

           real rand1 = tyche_i_float(state);
           NLOOP = NLOOP + 1;
           if(NLOOP > 10000 && weight_visc < 0.0f)
           {
               real8 ptc1 = (real8) {NAN,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
               d_ptc[I] = ptc1;
               break;
           }

           if(rand1 <= weight_visc){
               real4 momentum = LorentzBoost(-flowvelocity, momentum_in_lrf);
               real mt = sqrt(momentum.s0*momentum.s0 - momentum.s3*momentum.s3 );
               real rapidity = atanh(momentum.s3/(momentum.s0)) + etas;
               real4 momentum1 = (real4){mt*cosh(rapidity), momentum.s1, momentum.s2,mt*sinh(rapidity)};
               
	           real tiny_move = tyche_i_float(state)*1e-6;
               real4 position = (real4){txyz.s0,txyz.s1+tiny_move,txyz.s2+tiny_move,txyz.s3+tiny_move};
               real8 ptc1 = (real8) {momentum1.s0,momentum1.s1,momentum1.s2,momentum1.s3,
                                     position.s0,position.s1,position.s2,position.s3};

               d_ptc[I] = ptc1;
               break;
            }
        }


    }

}


__kernel void sample_vorticity(
        __global real8 * d_SF,
        __global real4 * d_nmtp,
        __global real * d_HadronInfo,
        __global real * d_omega_th,
        __global real * d_omega_shear1,
        __global real * d_omega_shear2,
        __global real * d_omega_accT,
        __global real * d_omega_chemical,
        __global real4* d_lptc,
        __global real4* d_hptc,
        __global  int * Nlptc,
        __global  int * Nhptc,
        __global real4 *d_spin_th,
        __global real4 *d_spin_shear,
        __global real4 *d_spin_accT,
        __global real4 *d_spin_chemical,
        __global real8 *d_ptc
        ){


        int nh = (*Nhptc);
        int nl = (*Nlptc);

        for(int I = get_global_id(0);I < nh+nl; I = I + get_global_size(0) )
        {

            real4 ptc;
            if (I < nh)
            {
                ptc = d_hptc[I];

            }
            else{

                ptc = d_lptc[I-nh];
            }

            int type = (int) ptc.s1;
            int ID  = (int) ptc.s0;
            real8 SF = d_SF[ID];
            real mass =  d_HadronInfo[ 5*type+0 ];
            real gspin = d_HadronInfo[ 5*type+1 ];
            real fermi_boson = d_HadronInfo[ 5*type+2 ];
            real baryon = d_HadronInfo[ 5*type+3 ];

             real4 nmtp = d_nmtp[ID];
#ifdef FLAG_MU_PCE
            real muB = d_HadronInfo[ 5*type+4 ];
            real chem = muB;
#else
            real muB = nmtp.s1;
            real chem = muB*baryon;
#endif


            real8 ptc1 = d_ptc[I];
           
	        real tfrz = nmtp.s2;
            d_spin_th[I] = get_spin_vector_th(SF,ptc1,d_omega_th,ID,mass,gspin,fermi_boson,tfrz,chem);
            d_spin_shear[I] = get_spin_vector_shear(SF,ptc1,d_omega_shear1,d_omega_shear2,ID,mass,gspin,fermi_boson,tfrz,chem);
            d_spin_accT[I] = get_spin_vector_accT(SF,ptc1,d_omega_accT,ID,mass,gspin,fermi_boson,tfrz,chem);
            d_spin_chemical[I] = get_spin_vector_chemical(SF,ptc1,d_omega_chemical,ID,mass,gspin,fermi_boson,tfrz,chem);



        }

}

