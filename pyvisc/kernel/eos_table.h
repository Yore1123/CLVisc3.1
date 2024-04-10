#ifndef __EOS_H__
#define __EOS_H__

#include "real_type.h"
constant real hbarc = 0.1973269f;


/** \breif EOS EOSI, s95p-PCE165-v0 from TECHQM */
/** Pressure as a function of energy density in units GeV/fm^3 */
#ifdef IDEAL_GAS
//
#define  dof  (169.0f/4.0f)
#define  hbarc1  0.1973269f
#define  hbarc3  pow(0.1973269631f, 3.0f)
#define  coef  (M_PI_F*M_PI_F/30.0f)
//
inline real eos_P(real eps, real nb, __global real* eos_table){
     return eps/3.0f;
}
//
inline real eos_T(real eps, real nb, __global real* eos_table){
     return  hbarc1*pow( (real)1.0f/(dof*coef)*eps/hbarc1, (real)0.25f);
}
//
inline real eos_s(real eps, real nb, __global real* eos_table){
     return  ( eps + eos_P(eps, nb, eos_table)) / fmax((real)1.0E-10f, eos_T(eps, nb,eos_table));
}

inline real eos_CS2(real eps, real nb, __global real* eos_table){
     return  0.33333333f;
}

inline real eos_mu(real eps, real nb, __global real* eos_table){
     return  0.0f;
}


inline real eos_s2ed(real s, real nb, __global real* eos_table){


       real ed_max = 10000.0f*hbarc;
       real ed_min = 1e-15f;
       real ed_mid = (ed_max+ed_min)/2.0f;

       real s_lo = eos_s(ed_min,nb,eos_table);
       if (s < s_lo)
       {
           return ed_min;
       }
       real s_hi = eos_s(ed_max,nb,eos_table);
       real s_mid;
       int i=0;
       while( ((ed_max-ed_min)>1e-6) && (i <100) &&(  ((ed_max-ed_min)/ed_mid)>1e-12)  ){

           s_mid = eos_s(ed_mid,nb,eos_table);
           if(s < s_mid){
               ed_max = ed_mid;
           }
           else{
               ed_min = ed_mid;
           }

           ed_mid = (ed_max + ed_min)/2.0f;
           i++;

       }

       return ed_mid;


}

#endif



#if  ((defined LATTICE_PCE165)|| (defined LATTICE_PCE150) || (defined LATTICE_WB) || (defined HOTQCD2014) ||  (defined FIRST_ORDER) || (PURE_GAUGE)) 


inline real interpolate1D(real eps, __global real* eos_table, int kind){
    real e0 = EOS_ED_START;
    real d_e = EOS_ED_STEP;
    int  N_e = EOS_NUM_ED;
    int idx_e = (eps- e0)/d_e;
    idx_e  = min(N_e -2 , idx_e);
    idx_e = max(0,idx_e);
    real frac_e = (eps - (idx_e*d_e + e0) )/d_e;
    int idx1 = kind*N_e + idx_e;
    int idx2 = kind*N_e + idx_e+1;
    real result = eos_table[idx1]*(1.0 - frac_e) + eos_table[idx2]*frac_e;
    return result;
}

// get the pressure from eos_table
inline real eos_P(real eps, real nb, __global real* eos_table){
        int kind = 1;
        return interpolate1D(eps, eos_table,kind);
}

// get the entropy density from eos_table
inline real eos_s(real eps, real nb, __global real* eos_table){
        int kind = 3;
        return interpolate1D(eps, eos_table,kind);
}


// get the temperature from eos_table
inline real eos_T(real eps, real nb, __global real* eos_table){
        int kind = 2;
        return interpolate1D(eps, eos_table,kind);
}

// get the speed of sound square
inline real eos_CS2(real eps, real nb, __global real* eos_table){
        int kind = 0;

        real cs2_max = 1.0f/3.0f;
        real cs2_min = 0.01f;
        real cs2 = interpolate1D(eps, eos_table,kind);
    
        return max(cs2_min,min(cs2,cs2_max));
}


inline real eos_mu(real eps, real nb, __global real* eos_table){
        return 0.0f;
}

inline real eos_mus(real eps, real nb, __global real* eos_table){
        return 0.0f;
}

inline real eos_muQ(real eps, real nb, __global real* eos_table){
        return 0.0f;
}



inline real eos_s2ed(real s, real nb, __global real* eos_table){
    
       


       real ed_max = 10000.0f*hbarc;
       real ed_min = 1e-15f;
       real ed_mid = (ed_max+ed_min)/2.0f;
       
       real s_lo = eos_s(ed_min,nb,eos_table);
       if (s < s_lo)
       {
           return ed_min;
       }
       real s_hi = eos_s(ed_max,nb,eos_table);
       real s_mid;
       int i=0;
       while( ((ed_max-ed_min)>1e-6) && (i <100) &&(  ((ed_max-ed_min)/ed_mid)>1e-12)  ){
     
           s_mid = eos_s(ed_mid,nb,eos_table);
           if(s < s_mid){
               ed_max = ed_mid;
           }
           else{
               ed_min = ed_mid;
           }

           ed_mid = (ed_max + ed_min)/2.0f;
           i++;
           
       }       
       
       return ed_mid;
    

}


inline real eos_dpde(real ed, real nb, __global real* eos_table){
    real edlo = 0.9*ed;
    real edhi = 1.1*ed;

    real plo = eos_P(edlo,nb,eos_table);
    real phi = eos_P(edhi,nb,eos_table);

    real dpde = (phi-plo)/(edhi-edlo);
    return dpde;
}

inline real eos_dpdn(real ed, real nb, __global real* eos_table){
    
    real nblo = 0.9*nb;
    real nbhi = 1.1*nb;

    real plo = eos_P(ed,nblo,eos_table);
    real phi = eos_P(ed,nbhi,eos_table);

    real dpdnb = (phi-plo)/(nbhi-nblo);
    return dpdnb;
}

#endif


#ifdef EOSQ

constant int ntable = 2;
constant real ed_step[] = {0.01f,0.1f}; 
constant real ed_bounds[] ={0.005f,0.455f};
constant int ed_length[] = {46,250};


constant real nb_step[] = {0.01f,0.01f};
constant real nb_bounds[] ={0.0f,0.0f} ;
constant int  nb_length[] ={150,250} ;

#endif




#ifdef NEOSBQS
constant int nelement = 5;
#endif

#if  ((defined NEOSB)|| (defined EOSQ) || (defined NJL_MODEL) ) 
constant int nelement = 3;
#endif


#if ( (defined NEOSB) || (defined NEOSBQS) )
constant int ntable = 7;
constant real ed_step[] = {0.0003f, 0.0006f, 0.001f,0.01f, 0.1f, 1.0f , 10.0f}; 
constant real ed_bounds[] ={0.0f, 0.0036f, 0.015f, 0.045f, 0.455f, 20.355f, 219.355f };
constant int ed_length[] = {13, 20, 31, 42, 200, 200, 200};


constant real nb_step[] = {1e-5, 5e-5, 0.00025, 0.002, 0.01, 0.05, 0.2};
constant real nb_bounds[] ={0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f} ;
constant int  nb_length[] ={500, 300, 180, 250, 350, 250, 200} ;
#endif

#ifdef NJL_MODEL

constant int ntable = 6;
constant real ed_step[] = {0.0003f, 0.0006f, 0.001f,0.01f, 0.1f, 1.0f}; 
constant real ed_bounds[] ={0.0f, 0.0036f, 0.015f, 0.045f, 0.455f, 20.355f };
constant int ed_length[] = {13, 20, 31, 42, 200, 200, 200};


constant real nb_step[] = {1e-5, 5e-5, 0.00025, 0.002, 0.01, 0.05 };
constant real nb_bounds[] ={0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f } ;
constant int  nb_length[] ={500, 300, 180, 250, 350, 250 } ;
#endif


#if ( (((defined NEOSB)|| (defined EOSQ)) || (defined NJL_MODEL) )|| (defined NEOSBQS) )


inline int get_idxfile(real local_ed){  //local_ed:fm^(-4)
    int indx_file=0;
    for(int file_id = 1; file_id < ntable  ;file_id++){
        if(local_ed < (ed_bounds[file_id]/hbarc)){
            indx_file = file_id -1;
            break;
        }
    }

    if(local_ed > (ed_bounds[ntable - 1]/hbarc)){
        indx_file = ntable - 1;
    }

    return indx_file;
}



inline int get_idx0(int idx){
    int idx0 = 0; 
    for(int i = 0; i < idx ; i++ ){
        idx0+= nelement*ed_length[i]*nb_length[i];
    }

    return idx0;

}




inline real interpolate2D(real eps, real nb, __global real* eos_table, int idx, int kind ){
    real e0 = ed_bounds[idx]/hbarc;
    real nb0 = nb_bounds[idx];

    real d_e = ed_step[idx]/hbarc;
    real d_nb = nb_step[idx];

    int N_e  = ed_length[idx];
    int N_nb = nb_length[idx];

    int idx_e  = ( eps - e0 )/d_e; 
    int idx_nb = ( nb  - nb0)/d_nb;

    idx_e  = min(N_e -2 , idx_e);
    idx_nb = min(N_nb -2 , idx_nb);

    idx_e = max(0,idx_e);
    idx_nb = max(0,idx_nb);

    real frac_e = (eps - (idx_e*d_e + e0) )/d_e;
    real frac_nb = (nb - (idx_nb*d_nb +nb0))/d_nb;

    real result;
    
    int idx0 = get_idx0(idx);
    int idx1 = idx0 + kind*nb_length[idx]*ed_length[idx] + idx_nb*ed_length[idx]+idx_e;
    int idx2 = idx0 + kind*nb_length[idx]*ed_length[idx] + idx_nb*ed_length[idx]+idx_e+1;
    int idx3 = idx0 + kind*nb_length[idx]*ed_length[idx] + (idx_nb+1)*ed_length[idx]+idx_e+1;
    int idx4 = idx0 + kind*nb_length[idx]*ed_length[idx] + (idx_nb+1)*ed_length[idx]+idx_e;

                                                             
                                                                
    real temp1 = eos_table[idx1];
    real temp2 = eos_table[idx2];
    real temp3 = eos_table[idx3];
    real temp4 = eos_table[idx4];
                           
    result = ((temp1*(1. - frac_e) + temp2*frac_e)*(1. - frac_nb)
                  + (temp3*frac_e + temp4*(1. - frac_e))*frac_nb);
    return result;
   
   
}
    
inline real eos_P(real eps, real nb, __global real* eos_table){
    //if( eps< EOS_ED_START + EOS_NUM_ED*EOS_ED_STEP \
    //     && nb < EOS_NB_START + EOS_NUM_NB*EOS_NB_STEP){

    {
    int kind = 0;
    eps = eps/hbarc; //fm^{-4}
    int idx_file = get_idxfile(eps);
    real acu1 = 1e-15;
    return max(interpolate2D(eps,fabs(nb),eos_table,idx_file,kind)*hbarc,acu1);
    }
}


inline real eos_T(real eps, real nb, __global real* eos_table){
   //if( eps< EOS_ED_START + EOS_NUM_ED*EOS_ED_STEP \
   //      && nb < EOS_NB_START + EOS_NUM_NB*EOS_NB_STEP){
    {
       int kind = 1;
       eps = eps/hbarc;
       int idx_file = get_idxfile(eps);
       real acu1 = 1e-15;
       return max(interpolate2D(eps,fabs(nb),eos_table,idx_file,kind)*hbarc,acu1);
    }
}


inline real eos_mu(real eps, real nb, __global real* eos_table){
    
     //if( eps< EOS_ED_START + EOS_NUM_ED*EOS_ED_STEP \
         && nb < EOS_NB_START + EOS_NUM_NB*EOS_NB_STEP){
    {
       int kind = 2;
       eps = eps/hbarc;
       int idx_file = get_idxfile(eps);
       real sign = nb/(fabs(nb)+1e-15);
       return sign*interpolate2D(eps,fabs(nb),eos_table,idx_file,kind)*hbarc;

    }
}


#ifdef NEOSBQS
inline real eos_mus(real eps, real nb, __global real* eos_table)
{
       int kind = 3;
       eps = eps/hbarc;
       int idx_file = get_idxfile(eps);
       real sign = nb/(fabs(nb)+1e-15);
       return sign*interpolate2D(eps,fabs(nb),eos_table,idx_file,kind)*hbarc;

}


inline real eos_muc(real eps, real nb, __global real* eos_table)
{
       int kind = 4;
       eps = eps/hbarc;
       int idx_file = get_idxfile(eps);
       real sign = nb/(fabs(nb)+1e-15);
       return sign*interpolate2D(eps,fabs(nb),eos_table,idx_file,kind)*hbarc;

}
#endif

inline real eos_s(real eps, real nb, __global real* eos_table){
    //if( eps< EOS_ED_START + EOS_NUM_ED*EOS_ED_STEP \
         && nb < EOS_NB_START + EOS_NUM_NB*EOS_NB_STEP){
    {
    real pr = eos_P(eps,nb,eos_table)/hbarc;
    real t = eos_T(eps,nb,eos_table)/hbarc;
    real mu = eos_mu(eps,nb,eos_table)/hbarc;
    mu = nb*mu;
#ifdef NEOSBQS

    real mus = eos_mus(eps,nb,eos_table)/hbarc;
    real muc = eos_muc(eps,nb,eos_table)/hbarc;
    real rhos = 0.0;
    real rhoc = 0.4*nb; 
    mu = mu + rhos*mus+rhoc*muc;
#endif 
    

    eps = eps/hbarc;

    real s = (eps + pr - mu)/(t+1e-15); 
    real acu1 =1e-15;
    return max(s,acu1);

    }
}


inline real eos_CS2(real eps, real nb, __global real* eos_table){
    //if( eps< EOS_ED_START + EOS_NUM_ED*EOS_ED_STEP \
    //     && nb < EOS_NB_START + EOS_NUM_NB*EOS_NB_STEP){
    {
    real cs2 = 0.0f;
    eps = eps/hbarc;
    real ed_lo = eps*0.9;
    real ed_hi = eps*1.1;
    int i_elo = get_idxfile(ed_lo);
    int i_ehi = get_idxfile(ed_hi);

    real pr_lo = interpolate2D(ed_lo,nb,eos_table,i_elo,0);
    real pr_hi = interpolate2D(ed_hi,nb,eos_table,i_ehi,0);

    cs2 = (pr_hi-pr_lo) /(ed_hi-ed_lo);

    int i_e0 = get_idxfile(eps);
    real pr = interpolate2D(eps,nb,eos_table,i_e0,0);

    real nb_lo = nb - 0.5*nb_step[i_e0];
    real nb_hi = nb + 0.5*nb_step[i_e0];

    pr_lo = interpolate2D(eps, nb_lo, eos_table,i_e0,0);
    pr_hi = interpolate2D(eps, nb_hi, eos_table,i_e0,0);

    cs2 += (nb/(eps + pr))*( (pr_hi - pr_lo)/(nb_hi - nb_lo  +1e-15) );
    
    real cs2_max = 1.0f/3.0f;
    real cs2_min = 0.01f;
    
    return max(cs2_min,min(cs2,cs2_max));
    }

}


inline real eos_s2ed(real s, real nb, __global real* eos_table){
    
       


       real ed_max = 10000.0f;
       real ed_min = 1e-15f;
       real ed_mid = (ed_max+ed_min)/2.0f;
       
       real s_lo = eos_s(ed_min*hbarc,nb,eos_table);
       if (s < s_lo)
       {
           return ed_min*hbarc;
       }
       real s_hi = eos_s(ed_max*hbarc,nb,eos_table);
       real s_mid;
       int i=0;
       while( ((ed_max-ed_min)>1e-6) && (i <100) &&(  ((ed_max-ed_min)/ed_mid)>1e-12)  ){
     
           s_mid = eos_s(ed_mid*hbarc,nb,eos_table);
           if(s < s_mid){
               ed_max = ed_mid;
           }
           else{
               ed_min = ed_mid;
           }

           ed_mid = (ed_max + ed_min)/2.0f;

           i++;
           
       }       
       
       return ed_mid*hbarc;
    

}
inline real eos_dpde(real ed, real nb, __global real* eos_table){
    real edlo = 0.9*ed;
    real edhi = 1.1*ed;

    real plo = eos_P(edlo,nb,eos_table);
    real phi = eos_P(edhi,nb,eos_table);

    real dpde = (phi-plo)/(edhi-edlo);
    return dpde;
}

inline real eos_dpdn(real ed, real nb, __global real* eos_table){
    
    real nblo = 0.9*nb;
    real nbhi = 1.1*nb;

    real plo = eos_P(ed,nblo,eos_table);
    real phi = eos_P(ed,nbhi,eos_table);

    real dpdnb = (phi-plo)/(nbhi-nblo);
    return dpdnb;
}
#endif










#ifdef CHIRAL

inline real eos_chiralsmall(real eps, real nb, __global real* eos_table,int shift_id, int kind){
    
    real res = 0.0;
    if(eps < 0.0f ){
       return res;
    }	    
    real de = (EDMAXSMALL - EDMINSMALL)/(NESMALL - 1.0f);
    real dn = (NBMAXSMALL - NBMINSMALL)/(NBSMALL - 1.0f);
    
    int ie = (int)((eps - EDMINSMALL)/de);
    int in = (int)((nb  - NBMINSMALL)/dn);

    ie  = min(NESMALL -2 , ie);
    in  = min(NBSMALL -2 , in);
    
    ie  = max(ie,0);
    in  = max(in,0);

    real em = eps - EDMINSMALL - ie*de;
    real nm = nb  - NBMINSMALL - in*dn;

    real we[2] = {1.0f - em/de, em/de};
    real wn[2] = {1.0f - nm/dn, nm/dn};

    //int index_ll = ie*NBSMALL*5 + in*5 + kind;
    //int index_lh = ie*NBSMALL*5 + (in+1)*5 + kind;
    //int index_hl = ie*(NBSMALL+1)*5 + in*5 + kind;
    //int index_hh = ie*(NBSMALL+1)*5 + (in+1)*5 + kind;

    int index_ll = in*NESMALL*5 + ie*5 + kind;
    int index_lh = in*NESMALL*5 + (ie+1)*5 + kind;
    int index_hl = (in+1)*NESMALL*5 + ie*5 + kind;
    int index_hh = (in+1)*NESMALL*5 + (ie+1)*5 + kind;
    res = we[0]*wn[0]*eos_table[index_ll] + we[0]*wn[1]*eos_table[index_hl]
	     + we[1]*wn[0]*eos_table[index_lh] + we[1]*wn[1]*eos_table[index_hh];


    return res;
}



inline real eos_chiralbig(real eps, real nb, __global real* eos_table,int shift_id, int kind){
    
    real res = 0.0;
    if(eps < 0.0f ){
       return res;
    }	    
    real de = (EDMAXBIG - EDMINBIG)/(NEBIG - 1.0f);
    real dn = (NBMAXBIG - NBMINBIG)/(NBBIG - 1.0f);
    
    int ie = (int)((eps - EDMINBIG)/de);
    int in = (int)((nb  - NBMINBIG)/dn);

    ie  = min(NEBIG -2 , ie);
    in  = min(NBBIG -2 , in);
    
    ie  = max(ie,0);
    in  = max(in,0);
    real em = eps - EDMINBIG - ie*de;
    real nm = nb  - NBMINBIG - in*dn;

    real we[2] = {1.0f - em/de, em/de};
    real wn[2] = {1.0f - nm/dn, nm/dn};

    
    int index_ll = in*NEBIG*5 + ie*5 + kind + shift_id;
    int index_lh = in*NEBIG*5 + (ie+1)*5 + kind + shift_id;
    int index_hl = (in+1)*NEBIG*5 + ie*5 + kind + shift_id;
    int index_hh = (in+1)*NEBIG*5 + (ie+1)*5 + kind+ shift_id;
    res = we[0]*wn[0]*eos_table[index_ll] + we[0]*wn[1]*eos_table[index_hl]
	     + we[1]*wn[0]*eos_table[index_lh] + we[1]*wn[1]*eos_table[index_hh];
    return res;
}




inline real eos_P(real eps, real nb, __global real* eos_table){
    
     real pr = 0.0;
     int shift_id = 0;
     nb = fabs(nb);
     if(eps < EDMAXSMALL && nb < NBMAXSMALL)
     {
	 shift_id = 0;
	 pr = eos_chiralsmall(eps,nb,eos_table,shift_id,2 );
     }
     else if (eps< EDMAXBIG && nb < NBMAXBIG)
     {
         shift_id = NESMALL*NBSMALL*5;
	 pr = eos_chiralbig(eps,nb,eos_table,shift_id,2 );
     }
     else{
	 pr = 0.2964*eps;
     }
     real acu1 = 1e-15;
     return  max(pr,acu1);
}

inline real eos_T(real eps, real nb, __global real* eos_table){
     
     real T = 0.0;
     int shift_id = 0;
     nb = fabs(nb);
     if(eps < EDMAXSMALL && nb < NBMAXSMALL)
     {
	 shift_id = 0;
	 T = eos_chiralsmall(eps,nb,eos_table,shift_id,0 );
     }
     else if (eps< EDMAXBIG && nb < NBMAXBIG)
     {
         shift_id = NESMALL*NBSMALL*5;
	 T = eos_chiralbig(eps,nb,eos_table,shift_id,0 );
     }
     else{
	 T = 0.15120476935*pow(eps,0.25f);
     }
     real acu1 = 1e-15;
     
     return  max(T,acu1);
}

inline real eos_mu(real eps, real nb, __global real* eos_table){
     
     real mu = 0.0;
     int shift_id = 0;
     
     real sign = nb/(fabs(nb)+1e-15);
     nb = fabs(nb);
     if(eps < EDMAXSMALL && nb < NBMAXSMALL)
     {
	 shift_id = 0;
	 mu = eos_chiralsmall(eps,nb,eos_table,shift_id,1 );
     }
     else if (eps< EDMAXBIG && nb < NBMAXBIG)
     {
         shift_id = NESMALL*NBSMALL*5;
	 mu = eos_chiralbig(eps,nb,eos_table,shift_id,1 );
     }
     else{
	 mu = 0.0f;
     }
 
     return sign*mu ;
 
}


inline real eos_s(real eps, real nb, __global real* eos_table){
     
     real acu1 = 1e-15;

      real pr  = eos_P(eps,nb,eos_table);
      real mub = eos_mu(eps,nb,eos_table);
      real T   = eos_T(eps,nb,eos_table);
      real s = (eps + pr - mub*nb)/(T);
      return max(acu1,s) ;
 
}


inline real eos_CS2(real eps, real nb, __global real* eos_table){
     
     real cs2  = 0.0;
     int shift_id = 0;
     real ed_lo = eps*0.9;
     real ed_hi = eps*1.1;
     
     real pr_lo = eos_P(ed_lo,nb,eos_table);
     real pr_hi = eos_P(ed_hi,nb,eos_table);
     cs2 = (pr_hi - pr_lo)/(ed_hi - ed_lo);

     real pr = eos_P(eps,nb,eos_table);
     real nb_lo = nb*0.9;
     real nb_hi = nb*1.1;

     pr_lo = eos_P(eps,nb_lo,eos_table);
     pr_hi = eos_P(eps,nb_hi,eos_table);

     cs2 += (nb/(eps+pr))*((pr_hi-pr_lo)/(nb_hi-nb_lo+1e-15));

     real cs2_max = 1.0f/3.0f;
     real cs2_min = 0.01f;

     return max(cs2_min,min(cs2,cs2_max));

}


inline real eos_s2ed(real s, real nb, __global real* eos_table){

       real ed_max = 10000.0f;
       real ed_min = 1e-15f;
       real ed_mid = (ed_max+ed_min)/2.0f;
       
       real s_lo = eos_s(ed_min*hbarc,nb,eos_table);
       if (s < s_lo)
       {
           return ed_min*hbarc;
       }
       real s_hi = eos_s(ed_max*hbarc,nb,eos_table);
       real s_mid;
       int i=0;
       while( ((ed_max-ed_min)>1e-6) && (i <100) &&(  ((ed_max-ed_min)/ed_mid)>1e-12)  ){
     
           s_mid = eos_s(ed_mid*hbarc,nb,eos_table);
           if(s < s_mid){
               ed_max = ed_mid;
           }
           else{
               ed_min = ed_mid;
           }

           ed_mid = (ed_max + ed_min)/2.0f;
           i++;
           
       }       
       
       return ed_mid*hbarc;
    

}



#endif 

#ifdef HARDON_GAS


inline real eos_hardongas(real eps, real nb, __global real* eos_table, int kind){
    
    real res = 0.0;
    if(eps < nb ){
       return res;
    }	    
    real de = EOS_ED_STEP;
    real dn = EOS_NB_STEP;
    
    int ie = (int)((eps - EOS_ED_START)/de);
    int in = (int)((nb  - EOS_NB_START)/dn);
    //int ie = (int)((eps )/de);
    //int in = (int)((nb )/dn);

    ie  = min(EOS_NUM_ED -2 , ie);
    in  = min(EOS_NUM_NB -2 , in);
    
    ie  = max(ie,0);
    in  = max(in,0);
    real em = eps - EOS_ED_START - ie*de;
    real nm = nb  - EOS_NB_START - in*dn;

    real we[2] = {1.0f - em/de, em/de};
    real wn[2] = {1.0f - nm/dn, nm/dn};

    
    int index_ll = ie*EOS_NUM_NB*3 + in*3 + kind ;
    int index_lh = ie*EOS_NUM_NB*3 + (in+1)*3 + kind ;
    int index_hl = (ie+1)*EOS_NUM_NB*3 + in*3 + kind ;
    int index_hh = (ie+1)*EOS_NUM_NB*3 + (in+1)*3 + kind;
    res = we[0]*wn[0]*eos_table[index_ll] + we[0]*wn[1]*eos_table[index_lh]
	     + we[1]*wn[0]*eos_table[index_hl] + we[1]*wn[1]*eos_table[index_hh];

    return res;
}




inline real eos_P(real eps, real nb, __global real* eos_table){
    
     real pr = 0.0;
     nb = fabs(nb);

     if(eps < EOS_ED_START + (EOS_NUM_ED-1)*EOS_ED_STEP &&
        nb  < EOS_NB_START + (EOS_NUM_NB-1)*EOS_NB_STEP)
     {
       pr = eos_hardongas(eps,nb,eos_table,1 );
     }
     else 
     {
      	 pr = 0.0;
     }
     real acu1 = 1e-15;
     return  max(pr,acu1);
}

inline real eos_T(real eps, real nb, __global real* eos_table){
     
     real T = 0.0;
     nb = fabs(nb);

     if(eps < EOS_ED_START + (EOS_NUM_ED-1)*EOS_ED_STEP &&
        nb  < EOS_NB_START + (EOS_NUM_NB-1)*EOS_NB_STEP)
     {
       T = eos_hardongas(eps,nb,eos_table,0 );
     }
     else 
     {
     	 T = 0.0;
     }
     real acu1 = 1e-15;
     
     return  max(T,acu1);
}

inline real eos_mu(real eps, real nb, __global real* eos_table){
     
     real mu = 0.0;
     int shift_id = 0;
     
     real sign = nb/(fabs(nb)+1e-15);
     nb = fabs(nb);
     if(eps < EOS_ED_START + (EOS_NUM_ED-1)*EOS_ED_STEP &&
        nb  < EOS_NB_START + (EOS_NUM_NB-1)*EOS_NB_STEP)
     {
       mu = eos_hardongas(eps,nb,eos_table,2);
     }
     else 
     {
         mu = 0.0;
     }
 
     return sign*mu ;
 
}


inline real eos_s(real eps, real nb, __global real* eos_table){
     
     real acu1 = 1e-15;

      real pr  = eos_P(eps,nb,eos_table);
      real mub = eos_mu(eps,nb,eos_table);
      real T   = eos_T(eps,nb,eos_table);
      real s = (eps + pr - mub*nb)/(T);
      return max(acu1,s) ;
 
}


inline real eos_CS2(real eps, real nb, __global real* eos_table){
     
     real cs2  = 0.0;
     int shift_id = 0;
     real ed_lo = eps*0.9;
     real ed_hi = eps*1.1;
     
     real pr_lo = eos_P(ed_lo,nb,eos_table);
     real pr_hi = eos_P(ed_hi,nb,eos_table);
     cs2 = (pr_hi - pr_lo)/(ed_hi - ed_lo);

     real pr = eos_P(eps,nb,eos_table);
     real nb_lo = nb*0.9;
     real nb_hi = nb*1.1;

     pr_lo = eos_P(eps,nb_lo,eos_table);
     pr_hi = eos_P(eps,nb_hi,eos_table);

     cs2 += (nb/(eps+pr))*((pr_hi-pr_lo)/(nb_hi-nb_lo+1e-15));

     real cs2_max = 1.0f/3.0f;
     real cs2_min = 0.01f;

     return max(cs2_min,min(cs2,cs2_max));

}



inline real eos_s2ed(real s, real nb, __global real* eos_table){
    
       


       real ed_max = 10000.0f*hbarc;
       real ed_min = 1e-15f;
       real ed_mid = (ed_max+ed_min)/2.0f;
       
       real s_lo = eos_s(ed_min,nb,eos_table);
       if (s < s_lo)
       {
           return ed_min;
       }
       real s_hi = eos_s(ed_max,nb,eos_table);
       real s_mid;
       int i=0;
       while( ((ed_max-ed_min)>1e-6) && (i <100) &&(  ((ed_max-ed_min)/ed_mid)>1e-12)  ){
     
           s_mid = eos_s(ed_mid,nb,eos_table);
           if(s < s_mid){
               ed_max = ed_mid;
           }
           else{
               ed_min = ed_mid;
           }

           ed_mid = (ed_max + ed_min)/2.0f;
           i++;
           
       }       
       
       return ed_mid;
    

}



#endif









#endif

