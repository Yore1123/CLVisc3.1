#ifndef __DISTRIBUTION__
#define __DISTRIBUTION__

#include <helper.h>

inline real juttner_distribution_func(real momentum_radial,real mass,
                                      real tfrz, real muB, real lam){

    real f = 1.0/(exp( (sqrt(momentum_radial*momentum_radial+mass*mass) - muB)/tfrz )+lam );
    
    return max(f,acu);

}

inline real poisson_distribution_func(real mean, int n){
    real nn =  n*1.0f;
    real f = pown(mean,n)*exp(-mean)/(tgamma(nn+1.0)*1.0);
    return f;
}





#endif