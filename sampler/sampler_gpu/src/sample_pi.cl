#include "tyche_i.cl"

__kernel void sample_pi(const uint iters, 
			__global ulong* seed,
			__global uint* res){

	uint gid=get_global_id(0);
    tyche_i_state state;
    tyche_i_seed(&state,seed[gid]);
    uint cnt=0;
    for(uint i=0;i<iters;i++){
        float a=tyche_i_float(state);
        float b=tyche_i_float(state);
        if(a * a + b * b < 1.f){
            cnt++;
        }
    }
    res[gid]=cnt;
    
}
