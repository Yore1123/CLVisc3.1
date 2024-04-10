#include<helper.h>



#define SIGMA_R2 (jet_r_gw*jet_r_gw)
#define SIGMA_ETAS2 (jet_eta_gw*jet_eta_gw)


real linear_int(real x0,real x1,real y0, real y1, real x)
{
  real temp=0.0;
  temp=((x-x0)*y1+(x1-x)*y0)/(x1-x0);
  return temp;
}


//inspired by numerical recipes
//x0,x1: grid points in x-direction
//y0,y1: grid points in y-direction
//f0-f3: function value starting at x0,y0, continue counterclockwise
//put differently: f0=f(x0,y0)
//f1=f(x1,y0)
//f2=f(x1,y1)
//f3=f(x0,y1)
real bilinear_int(real x0,real x1,real y0, real y1, real f0, real f1, real f2, real f3, real x, real y)
{
  real temp=0.0;
  real t=(x-x0)/(x1-x0);
  real u=(y-y0)/(y1-y0);

  if ((isfinite(u)==1)&&(isfinite(t)==1))
    {
      temp=(1-t)*(1-u)*f0+t*(1-u)*f1+t*u*f2+(1-t)*u*f3;
    }
  else
    {
      if (isfinite(u)==0)
        temp=linear_int(x0,x1,f0,f2,x);
      if (isfinite(t)==0)
        temp=linear_int(y0,y1,f0,f2,y);
    }

    return temp;
}


real trilinear_int(real x0,real x1,real y0, real y1, real z0, real z1,real f000, real f001, real f010, real f011, real f100, real f101,real f110, real f111, real x, real y,real z)
{
  real temp=0.0;
  real t=(x-x0)/(x1-x0);
  real u=(y-y0)/(y1-y0);
  //real v=(z-y0)/(z1-z0);
  real v=(z-z0)/(z1-z0);

  if ((isfinite(u)==1)&&(isfinite(t)==1)&&(isfinite(v)==1))
    {
      temp=(1-t)*(1-u)*(1-v)*f000;
      temp+=(1-t)*(1-u)*v*f001;
      temp+=(1-t)*u*(1-v)*f010;
      temp+=(1-t)*u*v*f011;
      temp+=t*(1-u)*(1-v)*f100;
      temp+=t*(1-u)*v*f101;
      temp+=t*u*(1-v)*f110;
      temp+=t*u*v*f111;
    }
  else
    {
      if (isfinite(t)==0)
        temp=bilinear_int(y0,y1,z0,z1,f000,f010,f011,f001,y,z);
      if (isfinite(v)==0)
        temp=bilinear_int(x0,x1,y0,y1,f000,f100,f110,f010,x,y);
      if (isfinite(u)==0)
        temp=bilinear_int(x0,x1,z0,z1,f000,f100,f101,f001,x,z);

    }

  return temp;
}





/* \nabla T^{\mu\nu} = J^{\nu}
where $J^{\nu} is the energy momentum deposition from jet
energy loss;
*/
__kernel void jetsource(__global real4 * d_x,
			__global real4 * d_p,
			__global real4 * d_jetsrc,
			const real tau,
			const uint size) {
    int I = get_global_id(0);
    int J = get_global_id(1);
    int K = get_global_id(2);
    real x = (I - (NX-1)/2) * DX;
    real y = (J - (NY-1)/2) * DY;
    real z = (K - (NZ-1)/2) * DZ;

    int IND = I*NY*NZ+J*NZ+K;
    d_jetsrc[IND]= (real4)(0.0f , 0.0f , 0.0f , 0.0f);	
    real4 dEdX = (real4) (0.0f , 0.0f , 0.0f , 0.0f);
    real4 coef = {tau, tau, tau, tau*tau};

    real norm = 1.0f / (pow(sqrt(2.0f*M_PI_F), 3.0f) * SIGMA_R2 * sqrt(SIGMA_ETAS2) * tau);

    real4 dxn;
    real4 dpn;
    for(int n=0;n!=size;n++)
    {
		dxn = d_x[n];
		dpn = d_p[n];
                // printf("DPN=%d/n",dpn);
		real distx2 = -max((x-dxn.s0)*(x-dxn.s0),0.0000001f);
		real disty2 = -max((y-dxn.s1)*(y-dxn.s1),0.0000001f);
		real distz2 = -max((z-dxn.s2)*(z-dxn.s2),0.0000001f);
		real delta=exp((distx2+disty2)/(2.0f*SIGMA_R2)+distz2/(2.0f*SIGMA_ETAS2));	
		real dt=DT;
		dEdX += dpn*delta/dt;		
    }	
    d_jetsrc[IND] = coef*norm*dEdX;
}




__kernel void get_TV(__global real4 *d_x,
		    __global real4  *d_ev,
                    __global real *d_nb,
		    __global real4 *d_TVf,
                    __global real* eos_table,
                    const uint num)


{
    for(int i = get_global_id(0); i < num ; i += BSZ)
    {
    
    

	real x=d_x[i].s0;
	real y=d_x[i].s1;
	real z=d_x[i].s2;

	int lowNx=floor(x/DX)+(NX-1)/2;
	int lowNy=floor(y/DY)+(NY-1)/2;
	int lowNz=floor(z/DZ)+(NZ-1)/2;//wenbin 
	
	if(lowNx>NX-2||lowNx<2||lowNy>NY-2||lowNy<2||lowNz>NZ-2||lowNz<2){
		d_TVf[i] = (real4)(0.0,0.0,0.0,0.0);	
	}
	else{
		real lowx=(lowNx-(NX-1)/2)*DX;
		real highx=lowx+DX;
		real lowy=(lowNy-(NY-1)/2)*DY;
		real highy=lowy+DY;
		real lowz=(lowNz-(NZ-1)/2)*DZ;//wenbin
		real highz=lowz+DZ;

		int IND;
		IND=lowNx*NY*NZ+lowNy*NZ+lowNz;
                real4 ev000 =  d_ev[IND];
                real nb000 = d_nb[IND];

		IND=lowNx*NY*NZ+lowNy*NZ+(lowNz+1);	
                real4 ev001 =  d_ev[IND];
                real nb001 = d_nb[IND];
		
		IND=lowNx*NY*NZ+(lowNy+1)*NZ+lowNz;
                real4 ev010 =  d_ev[IND];
                real nb010 = d_nb[IND];

		IND=lowNx*NY*NZ+(lowNy+1)*NZ+(lowNz+1);	
		real4 ev011 =  d_ev[IND];
                real nb011 = d_nb[IND];

		IND=(lowNx+1)*NY*NZ+lowNy*NZ+lowNz;
		real4 ev100 =  d_ev[IND];
                real nb100 = d_nb[IND];

		IND=(lowNx+1)*NY*NZ+lowNy*NZ+(lowNz+1);
		real4 ev101 =  d_ev[IND];
                real nb101 = d_nb[IND];

		IND=(lowNx+1)*NY*NZ+(lowNy+1)*NZ+lowNz;
                real4 ev110 =  d_ev[IND];
                real nb110 = d_nb[IND];

		IND=(lowNx+1)*NY*NZ+(lowNy+1)*NZ+(lowNz+1);
		real4 ev111 =  d_ev[IND];
                real nb111 = d_nb[IND];


		real ed = trilinear_int(lowx,highx,lowy,highy,lowz,highz,ev000.s0,ev001.s0,ev010.s0,ev011.s0,ev100.s0,ev101.s0,ev110.s0,ev111.s0,x,y,z);
                real Vx = trilinear_int(lowx,highx,lowy,highy,lowz,highz,ev000.s1,ev001.s1,ev010.s1,ev011.s1,ev100.s1,ev101.s1,ev110.s1,ev111.s1,x,y,z);
                real Vy = trilinear_int(lowx,highx,lowy,highy,lowz,highz,ev000.s2,ev001.s2,ev010.s2,ev011.s2,ev100.s2,ev101.s2,ev110.s2,ev111.s2,x,y,z);
                real Vz = trilinear_int(lowx,highx,lowy,highy,lowz,highz,ev000.s3,ev001.s3,ev010.s3,ev011.s3,ev100.s3,ev101.s3,ev110.s3,ev111.s3,x,y,z);
        
                real nb = trilinear_int(lowx,highx,lowy,highy,lowz,highz,nb000,nb001,nb010,nb011,nb100,nb101,nb110,nb111,x,y,z);
                d_TVf[i]=(real4)( eos_T(ed, nb , eos_table), Vx,Vy,Vz); 


	}
    }
}


__kernel void update_src(__global real4 *d_jetsrc,
			__global real4 *d_src){
	int i=get_global_id(0);
	int j=get_global_id(1);
	int k=get_global_id(2);

	int IND=i*NY*NZ+j*NZ+k;
	d_src[IND]=d_src[IND]+d_jetsrc[IND];
}			  
