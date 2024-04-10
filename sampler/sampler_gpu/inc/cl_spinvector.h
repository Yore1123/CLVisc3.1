#ifndef __CL_SPINVECTOR__
#define __CL_SPINVECTOR__


inline real4 get_spin_vector_th(real8 SF, 
                                real8 ptc1,
                                __global real* d_omega_th,
                                const int ID,
                                const real mass,
                                const real gspin,
                                const real fermi_boson,
                                const real tfrz,
                                const real chem)
         {


            real4 mom_txyz = ptc1.s0123;
            real4 pos_txyz = ptc1.s4567;
            
            real px = mom_txyz.s1;
            real py = mom_txyz.s2;
            real mt = sqrt(mass*mass + px*px + py*py);
            real rapidity = atanh(mom_txyz.s3/mom_txyz.s0);
            real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;
            
            omega_tx = d_omega_th[6*ID + 0];
            omega_ty = d_omega_th[6*ID + 1];
            omega_tz = d_omega_th[6*ID + 2];
            omega_xy = d_omega_th[6*ID + 3];
            omega_xz = d_omega_th[6*ID + 4];
            omega_yz = d_omega_th[6*ID + 5];

            real4 umu = (real4)(1.0f, SF.s4, SF.s5, SF.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));

            real etas = SF.s7;

            real Y = rapidity;
            real mtcosh = mt * cosh(Y - etas);
            real mtsinh = mt * sinh(Y - etas);
            real4 pmu = (real4)(mtcosh, -px, -py, -mtsinh); //p_mu
      
            real beta = 1.0/tfrz;
            real pdotu = dot(umu, pmu);
            real tmp = (pdotu - chem )*beta;
            real feq = 1.0/(exp(tmp) + 1);

        
            real pomega_0 =  omega_tx * pmu.s1 + omega_ty * pmu.s2 + omega_tz * pmu.s3;
            real pomega_1 = -omega_tx * pmu.s0 + omega_xy * pmu.s2 + omega_xz * pmu.s3;
            real pomega_2 = -omega_ty * pmu.s0 - omega_xy * pmu.s1 + omega_yz * pmu.s3;
            real pomega_3 = -omega_tz * pmu.s0 - omega_xz * pmu.s1 - omega_yz * pmu.s2;
            real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);

            real normfactor =  4.0f*mass;
            real4 local_spin =  pomega * (1.0f - feq);

            real pol0 = local_spin.s0;
            real pol3 = local_spin.s3;
            local_spin.s0 = cosh(etas)*pol0 + sinh(etas)*pol3;
            local_spin.s3 = sinh(etas)*pol0 + cosh(etas)*pol3;

            local_spin = local_spin/normfactor;
            // if (fabs(local_spin.s1)<0.001)
            // {
            //    printf("%d %.12f %.12f %.12f %.12f  %.12f %.12f %.12f %.12f \n",ID,omega_tx,omega_ty,omega_tz,omega_xy,mom_txyz.s0,mom_txyz.s1,mom_txyz.s2,mom_txyz.s3);
            // }
            return local_spin;
            

         };


inline real4 get_spin_vector_shear(real8 SF,
                                   real8 ptc1,
                                   __global real* d_omega_shear1,
                                   __global real* d_omega_shear2,
                                   const int ID,
                                   const real mass,
                                   const real gspin,
                                   const real fermi_boson,
                                   const real tfrz,
                                   const real chem)
         {

            

            real4 mom_txyz = ptc1.s0123;
            real4 pos_txyz = ptc1.s4567;
            
            real px = mom_txyz.s1;
            real py = mom_txyz.s2;
            real mt = sqrt(mass*mass + px*px + py*py);
            real rapidity = atanh(mom_txyz.s3/mom_txyz.s0);
            real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;
            
       
            real4 umu = (real4)(1.0f, SF.s4, SF.s5, SF.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        

            real4 umu1 = (real4)(1.0f, -SF.s4, -SF.s5, -SF.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        

            real etas = SF.s7;

            real Y = rapidity;
            real mtcosh = mt * cosh(Y - etas);
            real mtsinh = mt * sinh(Y - etas);
            

            real4 pmu1 = (real4)(mtcosh, -px, -py, -mtsinh); //p_mu
            real4 pmu2 = (real4)(mtcosh, px, py, mtsinh); //p^mu


            real4 dudx_sym[4] = { (real4)( d_omega_shear1[ID*16 + 0]*2.0f, 
            d_omega_shear1[ID*16 +4] + d_omega_shear1[ID*16 +1],
            d_omega_shear1[ID*16 +8] + d_omega_shear1[ID*16 +2],
            d_omega_shear1[ID*16 +12] + d_omega_shear1[ID*16 +3]),
            (real4)( d_omega_shear1[ID*16 +1] + d_omega_shear1[ID*16 +4] , 
            d_omega_shear1[ID*16 +5] *2.0f,
            d_omega_shear1[ID*16 +9] + d_omega_shear1[ID*16 +6],
            d_omega_shear1[ID*16 +13] + d_omega_shear1[ID*16 +7]),
            (real4)( d_omega_shear1[ID*16 + 2] + d_omega_shear1[ID*16 +8] , 
            d_omega_shear1[ID*16 +6] + d_omega_shear1[ID*16 +9] ,
            d_omega_shear1[ID*16 +10] * 2.0f,
            d_omega_shear1[ID*16 +14] + d_omega_shear1[ID*16 +11]),
            (real4)( d_omega_shear1[3] + d_omega_shear1[ID*16 +12] , 
            d_omega_shear1[ID*16 +7] + d_omega_shear1[ID*16 +13] ,
            d_omega_shear1[ID*16 +11] + d_omega_shear1[ID*16 +14] ,
            d_omega_shear1[ID*16 +15] *2.0f)
            };


            real pdotu = dot(umu, pmu1);
            real4 omega_tep = (real4) (0.0f, 0.0f, 0.0f, 0.0f);
            omega_tep.s0 = dot(pmu2,dudx_sym[0]) - pdotu*d_omega_shear2[ID*4 + 0];
            omega_tep.s1 = dot(pmu2,dudx_sym[1]) - pdotu*d_omega_shear2[ID*4 + 1];
            omega_tep.s2 = dot(pmu2,dudx_sym[2]) - pdotu*d_omega_shear2[ID*4 + 2];
            omega_tep.s3 = dot(pmu2,dudx_sym[3]) - pdotu*d_omega_shear2[ID*4 + 3];


            omega_tx = pmu1.s2*umu1.s3 - pmu1.s3*umu1.s2;
            omega_ty = pmu1.s3*umu1.s1 - pmu1.s1*umu1.s3;
            omega_tz = pmu1.s1*umu1.s2 - pmu1.s2*umu1.s1;
            omega_xy = pmu1.s0*umu1.s3 - pmu1.s3*umu1.s0;
            omega_xz = pmu1.s2*umu1.s0 - pmu1.s0*umu1.s2;
            omega_yz = pmu1.s0*umu1.s1 - pmu1.s1*umu1.s0;

      
            real beta = 1.0/tfrz;
            real tmp = (pdotu - chem )*beta;
            real feq = 1.0/(exp(tmp) + 1);

        
            real pomega_0 =  omega_tx * omega_tep.s1 + omega_ty * omega_tep.s2 + omega_tz * omega_tep.s3;
            real pomega_1 = -omega_tx * omega_tep.s0 + omega_xy * omega_tep.s2 + omega_xz * omega_tep.s3;
            real pomega_2 = -omega_ty * omega_tep.s0 - omega_xy * omega_tep.s1 + omega_yz * omega_tep.s3;
            real pomega_3 = -omega_tz * omega_tep.s0 - omega_xz * omega_tep.s1 - omega_yz * omega_tep.s2;

            real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);

            real normfactor =  4.0f*mass;
            
            real4 local_spin = -hbarc * pomega *(1.0f - feq)/(pdotu*tfrz);
            //real4 local_spin = - pomega *(1.0f - feq)/(pdotu*tfrz);

            real pol0 = local_spin.s0;
            real pol3 = local_spin.s3;
            local_spin.s0 = cosh(etas)*pol0 + sinh(etas)*pol3;
            local_spin.s3 = sinh(etas)*pol0 + cosh(etas)*pol3;

            local_spin = local_spin/normfactor;
            // if (fabs(local_spin.s1)<0.001)
            // {
            //    printf("%d %.12f %.12f %.12f %.12f  %.12f %.12f %.12f %.12f \n",ID,omega_tx,omega_ty,omega_tz,omega_xy,mom_txyz.s0,mom_txyz.s1,mom_txyz.s2,mom_txyz.s3);
            // }
            return local_spin;
            

         };


inline real4 get_spin_vector_accT(real8 SF,
                                  real8 ptc1,
                                  __global real* d_omega_accT,
                                  const int ID,
                                  const real mass,
                                  const real gspin,
                                  const real fermi_boson,
                                  const real tfrz,
                                  const real chem)
         {

            
            real4 mom_txyz = ptc1.s0123;
            real4 pos_txyz = ptc1.s4567;
            
            real px = mom_txyz.s1;
            real py = mom_txyz.s2;
            real mt = sqrt(mass*mass + px*px + py*py);
            real rapidity = atanh(mom_txyz.s3/mom_txyz.s0);
            real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;
            
       
            real4 umu = (real4)(1.0f, SF.s4, SF.s5, SF.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        

            real4 umu1 = (real4)(1.0f, -SF.s4, -SF.s5, -SF.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        

            real etas = SF.s7;

            real Y = rapidity;
            real mtcosh = mt * cosh(Y - etas);
            real mtsinh = mt * sinh(Y - etas);
            

            real4 pmu1 = (real4)(mtcosh, -px, -py, -mtsinh); //p_mu
            real4 pmu2 = (real4)(mtcosh, px, py, mtsinh); //p^mu


            omega_tx = d_omega_accT[ID*6 + 0];
            omega_ty = d_omega_accT[ID*6 + 1];
            omega_tz = d_omega_accT[ID*6 + 2];
            omega_xy = d_omega_accT[ID*6 + 3];
            omega_xz = d_omega_accT[ID*6 + 4];
            omega_yz = d_omega_accT[ID*6 + 5];


            real pdotu = dot(umu, pmu1);

      
            real beta = 1.0/tfrz;
            real tmp = (pdotu - chem )*beta;
            real feq = 1.0/(exp(tmp) + 1);

        
            real pomega_0 =  omega_tx * pmu1.s1 + omega_ty * pmu1.s2 + omega_tz * pmu1.s3;
            real pomega_1 = -omega_tx * pmu1.s0 + omega_xy * pmu1.s2 + omega_xz * pmu1.s3;
            real pomega_2 = -omega_ty * pmu1.s0 - omega_xy * pmu1.s1 + omega_yz * pmu1.s3;
            real pomega_3 = -omega_tz * pmu1.s0 - omega_xz * pmu1.s1 - omega_yz * pmu1.s2;

            real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);

            real normfactor =  4.0f*mass;
            
            real4 local_spin = -hbarc * pomega * (1.0f - feq)/(tfrz);
            //real4 local_spin = - pomega *(1.0f - feq)/(pdotu*tfrz);

            real pol0 = local_spin.s0;
            real pol3 = local_spin.s3;
            local_spin.s0 = cosh(etas)*pol0 + sinh(etas)*pol3;
            local_spin.s3 = sinh(etas)*pol0 + cosh(etas)*pol3;

            local_spin = local_spin/normfactor;
            // if (fabs(local_spin.s1)<0.001)
            // {
            //    printf("%d %.12f %.12f %.12f %.12f  %.12f %.12f %.12f %.12f \n",ID,omega_tx,omega_ty,omega_tz,omega_xy,mom_txyz.s0,mom_txyz.s1,mom_txyz.s2,mom_txyz.s3);
            // }
            return local_spin;
            

         };



   inline real4 get_spin_vector_chemical(real8 SF,
                                         real8 ptc1,
                                         __global real* d_omega_chemical,
                                         const int ID,
                                         const real mass,
                                         const real gspin,
                                         const real fermi_boson,
                                         const real tfrz,
                                         const real chem)
         {

            

            real4 mom_txyz = ptc1.s0123;
            real4 pos_txyz = ptc1.s4567;
            
            real px = mom_txyz.s1;
            real py = mom_txyz.s2;
            real mt = sqrt(mass*mass + px*px + py*py);
            real rapidity = atanh(mom_txyz.s3/mom_txyz.s0);
            real omega_tx, omega_ty, omega_tz, omega_xy, omega_xz, omega_yz;
            
       
            real4 umu = (real4)(1.0f, SF.s4, SF.s5, SF.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        

            real4 umu1 = (real4)(1.0f, -SF.s4, -SF.s5, -SF.s6) * \
            1.0f/sqrt(max((real)1.0E-10f, \
            (real)(1.0f-SF.s4*SF.s4 - SF.s5*SF.s5 - SF.s6*SF.s6)));
        

            real etas = SF.s7;

            real Y = rapidity;
            real mtcosh = mt * cosh(Y - etas);
            real mtsinh = mt * sinh(Y - etas);
            

            real4 pmu1 = (real4)(mtcosh, -px, -py, -mtsinh); //p_mu
            real4 pmu2 = (real4)(mtcosh, px, py, mtsinh); //p^mu


            omega_tx = d_omega_chemical[ID*6 + 0];
            omega_ty = d_omega_chemical[ID*6 + 1];
            omega_tz = d_omega_chemical[ID*6 + 2];
            omega_xy = d_omega_chemical[ID*6 + 3];
            omega_xz = d_omega_chemical[ID*6 + 4];
            omega_yz = d_omega_chemical[ID*6 + 5];


            real pdotu = dot(umu, pmu1);

      
            real beta = 1.0/tfrz;
            real tmp = (pdotu - chem )*beta;
            real feq = 1.0/(exp(tmp) + 1);

        
            real pomega_0 =  omega_tx * pmu1.s1 + omega_ty * pmu1.s2 + omega_tz * pmu1.s3;
            real pomega_1 = -omega_tx * pmu1.s0 + omega_xy * pmu1.s2 + omega_xz * pmu1.s3;
            real pomega_2 = -omega_ty * pmu1.s0 - omega_xy * pmu1.s1 + omega_yz * pmu1.s3;
            real pomega_3 = -omega_tz * pmu1.s0 - omega_xz * pmu1.s1 - omega_yz * pmu1.s2;

            real4 pomega = (real4)(pomega_0, pomega_1, pomega_2, pomega_3);

            real normfactor =  4.0f*mass;
            
            real4 local_spin = hbarc * pomega  * (1.0f - feq)/(pdotu);
            

            real pol0 = local_spin.s0;
            real pol3 = local_spin.s3;
            local_spin.s0 = cosh(etas)*pol0 + sinh(etas)*pol3;
            local_spin.s3 = sinh(etas)*pol0 + cosh(etas)*pol3;

            local_spin = local_spin/normfactor;

            //if (fabs(local_spin.s1)<0.001)
            // {
            //    printf("%d %.12f %.12f %.12f %.12f  %.12f %.12f %.12f %.12f \n",ID,omega_tx,omega_ty,omega_tz,omega_xy,mom_txyz.s0,mom_txyz.s1,mom_txyz.s2,mom_txyz.s3);
            // }
           
            return local_spin;
            

         };






#endif
