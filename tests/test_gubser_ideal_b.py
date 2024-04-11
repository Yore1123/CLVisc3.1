#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import pyopencl as cl
from pyopencl import array
import os, sys
from time import time
import numpy as np
from scipy.special import hyp2f1
import unittest
import matplotlib.pyplot as plt

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '..'))

from ideal import CLIdeal
from config import cfg


def gubser_ed(tau, r, q):
    return (2.0*q)**(8.0/3.0)/(tau*(1+2*q*q*(tau*tau+r*r)+q**4*(tau*tau-r*r)**2))**(4.0/3.0)

def gubser_vr(tau, r, q):
    return 2.0*q*q*tau*r/(1.0+q*q*tau*tau+q*q*r*r)

class TestGubser():
    def __init__(self):
        self.cfg = cfg
        self.cfg.NZ = 1
        self.cfg.eos_type = 'ideal_gas_baryon' 
        self.ideal = CLIdeal(self.cfg)
        self.ctx = self.ideal.ctx
        self.queue = self.ideal.queue

    def test_gubser(self):
        ''' initialize with gubser energy density in (tau, x, y, eta) coordinates
        to test the gubser expansion:
           eps/eps0 = (tau/tau0)**(-4.0/3.0)
        '''

        kernel_src = """
        # include "real_type.h"
        __kernel void init_ev(global real4 * d_ev1,
                   const real tau,
                   const real q) {
          int i = (int) get_global_id(0);
          int j = (int) get_global_id(1);
          real x = (i-0.5*NX)*DX;
          real y = (j-0.5*NY)*DY;
          real r = sqrt(x*x + y*y);
          real ed = pow(2.0f*q, 8.0f/3.0f) / pow((1.0f + 2.0f*q*q*(tau*tau+r*r) + 
                    q*q*q*q*(tau*tau-r*r)*(tau*tau-r*r))*tau, 4.0f/3.0f);

          real vx = 2.0f*q*q*tau*x/(1.0f + q*q*tau*tau + q*q*r*r);
          real vy = 2.0f*q*q*tau*y/(1.0f + q*q*tau*tau + q*q*r*r);

          for ( int k = 0; k < NZ; k ++ ) {
             int gid = i*NY*NZ + j*NZ + k;
             d_ev1[gid] = (real4)(ed, vx, vy, 0.0);
             //d_ev1[gid] = (real4)(20.0, 0.0, 0.0, 0.0);
          }
        }
        """
        cwd, cwf = os.path.split(__file__)
    
        q = 0.25
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        compile_options = list(self.ideal.compile_options)
        compile_options.append('-I %s'%os.path.join(cwd, '..', 'kernel'))
        compile_options.append('-D USE_SINGLE_PRECISION')
        prg = cl.Program(self.ctx, kernel_src).build(compile_options)
        prg.init_ev(self.queue, (self.cfg.NX, self.cfg.NY), None,
                    self.ideal.d_ev[1], self.cfg.real(self.cfg.TAU0),
                    self.cfg.real(q)).wait()

        max_loops = 1 
        for n in range(max_loops):
            self.ideal.edmax = self.ideal.max_energy_density()
            print ("tau = %s"%self.ideal.tau, "Emax = %s"%self.ideal.edmax)
            self.ideal.stepUpdate(step=1)
            self.ideal.tau = self.cfg.real(self.cfg.TAU0 + (n+1)*self.cfg.DT)
            self.ideal.stepUpdate(step=2)
            if n%1 == 0:
                cl.enqueue_copy(self.queue, self.ideal.h_ev1, self.ideal.d_ev[1]).wait()
                edr = self.ideal.h_ev1[:,0].reshape(NX, NY, NZ)[:, NY//2, NZ//2]
                vr = self.ideal.h_ev1[:,1].reshape(NX, NY, NZ)[:, NY//2, NZ//2]

                x = np.linspace(-NX//2*self.cfg.DX, NX//2*self.cfg.DX, NX)

                a = gubser_ed(self.ideal.tau, x, q)
                b = edr[:]
                c = gubser_vr(self.ideal.tau, x, q)
                d = vr[:]

                plt.semilogy(x, a, 'r-')
                plt.semilogy(x, b, 'b--')
                #plt.plot(x, c, 'r-', label='analy')
                #plt.plot(x, d, 'b--', label='CLVisc')
        plt.xlabel(r'$r\ [fm]$', fontsize=30)
        #plt.ylabel(r'$v_{\perp}$', fontsize=30)
        plt.title(r'Gubser solution test', fontsize=30)
        plt.ylabel('ed', fontsize=30)
        plt.subplots_adjust(left=0.13, bottom=0.13)
        plt.legend(loc='best')
        plt.show()
    

if __name__ == '__main__':
    g=TestGubser()
    g.test_gubser()
