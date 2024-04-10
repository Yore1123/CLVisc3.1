#/usr/bin/env python
#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>

from __future__ import absolute_import, division, print_function
import numpy as np
import pyopencl as cl
from pyopencl import array
import os
import sys
from time import time


def roundUp(value, multiple):
    '''This function rounds one integer up to the nearest multiple of another integer,
    to get the global work size (which are multiples of local work size) from NX, NY, NZ.  '''
    remainder = value % multiple
    if remainder != 0:
        value += multiple - remainder
    return value


class SmearingP4X4(object):
    '''The pyopencl version for gaussian smearing ini condition'''
    def __init__(self, cfg, ctx, queue, compile_options, h_ev, h_nb, d_ev1, d_nb1,
        particle_list, eos_table, model = "smash",NEVENT=1):
        '''initialize d_ev1 with partons p4x4, which is one size*8 np.array '''
        self.cwd, cwf = os.path.split(__file__)
        self.compile_options = compile_options
        self.nevent = NEVENT
        self.__loadAndBuildCLPrg(ctx, cfg,model)
        size = cfg.NX*cfg.NY*cfg.NZ
        if model.upper() == "SMASH":
            nhadron = len(particle_list[:,0])
            h_hadron = np.zeros((nhadron*10),cfg.real)
            h_hadron[:] = particle_list.flatten().astype(cfg.real)
            d_hadron = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=h_hadron.nbytes)
            cl.enqueue_copy(queue, d_hadron, h_hadron).wait()

            h_norm = np.zeros((nhadron),cfg.real)
            d_norm = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=h_norm.nbytes)
            
            h_norm2 = np.zeros((nhadron),cfg.real)
            d_norm2 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=h_norm2.nbytes)
            
            cl.enqueue_copy(queue, d_norm, h_norm).wait()
            cl.enqueue_copy(queue, d_norm2, h_norm2).wait()
            NX5 = roundUp(cfg.NX, 5)
            NY5 = roundUp(cfg.NY, 5)
            NZ5 = roundUp(cfg.NZ, 5)
        
            self.prg.smearing_baryon_norm(queue, (NX5, NY5, NZ5), (5,5,5),
                    d_hadron, np.int32(nhadron),
                    np.int32(size),d_norm,d_norm2).wait()
            cl.enqueue_copy(queue,h_norm,d_norm).wait()
            self.prg.smearing_baryon(queue, (NX5, NY5, NZ5), (5,5,5),
                    d_ev1, d_nb1, d_hadron, eos_table, np.int32(nhadron),
                    np.int32(size),d_norm,d_norm2).wait()
            cl.enqueue_copy(queue,h_nb,d_nb1).wait() 
            cl.enqueue_copy(queue,h_ev,d_ev1).wait() 
            
            print( "check: ", np.sum(h_ev[:,0])*cfg.TAU0*cfg.DX*cfg.DY*cfg.DZ, nhadron, nhadron*( cfg.TAU0*cfg.DX*cfg.DY*cfg.DZ ) ) 
            print(np.sum(h_nb)*cfg.TAU0*cfg.DX*cfg.DY*cfg.DZ, nhadron, nhadron*( cfg.TAU0*cfg.DX*cfg.DY*cfg.DZ ) ) 
        if model.upper() == "AMPT":
            nparton = len(particle_list[:,0])
            h_parton = np.zeros((nparton*9),cfg.real)
            h_parton[:] = particle_list.flatten().astype(cfg.real)
            d_parton = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=h_parton.nbytes)
            cl.enqueue_copy(queue, d_parton, h_parton).wait()
 
            h_norm = np.zeros((nparton),cfg.real)
            d_norm = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=h_norm.nbytes)
            h_norm2 = np.zeros((nparton),cfg.real)
            d_norm2 = cl.Buffer(ctx, cl.mem_flags.READ_ONLY, size=h_norm2.nbytes)
            cl.enqueue_copy(queue, d_norm, h_norm).wait()
            cl.enqueue_copy(queue, d_norm2, h_norm2).wait()
            NX5 = roundUp(cfg.NX, 5)
            NY5 = roundUp(cfg.NY, 5)
            NZ5 = roundUp(cfg.NZ, 5)

            self.prg.smearing_baryon_norm(queue, (NX5, NY5, NZ5), (5,5,5),
                    d_parton, np.int32(nparton),
                    np.int32(size),d_norm,d_norm2).wait()
            cl.enqueue_copy(queue,h_norm,d_norm).wait()
            cl.enqueue_copy(queue,h_norm2,d_norm2).wait()
            
            self.prg.smearing_baryon(queue, (NX5, NY5, NZ5), (5,5,5),
                    d_ev1, d_nb1, d_parton, eos_table, np.int32(nparton),
                    np.int32(size),d_norm,d_norm2).wait()
            
            cl.enqueue_copy(queue,h_nb,d_nb1).wait() 
            cl.enqueue_copy(queue,h_ev,d_ev1).wait() 
            print (h_norm)
            print (h_norm2)

            print( "check: ", np.sum(h_ev[:,0])*cfg.TAU0*cfg.DX*cfg.DY*cfg.DZ, nparton, nparton*( cfg.TAU0*cfg.DX*cfg.DY*cfg.DZ ) ) 
            print(np.sum(h_nb)*cfg.TAU0*cfg.DX*cfg.DY*cfg.DZ, nparton, nparton*( cfg.TAU0*cfg.DX*cfg.DY*cfg.DZ ) ) 
            

    def __loadAndBuildCLPrg(self, ctx, cfg, model):
        #load and build *.cl programs with compile self.compile_options
        smearing_options = list(self.compile_options)
        smearing_options.append('-D {key}={value}f'.format(key='SQRTS', value=cfg.SQRTS))
        smearing_options.append('-D {key}={value}f'.format(key='SIGR', value=cfg.r_gw))
        smearing_options.append('-D {key}={value}f'.format(key='SIGZ', value=cfg.Eta_gw))
        smearing_options.append('-D {key}={value}f'.format(key='NSIGR', value=cfg.r_gw))
        smearing_options.append('-D {key}={value}f'.format(key='NSIGZ', value=cfg.Eta_gw))
        smearing_options.append('-D {key}={value}f'.format(key='KFACTOR', value=cfg.KFACTOR))
        smearing_options.append('-D {key}={value}'.format(key='NEVENT', value=self.nevent))
        if model.upper() == "SMASH":
            smearing_options.append('-D SMASH')
        if model.upper() == "AMPT":
            smearing_options.append('-D AMPT')
        
        with open(os.path.join(self.cwd, 'kernel', 'kernel_gaussian_smearing_new.cl'), 'r') as f:
            prg_src = f.read()
            self.prg = cl.Program(ctx, prg_src).build(options=' '.join(smearing_options))


