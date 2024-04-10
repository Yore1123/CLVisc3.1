#/usr/bin/env python
#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>


from __future__ import absolute_import, division, print_function
import numpy as np
import pyopencl as cl
from pyopencl import array
import pyopencl.array as cl_array
from pyopencl.array import Array

import os
import sys
from time import time
from math import floor

cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)
from eos.eos import Eos
import h5py


class BulkInfo(object):
    '''The bulk information like:
       ed(x), ed(y), ed(eta), T(x), T(y), T(eta)
       vx, vy, veta, ecc_x, ecc_p'''
    def __init__(self, cfg, ctx, queue, eos_table, compile_options):
        self.cfg = cfg
        self.ctx = ctx
        self.queue = queue
        self.eos_table = eos_table
        self.compile_options = list(compile_options)

        NX, NY, NZ = cfg.NX, cfg.NY, cfg.NZ
        if NX%2 == 1:
            self.x = np.linspace(-floor(NX/2)*cfg.DX, floor(NX/2)*cfg.DX, NX, endpoint=True)
        else:
            self.x = np.linspace(-((NX-1)/2.0)*cfg.DX, ((NX-1)/2.0)*cfg.DX, NX, endpoint=True)

        if NY%2 == 1:
            self.y = np.linspace(-floor(NY/2)*cfg.DY, floor(NY/2)*cfg.DY, NY, endpoint=True)
        else:
            self.y = np.linspace(-((NY-1)/2.0)*cfg.DY, ((NY-1)/2.0)*cfg.DY, NY, endpoint=True)

        if NZ%2 == 1:
            self.z = np.linspace(-floor(NZ/2)*cfg.DZ, floor(NZ/2)*cfg.DZ, NZ, endpoint=True)
        else:
            self.z = np.linspace(-((NZ-1)/2.0)*cfg.DZ, ((NZ-1)/2.0)*cfg.DZ, NZ, endpoint=True)
            


        self.h_ev = np.zeros((NX*NY*NZ, 4), cfg.real)
        self.h_nb = np.zeros(NX*NY*NZ, cfg.real)       
        self.h_qb = np.zeros((NX*NY*NZ*4), cfg.real)
        self.h_pi = np.zeros((NX*NY*NZ*10), cfg.real)
        self.h_bulkpr = np.zeros((NX*NY*NZ), cfg.real)

        
        # store the data in hdf5 file
        h5_path = os.path.join(cfg.fPathOut, 'bulkinfo.h5')
        self.f_hdf5 = h5py.File(h5_path, 'w')


        # time evolution for , edmax and ed, T at (x=0,y=0,etas=0)
        self.time = []
        self.edmax = []
        self.nbmax = []
        self.edcent = []
        self.Tcent = []

        # time evolution for total_entropy, eccp, eccx and <vr>
        self.energy = []
        self.baryon = []
        self.entropy = []
        self.eccp_vs_tau = []
        self.eccx = []
        self.vr= []


        
        self.X_2d = np.repeat(self.x,NY)
        self.X_3d = np.repeat(self.x,NY*NZ)
        self.Y_2d = np.tile(self.y,NX)
        self.Y_3d = np.repeat(self.Y_2d,NZ)
        self.Z_3d = np.tile(self.z,NX*NY)
    

    #@profile
    def get(self, tau, d_ev, d_nb, h_tpsmu, edmax, nbmax,d_qb=None, d_pi=None,d_bulkpr=None):
        ''' store the bulkinfo to hdf5 file '''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        
        self.time.append(tau)
        self.edmax.append(edmax)
        self.nbmax.append(nbmax)        

        cl.enqueue_copy(self.queue, self.h_ev, d_ev).wait()
        cl.enqueue_copy(self.queue, self.h_nb, d_nb).wait()
        if self.cfg.save_pimn:
            cl.enqueue_copy(self.queue, self.h_pi, d_pi).wait()
            pimn = self.h_pi.reshape(NX,NY,NZ,10)
        if self.cfg.save_qb:
            cl.enqueue_copy(self.queue, self.h_qb, d_qb).wait()
            qb = self.h_qb.reshape(NX,NY,NZ,4)
        if self.cfg.save_bulkpr:
            cl.enqueue_copy(self.queue, self.h_bulkpr, d_bulkpr).wait()
            bulkpr = self.h_bulkpr.reshape(NX,NY,NZ)
        bulk = self.h_ev.reshape(NX, NY, NZ, 4)
        bulk_nb = self.h_nb.reshape(NX, NY, NZ)
        tpsmu = h_tpsmu.reshape(NX, NY, NZ,4)

        # tau=0.6 changes to tau='0p6'
        time_stamp = ('%s'%tau).replace('.', 'p')

        i0, j0, k0 = NX//2, NY//2, NZ//2

        exy = bulk[:, :, k0, 0]
        vx = bulk[:, :, k0, 1]
        vy = bulk[:, :, k0, 2]
        vz = bulk[:, :, k0, 2]
    
        pressurexy = tpsmu[:,:,k0,1]
        


        self.f_hdf5.create_dataset('bulk2D/ed_tau%s'%time_stamp, data = bulk[:,:,k0,0].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk2D/vx_tau%s'%time_stamp, data = bulk[:,:,k0,1].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk2D/vy_tau%s'%time_stamp, data = bulk[:,:,k0,2].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk2D/vz_tau%s'%time_stamp, data = bulk[:,:,k0,3].flatten(),dtype="f")

        self.f_hdf5.create_dataset('bulk2D/nb_tau%s'%time_stamp, data = bulk_nb[:,:,k0].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk2D/t_tau%s'%time_stamp, data = tpsmu[:,:,k0,0].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk2D/pr_tau%s'%time_stamp, data = tpsmu[:,:,k0,1].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk2D/mu_tau%s'%time_stamp, data = tpsmu[:,:,k0,3].flatten(),dtype="f")

        if self.cfg.save_pimn:
            self.f_hdf5.create_dataset('bulk2D/pimn_tau%s'%time_stamp, data = pimn[:,:,k0,:].flatten(),dtype="f")
        if self.cfg.save_qb:
            self.f_hdf5.create_dataset('bulk2D/qb_tau%s'%time_stamp, data = qb[:,:,k0,:].flatten(),dtype="f")
        if self.cfg.save_bulkpr:
            self.f_hdf5.create_dataset('bulk2D/bulkpr_tau%s'%time_stamp, data = bulkpr[:,:,k0].flatten(),dtype="f")

         

        self.f_hdf5.create_dataset('bulk3D/ed_tau%s'%time_stamp, data = bulk[:,:,:,0].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk3D/vx_tau%s'%time_stamp, data = bulk[:,:,:,1].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk3D/vy_tau%s'%time_stamp, data = bulk[:,:,:,2].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk3D/vz_tau%s'%time_stamp, data = bulk[:,:,:,3].flatten(),dtype="f")

        self.f_hdf5.create_dataset('bulk3D/nb_tau%s'%time_stamp, data = bulk_nb[:,:,:].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk3D/t_tau%s'%time_stamp, data = tpsmu[:,:,:,0].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk3D/pr_tau%s'%time_stamp, data = tpsmu[:,:,:,1].flatten(),dtype="f")
        self.f_hdf5.create_dataset('bulk3D/mu_tau%s'%time_stamp, data = tpsmu[:,:,:,3].flatten(),dtype="f")

        if self.cfg.save_pimn:
            self.f_hdf5.create_dataset('bulk3D/pimn_tau%s'%time_stamp, data = pimn[:,:,:,:].flatten(),dtype="f")
        if self.cfg.save_qb:
            self.f_hdf5.create_dataset('bulk3D/qb_tau%s'%time_stamp, data = qb[:,:,:,:].flatten(),dtype="f")
        if self.cfg.save_bulkpr:
            self.f_hdf5.create_dataset('bulk3D/bulkpr_tau%s'%time_stamp, data = bulkpr[:,:,:].flatten(),dtype="f")
    

        self.eccp_vs_tau.append(self.eccp(exy, vx, vy,vz,pressurexy)[1])
        self.vr.append(self.mean_vr(exy, vx, vy, vz))
    
    def get_vorticity(self, tau, h_omega,h_omega_shear1,h_omega_shear2,h_omega_accT,h_omega_chemical):
        time_stamp = ('%s'%tau).replace('.', 'p')
        self.f_hdf5.create_dataset('vorticity3D/omega_tau%s'%time_stamp, data = h_omega.flatten(),dtype="f")
        self.f_hdf5.create_dataset('vorticity3D/omega_shear1_tau%s'%time_stamp, data = h_omega_shear1.flatten(),dtype="f")
        self.f_hdf5.create_dataset('vorticity3D/omega_shear2_tau%s'%time_stamp, data = h_omega_shear2.flatten(),dtype="f")
        self.f_hdf5.create_dataset('vorticity3D/omega_accT_tau%s'%time_stamp, data = h_omega_accT.flatten(),dtype="f")
        self.f_hdf5.create_dataset('vorticity3D/omega_chemical_tau%s'%time_stamp, data = h_omega_chemical.flatten(),dtype="f")

         




    def eccp(self, ed, vx, vy, vz, pressure):
        ''' eccx = <y*y-x*x>/<y*y+x*x> where <> are averaged 
            eccp = <Txx-Tyy>/<Txx+Tyy> '''
        ed[ed<1.0E-10] = 1.0E-10
        #ed[ed<0.05] = 0.0
        #pre = self.eos.f_P(ed)
        pre = pressure
        scale = np.ones_like(vx)
        vr2 = vx*vx + vy*vy + vz*vz
        scale[vr2>1.0] = np.sqrt(0.999999/vr2[vr2>1.0])
        vx[vr2>1.0] = scale[vr2>1.0]*vx[vr2>1.0]
        vy[vr2>1.0] = scale[vr2>1.0]*vy[vr2>1.0]
        vz[vr2>1.0] = scale[vr2>1.0]*vz[vr2>1.0]
        vr2[vr2>1.0] = 0.999999
        u0 = 1.0/np.sqrt(1.0 - vr2)

        Tyy = (ed + pre)*u0*u0*vy*vy + pre
        Txx = (ed + pre)*u0*u0*vx*vx + pre
        T0x = (ed + pre)*u0*u0*vx
        v2 = (ed*Txx - ed*Tyy).sum() / ( (ed*Txx + ed*Tyy).sum() + 1e-10)
        v1 = T0x.sum() / ((Txx + Tyy).sum() + 1e-10)
        return v1, v2

    def mean_vr(self, ed, vx, vy, vz=0.0):
        ''' <vr> = <gamma * ed * sqrt(vx*vx + vy*vy)>/<gamma*ed>
        where <> are averaged over whole transverse plane'''
        ed[ed<1.0E-10] = 1.0E-10
        vr2 = vx*vx + vy*vy + vz*vz
        vr2[vr2>1.0] = 0.999999
        u0 = 1.0/np.sqrt(1.0 - vr2)
        vr = (u0*ed*np.sqrt(vx*vx + vy*vy)).sum() / (u0*ed).sum()
        return vr
        
    def save(self ):
        self.f_hdf5.create_dataset('coord/tau', data = self.time)
        self.f_hdf5.create_dataset('coord/x', data = self.x)
        self.f_hdf5.create_dataset('coord/y', data = self.y)
        self.f_hdf5.create_dataset('coord/etas', data = self.z)

        self.f_hdf5.create_dataset('avg/eccp', data = np.array(self.eccp_vs_tau))
        self.f_hdf5.create_dataset('avg/vr', data = np.array(self.vr))
        self.f_hdf5.create_dataset('avg/ed_max', data = np.array(self.edmax))
        self.f_hdf5.create_dataset('avg/nb_max', data = np.array(self.nbmax))
        
        

        self.f_hdf5.close()




