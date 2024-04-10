#/usr/bin/env python
#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>
'''calc the Lambda polarization on the freeze out hypersurface'''

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import os
import sys
from time import time
import math
import four_momentum as mom
from scipy.interpolate import interp1d, splrep, splint
import pyopencl as cl
from pyopencl.array import Array
import pyopencl.array as cl_array
from tqdm import tqdm

os.environ['PYOPENCL_CTX']=':1'

class Polarization(object):
    '''The pyopencl version for lambda polarisation,
    initialize with freeze out hyper surface and omega^{mu}
    on freeze out hyper surface.'''
    def __init__(self, fpath,T=0.165, Mu=0.0,mass=1.116, chemical = False,gpu_id = 0,themal=True,shear=False,accT=False):
        '''Param:
             sf: the freeze out hypersf ds0,ds1,ds2,ds3,vx,vy,veta,etas
             omega: omega^tau, x, y, etas
             T: freezeout temperature
        '''
        self.cwd, cwf = os.path.split(__file__)

        platform = cl.get_platforms()
        devices = platform[0].get_devices(device_type=cl.device_type.GPU)
        devices = [devices[gpu_id]]
        self.ctx = cl.Context(devices=devices)
        self.queue = cl.CommandQueue(self.ctx)
        mf = cl.mem_flags

        self.Tfrz = T
        self.Mu = Mu
        self.mass = mass
        self.chemical = chemical

        self.themal = themal
        self.shear = shear
        self.accT = accT
        
        # calc umu since they are used for each (Y,pt,phi)
        sf = np.loadtxt(fpath+'/hypersf.dat', dtype=np.float32)
        self.size_sf = len(sf[:,0])
        h_sf = sf.astype(np.float32)
        self.d_sf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_sf)
        nbmutp = pd.read_csv(fpath+'/sf_nbmutp.dat', dtype=np.float32,sep=" ",header=None,comment="#").values.flatten()
        h_nbmutp = nbmutp.astype(np.float32)
        self.d_nbmutp = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_nbmutp)
        self.flag_nb = True
        
        nbmutp_tem = nbmutp.reshape(self.size_sf,4)
        nb_total = np.sum(nbmutp_tem[:,0])
        if np.fabs(nb_total) < 0.001:
            self.flag_nb = False
        print (self.flag_nb) 
        if self.themal:    
            omega = np.loadtxt(fpath+'/omegamu_sf.dat', dtype=np.float32).flatten()
            h_omega = omega.astype(np.float32)
            self.d_omega = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega)
        

        if self.shear:
            omega_shear1 = pd.read_csv(fpath+'/omegamu_shear1_sf.dat', dtype=np.float32,sep=" ",header=None,comment="#").values.flatten()
            omega_shear2 = pd.read_csv(fpath+'/omegamu_shear2_sf.dat', dtype=np.float32,sep=" ",header=None,comment="#").values.flatten()
            h_omega_shear1 = omega_shear1.astype(np.float32)
            h_omega_shear2 = omega_shear2.astype(np.float32)
            
            self.d_omega_shear1 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega_shear1)
            self.d_omega_shear2 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega_shear2)


        if self.accT:
            omega_accT = pd.read_csv(fpath+'/omegamu_accT_sf.dat', dtype=np.float32,sep=" ",header=None,comment="#").values.flatten()
            h_omega_accT = omega_accT.astype(np.float32)
            self.d_omega_accT = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega_accT)
            

        if self.chemical:
            omega_chemical = pd.read_csv(fpath+'/omegamu_chemical_sf.dat', dtype=np.float32,sep=" ",header=None,comment="#").values.flatten()
            h_omega_chemical = omega_chemical.astype(np.float32)
            self.d_omega_chemical = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_omega_chemical)
            
            
            

    def set_momentum(self,momentum_list):
        self.d_momentum = cl_array.to_device(self.queue,
                                        momentum_list.astype(np.float32))

        num_of_mom = len(momentum_list)

        print('num_of_mom=', num_of_mom)

        compile_options = ['-D num_of_mom=%s '%num_of_mom]

        cwd, cwf = os.path.split(__file__)

        self.block_size = 256
        compile_options.append('-D BSZ=%s '%self.block_size)
        compile_options.append('-D SIZE_SF=%s '%np.int32(self.size_sf))
        compile_options.append('-D MASS=%s '%np.float32(self.mass))
        
        
        if self.flag_nb:
            compile_options.append('-D BARYON_ON ')
        compile_options.append('-I '+os.path.join(cwd, '../pyvisc/kernel/ ')) 
        compile_options.append( '-D USE_SINGLE_PRECISION ' )
        
        #print (compile_options)
        fpath = os.path.join(cwd, '../pyvisc/kernel/kernel_polarization.cl')

        with open(fpath, 'r') as f:
            src = f.read()
            self.prg = cl.Program(self.ctx, src).build(options=''.join(compile_options))
        self.size = num_of_mom

    def get_themal(self,anti_baryon=False):

        
        h_pol_th = np.zeros((self.size, 4), np.float32)
        h_rho_th = np.zeros(self.size, np.float32)

        # boost Pi^{mu} to the local rest frame of Lambda
        h_pol_lrf_th = np.zeros((self.size, 4), np.float32)

        self.d_pol = cl_array.to_device(self.queue, h_pol_th)
        self.d_rho = cl_array.to_device(self.queue, h_rho_th)

        self.d_pol_lrf = cl_array.to_device(self.queue, h_pol_lrf_th)
        
        anti_paritcle = np.int32(anti_baryon)
        print (anti_paritcle) 


        # block_size*num_of_mom: gloabl size
        # block_size: local size
        self.prg.polarization_on_sf(self.queue, (self.block_size*self.size,),
                (self.block_size,), self.d_pol.data, self.d_rho.data, self.d_pol_lrf.data,
                self.d_sf,self.d_nbmutp, self.d_omega, self.d_momentum.data,anti_paritcle).wait()

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        pol_lrf = self.d_pol_lrf.get()
        return polarization, density, pol_lrf


    def get_shear(self,anti_baryon=False):

        
        h_pol_th = np.zeros((self.size, 4), np.float32)
        h_rho_th = np.zeros(self.size, np.float32)

        # boost Pi^{mu} to the local rest frame of Lambda
        h_pol_lrf_th = np.zeros((self.size, 4), np.float32)

        self.d_pol = cl_array.to_device(self.queue, h_pol_th)
        self.d_rho = cl_array.to_device(self.queue, h_rho_th)

        self.d_pol_lrf = cl_array.to_device(self.queue, h_pol_lrf_th)
        
        anti_paritcle = np.int32(anti_baryon)

        # block_size*num_of_mom: gloabl size
        # block_size: local size
        self.prg.polarization_shear_on_sf(self.queue, (self.block_size*self.size,),
                (self.block_size,), self.d_pol.data, self.d_rho.data, self.d_pol_lrf.data,
                self.d_sf,self.d_nbmutp, self.d_omega_shear1, self.d_omega_shear2,self.d_momentum.data,anti_paritcle).wait()

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        pol_lrf = self.d_pol_lrf.get()
        return polarization, density, pol_lrf

    def get_accT(self,anti_baryon=False):

        
        h_pol_th = np.zeros((self.size, 4), np.float32)
        h_rho_th = np.zeros(self.size, np.float32)

        # boost Pi^{mu} to the local rest frame of Lambda
        h_pol_lrf_th = np.zeros((self.size, 4), np.float32)

        self.d_pol = cl_array.to_device(self.queue, h_pol_th)
        self.d_rho = cl_array.to_device(self.queue, h_rho_th)

        self.d_pol_lrf = cl_array.to_device(self.queue, h_pol_lrf_th)
        

        anti_paritcle = np.int32(anti_baryon)


        # block_size*num_of_mom: gloabl size
        # block_size: local size
        self.prg.polarization_accT_on_sf(self.queue, (self.block_size*self.size,),
                (self.block_size,), self.d_pol.data, self.d_rho.data, self.d_pol_lrf.data,
                self.d_sf,self.d_nbmutp,  self.d_omega_accT,self.d_momentum.data,anti_paritcle).wait()

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        pol_lrf = self.d_pol_lrf.get()
        return polarization, density, pol_lrf


    def get_chemical(self,anti_baryon=False):


        h_pol_th = np.zeros((self.size, 4), np.float32)
        h_rho_th = np.zeros(self.size, np.float32)

        # boost Pi^{mu} to the local rest frame of Lambda
        h_pol_lrf_th = np.zeros((self.size, 4), np.float32)

        self.d_pol = cl_array.to_device(self.queue, h_pol_th)
        self.d_rho = cl_array.to_device(self.queue, h_rho_th)

        self.d_pol_lrf = cl_array.to_device(self.queue, h_pol_lrf_th)
        

        anti_paritcle = np.int32(anti_baryon)


        # block_size*num_of_mom: gloabl size
        # block_size: local size
        if self.flag_nb :
            self.prg.polarization_chemical_on_sf(self.queue, (self.block_size*self.size,), 
                    (self.block_size,), self.d_pol.data, self.d_rho.data, self.d_pol_lrf.data,
                    self.d_sf,self.d_nbmutp,  self.d_omega_chemical,self.d_momentum.data,anti_paritcle).wait()
    

        polarization = self.d_pol.get()
        density = self.d_rho.get()
        pol_lrf = self.d_pol_lrf.get()
        return polarization, density, pol_lrf



def rapidity_integral(spec_along_y, ylo=-0.5, yhi=0.5):
    '''1D integration along rapidity/pseudo-rapidity 
    The spline interpolation and integration is much faster than
    the interp1d() and quad combination'''
    #f = interp1d(Y, spec_along_y, kind='cubic')
    #return quad(f, ylo, yhi, epsrel=1.0E-5)[0]
    tck = splrep(mom.Y, spec_along_y)
    return splint(ylo, yhi, tck)

def pt_integral(spec_along_pt, ptlo=0.0, pthi=3.0):
    '''1D integration along rapidity/pseudo-rapidity 
    The spline interpolation and integration is much faster than
    the interp1d() and quad combination'''
    #f = interp1d(Y, spec_along_y, kind='cubic')
    #return quad(f, ylo, yhi, epsrel=1.0E-5)[0]
    tck = splrep(mom.PT, spec_along_pt)
    return splint(ptlo, pthi, tck)


def phi_integral(spec_along_phi, philo=0.0, phihi=2*np.pi):
    '''1D integration along rapidity/pseudo-rapidity 
    The spline interpolation and integration is much faster than
    the interp1d() and quad combination'''
    #f = interp1d(Y, spec_along_y, kind='cubic')
    #return quad(f, ylo, yhi, epsrel=1.0E-5)[0]
    tck = splrep(mom.PHI, spec_along_phi)
    return splint(philo, phihi, tck)
def etatorapidity(eta,mass,pt):
    p0 = np.sqrt (mass**2 + (pt*np.cosh(eta))**2)
    pz = pt*np.sinh(eta)

    rap = 0.5*(np.log(p0+pz)-np.log(p0-pz))
    return rap


def get_polar_phi(fpath,pid,polar,density,pol_lrf,mode="th"):
    
    polar = polar.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    pol_lrf = pol_lrf.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    density = density.reshape(mom.NY,mom.NPT,mom.NPHI)


    polar_int = np.zeros(4)
    polar_lrf_int = np.zeros(4)

    density_int = np.zeros(4)
    density_phi = np.zeros((mom.NPHI))
    density_phi_Y = np.zeros((mom.NPHI,mom.NY))

    polar_phi = np.zeros((mom.NPHI,4))
    polar_phi_Y = np.zeros((mom.NPHI,mom.NY,4))

    polar_phi_lrf = np.zeros((mom.NPHI,4))
    polar_phi_Y_lrf = np.zeros((mom.NPHI,mom.NY,4))

    polar_phi_output = np.zeros((mom.NPHI,4))
    polar_phi_lrf_output = np.zeros((mom.NPHI,4))


    for i in range(4):
        for k, phi in enumerate(mom.PHI):
            for j, Y in enumerate(mom.Y):
                
                spec_along_PT = density[j,:,k]
                density_phi_Y[k,j] = pt_integral(mom.PT*spec_along_PT,ptlo=0.5,pthi=3.0)
                
                spec_along_PT = polar[j,:,k,i]
                polar_phi_Y[k,j,i] = pt_integral(mom.PT*spec_along_PT,ptlo=0.5,pthi=3.0)

                spec_along_PT = pol_lrf[j,:,k,i]
                polar_phi_Y_lrf[k,j,i] = pt_integral(mom.PT*spec_along_PT,ptlo=0.5,pthi=3.0)
               

            spec_along_y = density_phi_Y[k,:]
            density_phi[k] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)
            
            spec_along_y = polar_phi_Y[k,:,i]
            polar_phi[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)
            polar_phi_output[k,i] = polar_phi[k,i]/density_phi[k]

            spec_along_y = polar_phi_Y_lrf[k,:,i]
            polar_phi_lrf[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)
            polar_phi_lrf_output[k,i] = polar_phi_lrf[k,i]/density_phi[k]

        polar_int[i] = phi_integral(polar_phi[:,i])/phi_integral(density_phi[:])
        polar_lrf_int[i] = phi_integral(polar_phi_lrf[:,i])/phi_integral(density_phi[:])
    polar_phi = np.column_stack((mom.PHI,polar_phi_output))
    polar_phi_lrf = np.column_stack((mom.PHI,polar_phi_lrf_output))

    np.savetxt('%s/polar_%s_%s.dat'%(fpath,mode,pid), polar_phi)
    np.savetxt('%s/polar_%s_lrf_%s.dat'%(fpath,mode,pid), polar_phi_lrf)
    np.savetxt('%s/polar_%s_int_%s.dat'%(fpath,mode,pid), polar_int)
    np.savetxt('%s/polar_%s_lrf_int_%s.dat'%(fpath,mode,pid), polar_lrf_int)

def get_polar_pt(fpath,pid,polar,density,pol_lrf,mode="th"):
    
    
    polar = polar.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    pol_lrf = pol_lrf.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    density = density.reshape(mom.NY,mom.NPT,mom.NPHI)


    density_pt = np.zeros((mom.NPT))
    density_pt_Y = np.zeros((mom.NPT,mom.NY))

    polar_pt = np.zeros((mom.NPT,4))
    polar_pt_Y = np.zeros((mom.NPT,mom.NY,4))

    polar_pt_lrf = np.zeros((mom.NPT,4))
    polar_pt_Y_lrf = np.zeros((mom.NPT,mom.NY,4))

    polar_pt_output = np.zeros((mom.NPT,4))
    polar_pt_lrf_output = np.zeros((mom.NPT,4))


    for i in range(4):
        for k, PT in enumerate(mom.PT):
            for j, Y in enumerate(mom.Y):
                
                spec_along_phi = density[j,k,:] #Y*PT*PHI*index
                density_pt_Y[k,j] = phi_integral(spec_along_phi)
                
                spec_along_phi = polar[j,k,:,i] #Y*PT*PHI*index
                polar_pt_Y[k,j,i] = phi_integral(spec_along_phi)

                spec_along_phi = pol_lrf[j,k,:,i] #Y*PT*PHI*index
                polar_pt_Y_lrf[k,j,i] = phi_integral(spec_along_phi)
               

            spec_along_y = density_pt_Y[k,:]
            density_pt[k] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)
            
            spec_along_y = polar_pt_Y[k,:,i]
            polar_pt[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)

            polar_pt_output[k,i] = polar_pt[k,i]/density_pt[k]

            spec_along_y = polar_pt_Y_lrf[k,:,i]
            polar_pt_lrf[k,i] = rapidity_integral(spec_along_y,ylo=-1,yhi=1)
            polar_pt_lrf_output[k,i] = polar_pt_lrf[k,i]/density_pt[k]


    polar_pt = np.column_stack((mom.PT,polar_pt_output))
    polar_pt_lrf = np.column_stack((mom.PT,polar_pt_lrf_output))

    np.savetxt('%s/polar_pt_%s_%s.dat'%(fpath,mode,pid), polar_pt)
    np.savetxt('%s/polar_pt_%s_lrf_%s.dat'%(fpath,mode,pid), polar_pt_lrf)


def get_polar_y(fpath,pid,polar,density,pol_lrf,mode="th"):
    
    polar = polar.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    pol_lrf = pol_lrf.reshape(mom.NY,mom.NPT,mom.NPHI,4)
    density = density.reshape(mom.NY,mom.NPT,mom.NPHI)


    density_int = np.zeros(4)
    density_Y = np.zeros((mom.NY))
    density_Y_pt = np.zeros((mom.NY,mom.NPT))

    polar_Y = np.zeros((mom.NY,4))
    polar_Y_pt = np.zeros((mom.NY,mom.NPT,4))

    polar_Y_lrf = np.zeros((mom.NY,4))
    polar_Y_pt_lrf = np.zeros((mom.NY,mom.NPT,4))

    polar_Y_output = np.zeros((mom.NY,4))
    polar_Y_lrf_output = np.zeros((mom.NY,4))


    for i in range(4):
        for k, Y in enumerate(mom.Y):
            for j, PT in enumerate(mom.PT):
                
                spec_along_phi = density[k,j,:]
                density_Y_pt[k,j] = phi_integral(spec_along_phi)
                
                spec_along_phi = polar[k,j,:,i]
                polar_Y_pt[k,j,i] = phi_integral(spec_along_phi)

                spec_along_phi = pol_lrf[k,j,:,i]
                polar_Y_pt_lrf[k,j,i] = phi_integral(spec_along_phi)
               

            spec_along_PT = density_Y_pt[k,:]
            density_Y[k] = pt_integral(mom.PT*spec_along_PT,ptlo=0.5,pthi=3.0)
            
            spec_along_PT = polar_Y_pt[k,:,i]
            polar_Y[k,i] = pt_integral(mom.PT*spec_along_PT,ptlo=0.5,pthi=3.0)
            polar_Y_output[k,i] = polar_Y[k,i]/density_Y[k]

            spec_along_PT = polar_Y_pt_lrf[k,:,i]
            polar_Y_lrf[k,i] = pt_integral(mom.PT*spec_along_PT,ptlo=0.5,pthi=3.0)
            polar_Y_lrf_output[k,i] = polar_Y_lrf[k,i]/density_Y[k]


    polar_Y = np.column_stack((mom.Y,polar_Y_output))
    polar_Y_lrf = np.column_stack((mom.Y,polar_Y_lrf_output))

    np.savetxt('%s/polar_y_%s_%s.dat'%(fpath,mode,pid), polar_Y)
    np.savetxt('%s/polar_y_%s_lrf_%s.dat'%(fpath,mode,pid), polar_Y_lrf)


def calc_pol_th(fpath,pid,rapidity = "Y"):
    
    mass = 1.116
    anti_baryon = False
    if "squark" in pid:
        mass = 0.3
    if "anti" in pid:
        anti_baryon = True
    
    
    pol = Polarization(fpath,mass = mass,themal=True,shear=True,accT=True,chemical=True)
     
    ny,npt, nphi = mom.NY, mom.NPT, mom.NPHI
    momentum_list = np.zeros((ny*npt*nphi, 4), dtype=np.float32)
    
    for k, Y in enumerate(mom.Y):
       for i, pt in enumerate(mom.PT):
           for j, phi in enumerate(mom.PHI):
               px = pt * math.cos(phi)
               py = pt * math.sin(phi)
               mt = math.sqrt(mass*mass + px*px + py*py)
               index = k*npt*nphi + i*nphi + j
               momentum_list[index, 0] = mt
               momentum_list[index, 1] = Y
               momentum_list[index, 2] = px
               momentum_list[index, 3] = py
               if rapidity.upper() == "ETA":
                   momentum_list[index, 1] = etatorapidity(Y,mass,pt) 
     
    pol.set_momentum(momentum_list)
    polar, density, pol_lrf = pol.get_themal(anti_baryon=anti_baryon)
    get_polar_phi(fpath,pid,polar,density,pol_lrf,mode = "th")
    get_polar_pt(fpath,pid,polar,density,pol_lrf,mode = "th")
    get_polar_y(fpath,pid,polar,density,pol_lrf,mode = "th")


    polar, density, pol_lrf = pol.get_shear(anti_baryon=anti_baryon)
    get_polar_phi(fpath,pid,polar,density,pol_lrf,mode = "shear")
    get_polar_pt(fpath,pid,polar,density,pol_lrf,mode = "shear")
    get_polar_y(fpath,pid,polar,density,pol_lrf,mode = "shear")


    polar, density, pol_lrf = pol.get_accT(anti_baryon=anti_baryon)
    get_polar_phi(fpath,pid,polar,density,pol_lrf,mode = "accT")
    get_polar_pt(fpath,pid,polar,density,pol_lrf,mode = "accT")
    get_polar_y(fpath,pid,polar,density,pol_lrf,mode = "accT")
    
    
    if pol.chemical and pol.flag_nb:
        
        polar, density, pol_lrf = pol.get_chemical(anti_baryon=anti_baryon)
        get_polar_phi(fpath,pid,polar,density,pol_lrf,mode = "chemical")
        get_polar_pt(fpath,pid,polar,density,pol_lrf,mode = "chemical")
        get_polar_y(fpath,pid,polar,density,pol_lrf,mode = "chemical")
        


if __name__ == '__main__':
    
    energylist = ["auau7p7"]
    for energy in energylist:
        for i in range(0,1):
             #fpath = '/media/xywu/DATA2/data/pol/smooth_ampt_initial/%s/20_50/event%s'%(energy,i)
             #fpath = '/media/xywu/DATA2/data/pol/smooth_smash_git_test/%s/20_50/event%s'%(energy,i)
             #fpath = '/media/xywu/DATA2/data/pol/smooth_smash_git_test_bulk/%s/20_50/event%s'%(energy,i)
             fpath = '/media/xywu/disk21/clviscBES_forqing/CLVisc3p0/CLVisc3p0/CLVisc3.0/pyvisc/results/polarization/auau7p7/event0/'
             #fpath = "/media/xywu/disk21/clviscBES_forqing/CLVisc3p0/CLVisc3p0/CLVisc3.0/pyvisc/results/polar_old/auau7p7/20_50/event0"
             #fpath = "/media/xywu/disk21/physics/code/pol/CLViscBES2/pyvisc/results/smooth_ampt_initial/auau7p7/20_50/event0/"
             print ("\n", energy)
             calc_pol_th(fpath,"lambda")
             #calc_pol_th(fpath,"anti_lambda")
             #calc_pol_th(fpath,"squark")
             #calc_pol_th(fpath,"anti_squark")
        
    
