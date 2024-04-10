#/usr/bin/env python
#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>


from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import pyopencl as cl
from pyopencl import array
import pyopencl.array as cl_array
import os
import sys
from time import time
from subprocess import call
import h5py


cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)
sys.path.append(os.path.abspath(os.path.join(cwd, '../pyvisc')))
sys.path.append(os.path.abspath(os.path.join(cwd, '../pyvisc/ini/trento')))
from eos.eos import Eos
from eos.chemical_potential import create_table as chemical_potential_on_hypersf
from config import read_config, write_config

def get_device_info(devices):
    print('image2d_max_width=', devices[0].image2d_max_width)
    print('local_mem_size=',    devices[0].local_mem_size)
    print('max_work_item_dimensions=', devices[0].max_work_item_dimensions)
    print('max_work_group_size=', devices[0].max_work_group_size)
    print('max_work_item_sizes=', devices[0].max_work_item_sizes)


class CLVisc(object):
    '''The pyopencl version for 3+1D CLVisc hydrodynamic simulation'''
    def __init__(self, configs,handcrafted_eos=None, gpu_id=0):
        '''Params:
        :param configs: hydrodynamic configurations, from configs import cfg
        :param gpu_id: use which gpu for the calculation if there are many per node
        '''
        # create opencl environment
        self.cfg = configs
        self.cwd, cwf = os.path.split(__file__)
        # create the fPathOut directory if not exists
        path = self.cfg.fPathOut
        if not os.path.exists(path):
            os.makedirs(path)

        # choose proper real, real4, real8 sizes
        self.determine_float_size(self.cfg)

        from backend_opencl import OpenCLBackend
        self.backend = OpenCLBackend(self.cfg, gpu_id)

        self.ctx = self.backend.ctx
        self.queue = self.backend.default_queue

        self.size= self.cfg.NX*self.cfg.NY*self.cfg.NZ
        self.tau = self.cfg.real(self.cfg.TAU0)

        self.compile_options = self.__compile_options()

        # set eos, create eos table for interpolation
        # self.eos_table must be before __loadAndBuildCLPrg() to pass
        # table information to definitions
        if handcrafted_eos is None:
            self.eos = Eos(self.cfg.eos_type)
        else:
            self.eos = handcrafted_eos
        
        if handcrafted_eos is not None:
            self.eos_table = self.eos.create_table(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() =='LATTICE_PCE165':
            self.eos_table = self.eos.create_table(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() =='LATTICE_PCE150':
            self.eos_table = self.eos.create_table(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'LATTICE_WB':
            self.eos_table = self.eos.create_table(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'HOTQCD2014':
            self.eos_table = self.eos.create_table(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'FIRST_ORDER':
            self.eos_table = self.eos.create_table(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'PURE_GAUGE':
            self.eos_table = self.eos.create_table(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'IDEAL_GAS_BARYON':
            self.eos_table = self.eos.create_table_nb(self.ctx,
                    self.compile_options, nrow=100, ncol=1555)
        elif self.cfg.eos_type.upper() == 'NEOSB':
            self.eos_table = self.eos.create_table_neosB(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'NEOSBQS':
            self.eos_table = self.eos.create_table_neosB(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'EOSQ':
            self.eos_table = self.eos.create_table_neosB(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'NJL_MODEL':
            self.eos_table = self.eos.create_table_neosB(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'CHIRAL':
            self.eos_table = self.eos.create_table_chiral(self.ctx,
                    self.compile_options)
        elif self.cfg.eos_type.upper() == 'HARDON_GAS':
            self.eos_table = self.eos.create_table_hardon_gas(self.ctx,
                    self.compile_options)
        else:
            self.eos_table = self.eos.create_table(self.ctx, self.compile_options)


        if self.cfg.Tfrz_on:
            self.efrz = self.eos.f_ed(self.cfg.TFRZ)
            self.cfg.Edfrz = self.efrz
        else:
            self.efrz = self.cfg.Edfrz
        
        chemical_potential_on_hypersf(self.efrz, path, eos_type=self.cfg.eos_type)
            
        
        self.copy_pdgfile()
        # copy pdgtable to output path
        
        if self.cfg.save_bulkinfo:
            # store 1D and 2d bulk info at each time step
            from bulkinfo import BulkInfo
            self.bulkinfo = BulkInfo(self.cfg, self.ctx, self.queue,
                    self.eos_table, self.compile_options)

        self.__loadAndBuildCLPrg()
        
        print (self.compile_options)
        self.switch_eos()
        # switch evolution eos to hypersurface eos and it must be behind of loadAndBuildCLPrg()
        
        
        #define buffer on device side, d_ev1 stores ed, vx, vy, vz
        mf = cl.mem_flags
        self.h_ev1 = np.zeros((self.size, 4), self.cfg.real)

        self.d_ev = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1)]
        self.d_Src = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1)
        self.d_Tmn = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1)
        
        self.h_nb = np.zeros((self.size),self.cfg.real)
        self.d_tpsmu = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1)
        self.h_tpsmu = np.zeros((self.size, 4), self.cfg.real)
    
        self.d_nb = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_nb),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_nb),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_nb)]
        self.d_J = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_nb)
    
        self.d_Src_nb = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_nb)
        
        self.d_nbmutp_old = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1)
        self.d_nbmutp_new = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_ev1)
        self.h_nbmutp = np.zeros((self.size, 4), self.cfg.real)
        self.d_sf_nbmutp = cl.Buffer(self.ctx, mf.READ_WRITE, size=1500000*self.cfg.sz_real4)
   
       

        self.submax = np.empty(64, self.cfg.real)
        self.d_submax = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, self.submax.nbytes)
        # d_ev_old: for hypersf calculation; 
        self.d_ev_old = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)
        self.d_ev_new = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)

        # d_hypersf: store the dSigma^{mu}, vx, vy, veta, tau, x, y, eta
        # on freeze out hyper surface
        self.d_hypersf = cl.Buffer(self.ctx, mf.READ_WRITE, size=1500000*self.cfg.sz_real8)
        # the position of the hyper surface in cartersian coordinates
        self.d_sf_txyz = cl.Buffer(self.ctx, mf.READ_WRITE, size=1500000*self.cfg.sz_real4)
        h_num_of_sf = np.zeros(1, np.int32)
        self.d_num_of_sf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=h_num_of_sf)



        self.h_pi0  = np.zeros(10*self.size, self.cfg.real)

        self.h_qb0  = np.zeros(4*self.size,self.cfg.real)
        self.h_mu  = np.zeros(self.size,self.cfg.real)


        self.d_qb = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.h_qb0),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.h_qb0),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = self.h_qb0)]

        self.d_qb_src = cl.Buffer(self.ctx, mf.READ_WRITE,size=self.h_qb0.nbytes)

        self.d_mubdx = cl.Buffer(self.ctx,mf.READ_WRITE ,size=self.h_mu.nbytes)
        self.d_mubdy = cl.Buffer(self.ctx,mf.READ_WRITE ,size=self.h_mu.nbytes)
        self.d_mubdz = cl.Buffer(self.ctx,mf.READ_WRITE ,size=self.h_mu.nbytes)

        self.d_mubdiff = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_mu.nbytes)



        self.d_qb_old = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_qb0.nbytes)
        self.d_qb_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=6000000*self.cfg.sz_real)
        
        self.copy_delta_qmu()


        # initialize the d_pi^{mu nu} with 0
        self.d_pi = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_pi0),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_pi0),
                     cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_pi0)]

        self.d_IS_src = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)
        # d_udx, d_udy, d_udz, d_udt are velocity gradients for viscous hydro
        # datatypes are real4






        nbytes_edv = self.h_ev1.nbytes
        self.d_udx = cl.Buffer(self.ctx, mf.READ_WRITE, size=nbytes_edv)
        self.d_udy = cl.Buffer(self.ctx, mf.READ_WRITE, size=nbytes_edv)
        self.d_udz = cl.Buffer(self.ctx, mf.READ_WRITE, size=nbytes_edv)



        self.h_bulkpr  = np.zeros(self.size,self.cfg.real)
        # initialize the bulk pressure with 0
        self.d_bulkpr = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_bulkpr),
                         cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_bulkpr),
                         cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_bulkpr)]

        self.d_ISpr_src = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_bulkpr.nbytes)

        if self.cfg.calc_vorticity:
            # d_omega thermal vorticity vector omega^{mu nu} = epsilon^{mu nu a b} d_a (beta u_b)
            # anti-symmetric 6 independent components

            self.h_omega = np.zeros(6*self.size, self.cfg.real)
            self.d_omega = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega),
                            cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega)]


            self.h_omega_shear1 = np.zeros(16*self.size, self.cfg.real)
            self.d_omega_shear1 = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega_shear1),
            cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega_shear1)]

            self.h_omega_shear2 = np.zeros(4*self.size, self.cfg.real)
            self.d_omega_shear2 = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega_shear2),
                    cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega_shear2)]


            self.h_omega_accT = np.zeros(6*self.size, self.cfg.real)
            self.d_omega_accT = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega_accT),
            cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega_accT)]

            
            self.h_omega_chemical = np.zeros(6*self.size, self.cfg.real)
            self.d_omega_chemical = [cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega_chemical),
            cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_omega_chemical)]

            # in case one wants to save omega^{mu} vector
            self.a_omega_mu = cl_array.empty(self.queue, 4*self.size, self.cfg.real)

            # get the vorticity on the freeze out hypersurface
            self.d_omega_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=9000000*self.cfg.sz_real)
            self.d_omega_shear1_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=24000000*self.cfg.sz_real)
            self.d_omega_shear2_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=6000000*self.cfg.sz_real)
            self.d_omega_accT_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=9000000*self.cfg.sz_real)


            
            self.d_omega_chemical_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=9000000*self.cfg.sz_real)




        # velocity difference between u_visc and u_ideal* for correction
        self.d_udiff = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_ev1.nbytes)


        # traceless and transverse check
        # self.d_checkpi = cl.Buffer(self.ctx, mf.READ_WRITE, size=self..h_ev1.nbytes)
        self.h_goodcell = np.ones(self.size, self.cfg.real)
        self.d_goodcell = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_goodcell.nbytes)

        cl.enqueue_copy(self.queue, self.d_pi[1], self.h_pi0).wait()
        cl.enqueue_copy(self.queue, self.d_goodcell, self.h_goodcell).wait()

        # used for freeze out hypersurface calculation
        self.d_pi_old = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_pi0.nbytes)

        # store the pi^{mu nu} on freeze out hyper surface
        self.d_pi_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=15000000*self.cfg.sz_real)
        self.d_bulkpr_old = cl.Buffer(self.ctx, mf.READ_WRITE, self.h_bulkpr.nbytes)
        self.d_bulkpr_sf = cl.Buffer(self.ctx, mf.READ_WRITE, size=1500000*self.cfg.sz_real)
        self.IS_initialize()


        self.history = []
        if self.cfg.Colhydro:
            from CoLBT.Jnu import Jnu, GetTV
            self.jnu=Jnu(self.backend,self.eos_table,self.compile_options)


    def __compile_options(self):
        optlist = ['TAU0', 'DT', 'DX', 'DY', 'DZ', 'CETA_XMIN', 'CETA_YMIN', \
                    'CETA_LEFT_SLOP', 'CETA_RIGHT_SLOP', 'LAM1']
        gpu_defines = [ '-D %s=%sf'%(key, value) for (key,value)
                in list(self.cfg.__dict__.items()) if key in optlist ]
        gpu_defines.append('-D {key}={value}'.format(key='NX', value=self.cfg.NX))
        gpu_defines.append('-D {key}={value}'.format(key='NY', value=self.cfg.NY))
        gpu_defines.append('-D {key}={value}'.format(key='NZ', value=self.cfg.NZ))
        gpu_defines.append('-D {key}={value}'.format(key='SIZE',
                           value=self.cfg.NX*self.cfg.NY*self.cfg.NZ))
        gpu_defines.append('-D {key}={value}'.format(key='CB_XMIN',value=self.cfg.CB_XMIN))
        #local memory size along x,y,z direction with 4 boundary cells
        gpu_defines.append('-D {key}={value}'.format(key='BSZ', value=self.cfg.BSZ))
        #determine float32 or double data type in *.cl file
        if self.cfg.use_float32:
            gpu_defines.append( '-D USE_SINGLE_PRECISION' )        
        
        if self.cfg.calc_vorticity:
            gpu_defines.append('-D CALC_VORTICITY_ON_SF')

        if self.cfg.CZETA_duke and self.cfg.bulkpr_on:
            gpu_defines.append('-D CZETA_DUKE')
        else:
            gpu_defines.append('-D {key}={value}'.format(key='CZETA_CONST',value=self.cfg.CZETA))

        if self.cfg.baryon_on:
            gpu_defines.append('-D BARYON_ON')
        if self.cfg.pimn_on:
            gpu_defines.append('-D PIMN_ON')
        if self.cfg.bulkpr_on:
            gpu_defines.append('-D BULKPR_ON')
        if self.cfg.qu_on:
            gpu_defines.append('-D QU_ON')

        #set the include path for the header file
        gpu_defines.append('-I '+os.path.join(self.cwd, 'kernel/'))
        return gpu_defines


    def copy_pdgfile(self):
        if self.cfg.afterburner.upper() == "SMASH":
            call(["cp",os.path.join(cwd,"eos/eos_table/pdg_smash.dat"),os.path.join(self.cfg.fPathOut,"pdgfile.dat")])
        elif self.cfg.afterburner.upper() == "URQMD":
            call(["cp",os.path.join(cwd,"eos/eos_table/pdg-urqmd.dat"),os.path.join(self.cfg.fPathOut,"pdgfile.dat")])
        else:
            call(["cp",os.path.join(cwd,"eos/eos_table/pdg05.dat"),os.path.join(self.cfg.fPathOut,"pdgfile.dat")])

        if self.cfg.Colhydro:
            call(["cp",os.path.join(cwd,"eos/eos_table/pdg_quark.dat"),os.path.join(self.cfg.fPathOut,"pdgfile.dat")])

 

    def __loadAndBuildCLPrg(self):

        with open(os.path.join(self.cwd, 'kernel', 'kernel_reduction.cl'), 'r') as f:
            src_maxEd = f.read()
            self.kernel_reduction = cl.Program(self.ctx, src_maxEd).build(
                                                 options=' '.join(self.compile_options))

 

        hypersf_defines = list(self.compile_options)
        hypersf_defines.append('-D {key}={value}'.format(key='nxskip', value=self.cfg.nxskip))
        hypersf_defines.append('-D {key}={value}'.format(key='nyskip', value=self.cfg.nyskip))
        hypersf_defines.append('-D {key}={value}'.format(key='nzskip', value=self.cfg.nzskip))
        hypersf_defines.append('-D {key}={value}f'.format(key='EFRZ', value=self.efrz))
        with open(os.path.join(self.cwd, 'kernel', 'kernel_hypersf.cl'), 'r') as f:
            src_hypersf = f.read()
            self.kernel_hypersf = cl.Program(self.ctx, src_hypersf).build(
                                             options=' '.join(hypersf_defines))
        #load and build *.cl programs with compile options
        if self.cfg.gubser_visc_test:
            self.compile_options.append('-D GUBSER_VISC_TEST')

        if self.cfg.riemann_test:
            self.compile_options.append('-D RIEMANN_TEST')
        
        if self.cfg.pimn_omega_coupling:
            self.compile_options.append('-D PIMUNU_OMEGA_COUPLING')

        if self.cfg.omega_omega_coupling:
            self.compile_options.append('-D OMEGA_OMEGA_COUPLING')

        with open(os.path.join(self.cwd, 'kernel', 'kernel_IS.cl'), 'r') as f:
           src = f.read()
           self.kernel_IS = cl.Program(self.ctx, src).build(options=' '.join(self.compile_options))

        with open(os.path.join(self.cwd, 'kernel', 'kernel_visc.cl'), 'r') as f:
            src = f.read()
            self.kernel_visc = cl.Program(self.ctx, src).build(options=' '.join(self.compile_options))

        with open(os.path.join(self.cwd, 'kernel', 'kernel_visc_nb.cl'), 'r') as f:
            src = f.read()
            self.kernel_visc_nb = cl.Program(self.ctx, src).build(options=' '.join(self.compile_options))

        with open(os.path.join(self.cwd, 'kernel', 'kernel_qu.cl'), 'r') as f:
           src = f.read()
           self.kernel_qub = cl.Program(self.ctx, src).build(options=' '.join(self.compile_options))

        with open(os.path.join(self.cwd, 'kernel', 'kernel_bulkpr.cl'), 'r') as f:
           src = f.read()
           self.kernel_bulkpr = cl.Program(self.ctx, src).build(options=' '.join(self.compile_options))
        

        with open(os.path.join(self.cwd, 'kernel', 'kernel_vorticity.cl'), 'r') as f:
            src = f.read()
            self.kernel_vorticity = cl.Program(self.ctx, src).build(options=' '.join(self.compile_options))



    def switch_eos(self):
        self.compile_options_hy = self.compile_options[:]
        self.eos_hy = self.eos
        self.eos_table_hy = self.eos_table

        if(self.cfg.switch_eos.upper() == "HARDON_GAS"):
            
            self.compile_options_hy.remove("-D %s"%self.cfg.eos_type.upper())
            for ele in self.compile_options_hy[:]:
                if "EOS_" in ele :
                    self.compile_options_hy.remove(ele)
            self.eos_hy = Eos(self.cfg.switch_eos)
            self.eos_table_hy = self.eos_hy.create_table_hardon_gas(self.ctx,self.compile_options_hy)

        with open(os.path.join(self.cwd, 'kernel', 'kernel_hypersf_eos.cl'), 'r') as f:
            prg_src = f.read()
            self.kernel_hypersf_eos = cl.Program(self.ctx, prg_src).build(
                                             options=' '.join(self.compile_options_hy))

        


    def determine_float_size(self, cfg):
        cfg.sz_int = np.dtype('int32').itemsize   #==sizeof(int) in c
        if cfg.use_float32 == True :
            cfg.real = np.float32
            cfg.real4 = array.vec.float4
            cfg.real2 = array.vec.float2
            cfg.real8 = array.vec.float8
            cfg.sz_real = np.dtype('float32').itemsize   #==sizeof(float) in c
            cfg.sz_real2 = array.vec.float2.itemsize
            cfg.sz_real4 = array.vec.float4.itemsize
            cfg.sz_real8 = array.vec.float8.itemsize
        else :
            cfg.real = np.float64
            cfg.real2 = array.vec.double2
            cfg.real4 = array.vec.double4
            cfg.real8 = array.vec.double8
            cfg.sz_real = np.dtype('float64').itemsize   #==sizeof(double) in c
            cfg.sz_real2= array.vec.double2.itemsize
            cfg.sz_real4= array.vec.double4.itemsize
            cfg.sz_real8= array.vec.double8.itemsize
    
    def copy_delta_qmu(self):
        call(["cp",os.path.join(os.path.abspath(cwd),"eos/eos_table/Coefficients_RTA_diffusion.dat"),os.path.join(self.cfg.fPathOut,"Coefficients_RTA_diffusion.dat")])

    def IS_initialize(self):
        '''initialize pi^{mu nu} tensor'''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        
        self.kernel_IS.visc_initialize(self.queue, (NX*NY*NZ,), None,
            self.d_qb[1],self.d_qb[2],self.d_pi[1],self.d_pi[2],self.d_bulkpr[1],self.d_bulkpr[2],self.d_goodcell, self.d_udiff,self.d_mubdiff, self.d_ev[1],self.d_nb[1],self.tau, self.eos_table).wait()
    
    def max_energy_density(self):
        '''Calc the maximum energy density on GPU and output the value '''
        self.kernel_reduction.reduction_stage1(self.queue, (256*64,), (256,), 
                self.d_ev[1], self.d_submax, np.int32(self.size) ).wait()
        cl.enqueue_copy(self.queue, self.submax, self.d_submax).wait()
        return self.submax.max()

    def max_baryon_density(self):
        '''Calc the maximum energy density on GPU and output the value '''
        self.kernel_reduction.reduction_stage3(self.queue, (256*64,), (256,), 
                self.d_nb[1], self.d_submax, np.int32(self.size) ).wait()
        cl.enqueue_copy(self.queue, self.submax, self.d_submax).wait()
        return self.submax.max()

    def get_hypersf_cornelius(self, n, ntskip, is_finished):
        '''get the freeze out hyper surface from d_ev_old and d_ev_new
        global_size=(NX//nxskip, NY//nyskip, NZ//nzskip}
        Params:
            :param n: the time step number
            :param ntskip: time step interval for hypersf calc
            :param is_finished: if True, the last time interval for hypersf
                   calculation will be different'''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        nx = (self.cfg.NX-1)//self.cfg.nxskip + 1
        ny = (self.cfg.NY-1)//self.cfg.nyskip + 1
        nz = (self.cfg.NZ-1)//self.cfg.nzskip + 1

        import h5py
        if n == 0:
            self.kernel_visc_nb.get_tmn(self.queue,(NX*NY*NZ,),None,\
                self.d_ev[1],self.d_nb[1],self.d_Tmn,self.d_J,self.eos_table) 
            self.kernel_hypersf_eos.reconst_hypersurface(self.queue,(NX*NY*NZ,), None,\
                self.d_J,self.d_nbmutp_old,self.d_ev_old,self.d_Tmn,self.eos_table_hy).wait() 
            self.tau_old = self.cfg.TAU0


            cl.enqueue_copy(self.queue, self.h_ev1,self.d_ev_old).wait()
            cl.enqueue_copy(self.queue, self.h_pi0,self.d_pi[1]).wait()
            cl.enqueue_copy(self.queue, self.h_qb0,self.d_qb[1]).wait()
            cl.enqueue_copy(self.queue, self.h_bulkpr,self.d_bulkpr[1]).wait()
            cl.enqueue_copy(self.queue, self.h_nbmutp, self.d_nbmutp_old).wait()
            
            
            outpath = os.path.join(self.cfg.fPathOut,"bulk_prev.h5")
            if self.cfg.calc_vorticity:
                
                cl.enqueue_copy(self.queue, self.h_omega, self.d_omega[1]).wait()
                cl.enqueue_copy(self.queue, self.h_omega_shear1, self.d_omega_shear1[1]).wait()
                cl.enqueue_copy(self.queue, self.h_omega_shear2, self.d_omega_shear2[1]).wait()
                cl.enqueue_copy(self.queue, self.h_omega_accT, self.d_omega_accT[1]).wait()
                cl.enqueue_copy(self.queue, self.h_omega_chemical, self.d_omega_chemical[1]).wait()
                
                with h5py.File(outpath,"w") as f:
                    f.create_dataset("ev",data = self.h_ev1)
                    f.create_dataset("pimn",data = self.h_pi0.reshape(self.size,10))
                    f.create_dataset("qb",data = self.h_qb0.reshape(self.size,4))
                    f.create_dataset("bulkpr",data = self.h_bulkpr.reshape(self.size,1))
                    f.create_dataset("nbmutp",data = self.h_nbmutp)
                    f.create_dataset("omega",data = self.h_omega.reshape(self.size,6))
                    f.create_dataset("omega_shear1",data = self.h_omega_shear1.reshape(self.size,16))
                    f.create_dataset("omega_shear2",data = self.h_omega_shear2.reshape(self.size,4))
                    f.create_dataset("omega_accT",data = self.h_omega_accT.reshape(self.size,6))
                    f.create_dataset("omega_chemical",data = self.h_omega_chemical.reshape(self.size,6))
                if self.cfg.corona:
                    call(["./cornelius/main","%s"%self.cfg.fPathOut,"%s"%self.tau_old,"%s"%self.tau_old,"0","1","1"])
            
            else:
                with h5py.File(outpath,"w") as f:
                    f.create_dataset("ev",data = self.h_ev1)
                    f.create_dataset("pimn",data = self.h_pi0.reshape(self.size,10))
                    f.create_dataset("qb",data = self.h_qb0.reshape(self.size,4))
                    f.create_dataset("bulkpr",data = self.h_bulkpr.reshape(self.size,1))
                    f.create_dataset("nbmutp",data = self.h_nbmutp)
                if self.cfg.corona:
                    call(["./cornelius/main","%s"%self.cfg.fPathOut,"%s"%self.tau_old,"%s"%self.tau_old,"0","1","0"])
            

                

        elif (n % ntskip == 0) or is_finished:

            tau_new = self.tau
            self.kernel_visc_nb.get_tmn(self.queue,(NX*NY*NZ,),None,\
                self.d_ev[1],self.d_nb[1],self.d_Tmn,self.d_J,self.eos_table) 
            self.kernel_hypersf_eos.reconst_hypersurface(self.queue,(NX*NY*NZ,), None,\
                self.d_J,self.d_nbmutp_new,self.d_ev_new,self.d_Tmn,self.eos_table_hy).wait() 
            
            cl.enqueue_copy(self.queue, self.h_ev1,self.d_ev_new).wait()
            cl.enqueue_copy(self.queue, self.h_pi0,self.d_pi[1]).wait()
            cl.enqueue_copy(self.queue, self.h_qb0,self.d_qb[1]).wait()
            cl.enqueue_copy(self.queue, self.h_bulkpr,self.d_bulkpr[1]).wait()
            cl.enqueue_copy(self.queue, self.h_nbmutp, self.d_nbmutp_new).wait()
            
                
            outpath = os.path.join(self.cfg.fPathOut,"bulk_curr.h5")

            if self.cfg.calc_vorticity:
                cl.enqueue_copy(self.queue, self.h_omega, self.d_omega[1]).wait()
                cl.enqueue_copy(self.queue, self.h_omega_shear1, self.d_omega_shear1[1]).wait()
                cl.enqueue_copy(self.queue, self.h_omega_shear2, self.d_omega_shear2[1]).wait()
                cl.enqueue_copy(self.queue, self.h_omega_accT, self.d_omega_accT[1]).wait()
                cl.enqueue_copy(self.queue, self.h_omega_chemical, self.d_omega_chemical[1]).wait()
                
                with h5py.File(outpath,"w") as f:                   
                    f.create_dataset("ev",data = self.h_ev1)
                    f.create_dataset("pimn",data = self.h_pi0.reshape(self.size,10))
                    f.create_dataset("qb",data = self.h_qb0.reshape(self.size,4))
                    f.create_dataset("bulkpr",data = self.h_bulkpr.reshape(self.size,1))
                    f.create_dataset("nbmutp",data = self.h_nbmutp)
                    f.create_dataset("omega",data = self.h_omega.reshape(self.size,6))
                    f.create_dataset("omega_shear1",data = self.h_omega_shear1.reshape(self.size,16))
                    f.create_dataset("omega_shear2",data = self.h_omega_shear2.reshape(self.size,4))
                    f.create_dataset("omega_accT",data = self.h_omega_accT.reshape(self.size,6))
                    f.create_dataset("omega_chemical",data = self.h_omega_chemical.reshape(self.size,6))

                header_flag = n//ntskip - 1
                if self.cfg.corona:
                    header_flag = 1 
                call(["./cornelius/main","%s"%self.cfg.fPathOut,"%s"%self.tau_old,"%s"%tau_new,"%d"%header_flag,"0","1"])
            else:
                with h5py.File(outpath,"w") as f:
                    f.create_dataset("ev",data = self.h_ev1)
                    f.create_dataset("pimn",data = self.h_pi0.reshape(self.size,10))
                    f.create_dataset("qb",data = self.h_qb0.reshape(self.size,4))
                    f.create_dataset("bulkpr",data = self.h_bulkpr.reshape(self.size,1))
                    f.create_dataset("nbmutp",data = self.h_nbmutp)
                header_flag = n//ntskip - 1
                if self.cfg.corona:
                    header_flag = 1 
                call(["./cornelius/main","%s"%self.cfg.fPathOut,"%s"%self.tau_old,"%s"%tau_new,"%d"%header_flag,"0","0"])
            call(["mv",os.path.join(self.cfg.fPathOut,"bulk_curr.h5"),os.path.join(self.cfg.fPathOut,"bulk_prev.h5")])
            if is_finished:
                call(["rm",os.path.join(self.cfg.fPathOut,"bulk_prev.h5")])

    
            self.tau_old = tau_new

    def get_hypersf_projection(self, n, ntskip, is_finished):
        '''get the freeze out hyper surface from d_ev_old and d_ev_new
        global_size=(NX//nxskip, NY//nyskip, NZ//nzskip}
        Params:
            :param n: the time step number
            :param ntskip: time step interval for hypersf calc
            :param is_finished: if True, the last time interval for hypersf
                   calculation will be different'''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        nx = (self.cfg.NX-1)//self.cfg.nxskip + 1
        ny = (self.cfg.NY-1)//self.cfg.nyskip + 1
        nz = (self.cfg.NZ-1)//self.cfg.nzskip + 1
        if n == 0:
            self.kernel_visc_nb.get_tmn(self.queue,(NX*NY*NZ,),None,\
                self.d_ev[1],self.d_nb[1],self.d_Tmn,self.d_J,self.eos_table) 
            self.kernel_hypersf_eos.reconst_hypersurface(self.queue,(NX*NY*NZ,), None,\
                self.d_J,self.d_nbmutp_old,self.d_ev_old,self.d_Tmn,self.eos_table_hy).wait() 
            self.tau_old = self.cfg.TAU0

            cl.enqueue_copy(self.queue, self.d_pi_old, self.d_pi[1]).wait()
            cl.enqueue_copy(self.queue, self.d_qb_old, self.d_qb[1]).wait()
            cl.enqueue_copy(self.queue, self.d_bulkpr_old, self.d_bulkpr[1]).wait()
            edlo=0.05
            if self.cfg.corona:
                if self.cfg.calc_vorticity:
                    self.kernel_hypersf.visc_corona(self.queue, (nx, ny, nz), None,self.d_hypersf,
                            self.d_sf_txyz,self.d_pi_sf,self.d_num_of_sf,
                            self.d_ev_old,self.d_pi_old,self.d_nbmutp_old,
                            self.d_sf_nbmutp,self.d_qb_old,self.d_qb_sf,
                            self.d_bulkpr_old,self.d_bulkpr_sf,self.d_omega_sf,
                            self.d_omega[0],self.d_omega_shear1_sf,self.d_omega_shear1[0],
                            self.d_omega_shear2_sf,self.d_omega_shear2[0],
                            self.d_omega_accT_sf,self.d_omega_accT[0],
                            self.d_omega_chemical_sf,self.d_omega_chemical[0],
                            self.cfg.real(self.cfg.TAU0),self.cfg.real(edlo),
                            self.cfg.real(self.cfg.Edfrz),self.eos_table)
                else:
                    self.kernel_hypersf.visc_corona(self.queue, (nx, ny, nz), None,self.d_hypersf,
                            self.d_sf_txyz,self.d_pi_sf,self.d_num_of_sf,
                            self.d_ev_old,self.d_pi_old,self.d_nbmutp_old,
                            self.d_sf_nbmutp,self.d_qb_old,self.d_qb_sf,
                            self.d_bulkpr_old,self.d_bulkpr_sf,
                            self.cfg.real(self.cfg.TAU0),self.cfg.real(edlo),
                            self.cfg.real(self.cfg.Edfrz),self.eos_table)


                

        elif (n % ntskip == 0) or is_finished:

            tau_new = self.tau
            self.kernel_visc_nb.get_tmn(self.queue,(NX*NY*NZ,),None,\
                self.d_ev[1],self.d_nb[1],self.d_Tmn,self.d_J,self.eos_table) 
            self.kernel_hypersf_eos.reconst_hypersurface(self.queue,(NX*NY*NZ,), None,\
                self.d_J,self.d_nbmutp_new,self.d_ev_new,self.d_Tmn,self.eos_table_hy).wait() 
            if self.cfg.calc_vorticity:
                self.kernel_hypersf.visc_hypersf(self.queue, (nx, ny, nz), None,
                        self.d_hypersf, self.d_sf_txyz,
                        self.d_pi_sf, self.d_num_of_sf,
                        self.d_ev_old, self.d_ev_new,
                        self.d_pi_old, self.d_pi[1],
                        self.d_nbmutp_old, self.d_nbmutp_new,self.d_sf_nbmutp,
                        self.d_qb_old, self.d_qb[1],self.d_qb_sf,
                        self.d_bulkpr_old,self.d_bulkpr[1],self.d_bulkpr_sf,
                        self.d_omega_sf, self.d_omega[0], self.d_omega[1],
                        self.d_omega_shear1_sf,self.d_omega_shear1[0],self.d_omega_shear1[1],
                        self.d_omega_shear2_sf,self.d_omega_shear2[0],self.d_omega_shear2[1],
                        self.d_omega_accT_sf,self.d_omega_accT[0],self.d_omega_accT[1],
                        self.d_omega_chemical_sf,self.d_omega_chemical[0],self.d_omega_chemical[1],
                        self.cfg.real(self.tau_old), self.cfg.real(tau_new),self.eos_table).wait()
            else:
                self.kernel_hypersf.visc_hypersf(self.queue, (nx, ny, nz), None,
                        self.d_hypersf, self.d_sf_txyz,
                        self.d_pi_sf, self.d_num_of_sf,
                        self.d_ev_old, self.d_ev_new,
                        self.d_pi_old, self.d_pi[1],
                        self.d_nbmutp_old, self.d_nbmutp_new,self.d_sf_nbmutp,
                        self.d_qb_old, self.d_qb[1],self.d_qb_sf,
                        self.d_bulkpr_old,self.d_bulkpr[1],self.d_bulkpr_sf,
                        self.cfg.real(self.tau_old), self.cfg.real(tau_new),self.eos_table).wait()


            # update with current tau and d_ev[1], d_pi[1] and d_omega[1]
            cl.enqueue_copy(self.queue, self.d_ev_old,
                            self.d_ev_new).wait()
            cl.enqueue_copy(self.queue, self.d_pi_old, self.d_pi[1]).wait()
            cl.enqueue_copy(self.queue, self.d_bulkpr_old, self.d_bulkpr[1]).wait()
            if self.cfg.calc_vorticity:
                cl.enqueue_copy(self.queue, self.d_omega[0], self.d_omega[1]).wait()
                cl.enqueue_copy(self.queue, self.d_omega_shear1[0], self.d_omega_shear1[1]).wait()
                cl.enqueue_copy(self.queue, self.d_omega_shear2[0], self.d_omega_shear2[1]).wait()
                cl.enqueue_copy(self.queue, self.d_omega_accT[0], self.d_omega_accT[1]).wait()
                cl.enqueue_copy(self.queue, self.d_omega_chemical[0], self.d_omega_chemical[1]).wait()
                cl.enqueue_copy(self.queue, self.d_qb_old, self.d_qb[1]).wait()
            cl.enqueue_copy(self.queue, self.d_nbmutp_old,self.d_nbmutp_new).wait()
    
            self.tau_old = tau_new

    def smear_from_parton_list_ampt(self, parton_list,NEVENT=1):

        parton_list0 = np.zeros((len(parton_list[:,0]),9))
        for i in range(len(parton_list[:,0])):
            if self.cfg.baryon_on:
                pid = int (parton_list[i,8])
                baryon = 1.0/3.0
                if np.isclose(pid ,1) or np.isclose(pid ,2) or np.isclose(pid ,3):
                    baryon = 1.0/3.0
                elif np.isclose(pid ,-1) or np.isclose(pid ,-2) or np.isclose(pid ,-3):
                    baryon = -1.0/3.0
                parton_list[i,8] = baryon
            else:    
                parton_list0[i,:8] = parton_list[i,:]
        if not self.cfg.baryon_on:
            parton_list = parton_list0[:,:]
        from smearing import SmearingP4X4
        SmearingP4X4(self.cfg, self.ctx, self.queue, self.compile_options,self.h_ev1,self.h_nb,
                self.d_ev[1], self.d_nb[1], parton_list, self.eos_table, model = "ampt",NEVENT=NEVENT)

    def smear_from_hadron_list_smash(self, cfg, hadron_list, PDGCODE,NEVENT=1):
        # to_do
        self.spectator = hadron_list[hadron_list[:,12]==0,:]
        #np.savetxt(os.path.join(self.cfg.fPathOut ,"spectators.dat"),self.spectator[:,:12],fmt = "%s") 
        hadron_list = hadron_list[hadron_list[:,12]!=0,:]
        
        nhadron = len(hadron_list[:,0])
        print (nhadron/NEVENT)
        for id in np.arange(nhadron):
            PID = hadron_list[id,9]
            if int( abs(PID) ) in PDGCODE :
                hadron_list[id,9] = np.sign(hadron_list[id,9])
            else:
                hadron_list[id,9] = 0
        print ("baryon", np.sum(hadron_list[:,9])) 
        

        input_hadron = hadron_list[:,:10]


        from smearing import SmearingP4X4
        SmearingP4X4(self.cfg, self.ctx, self.queue, self.compile_options,self.h_ev1,self.h_nb,
                self.d_ev[1], self.d_nb[1], input_hadron, self.eos_table,model = "smash",NEVENT=NEVENT)




    def optical_glauber_ini(self, system='Au+Au',save_binary_collisions=False):
        '''initialize with optical glauber model for Au+Au and Pb+Pb collisions
        Params:
            :param system: type string, Au+Au for Au+Au 200 GeV, Pb+Pb for Pb+Pb 2.76 TeV and 5.02 TeV
            :param save_binary_collisions: type bool, true to save num_of_binary_collisions for jet
                 energy loss study (used to sample jet creation position '''

        from glauber import Glauber, weight_mean_b
        cent_min = np.float(self.cfg.cent.split("_")[0])
        cent_max = np.float(self.cfg.cent.split("_")[-1])
        if cent_min is not None and cent_max is not None:
            mean_impact_parameter = weight_mean_b(cent_min, cent_max, system)
            # notice the config is changed here, write_to_config() must be called after this
            # to save the correct ImpactParameter
            self.cfg.ImpactParameter = mean_impact_parameter

        glauber = Glauber(self.cfg, self.ctx, self.queue, self.compile_options,
                        self.d_ev[1],self.d_nb[1])
        if save_binary_collisions:
            glauber.save_nbinary(self.ctx, self.queue, self.cfg)

    def trento2D_ini(self,s_scale):
        mf = cl.mem_flags
        trento_defines = list(self.compile_options)
        trento_defines.append('-D {key}={value:f}f'.format(key='Eta_flat', value=self.cfg.Eta_flat))
        trento_defines.append('-D {key}={value:f}f'.format(key='Eta_gw', value=self.cfg.Eta_gw))
        with open(os.path.join(self.cwd, 'kernel', 'kernel_trento.cl'), 'r') as f:
            prg_src = f.read()
            self.kernel_trento = cl.Program(self.ctx, prg_src).build(options=' '.join(trento_defines))
        s_scale = s_scale.T
        s_scale = s_scale.flatten()
        s_scale = s_scale.astype(self.cfg.real)
        d_s_scale = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=s_scale)
        self.kernel_trento.from_sd_to_ed(self.queue,(self.cfg.NX, self.cfg.NY),None, d_s_scale, self.d_ev[1],self.d_nb[1],self.eos_table ).wait()





    def get_vorticity(self, save_data=False,loop=0):
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

            
        print ("tau vorticity ", self.tau)
        if loop == 0:
            return 
        # the \partial_tau = 0 
        #if loop == 0:
        #    d_ev_old = self.d_ev[1]
        #    d_ev_new = self.d_ev[1]
        #    
        #    d_nb_old = self.d_nb[1]
        #    d_nb_new = self.d_nb[1]
        #else:
        #    d_ev_old = self.d_ev[0]
        #    d_ev_new = self.d_ev[1]

        #    d_nb_old = self.d_nb[0]
        #    d_nb_new = self.d_nb[1]

        d_ev_old = self.d_ev[0]
        d_ev_new = self.d_ev[1]

        d_nb_old = self.d_nb[0]
        d_nb_new = self.d_nb[1]

        self.kernel_vorticity.omega(self.queue, (NX, NY, NZ), None,
                d_ev_old, d_ev_new,
                d_nb_old, d_nb_new,
                self.d_omega[1],
                self.eos_table, self.tau).wait()


        self.kernel_vorticity.omega_shear(self.queue, (NX, NY, NZ), None,
                d_ev_old, d_ev_new,
                d_nb_old, d_nb_new,
                self.d_omega_shear1[1],self.d_omega_shear2[1],
                self.eos_table,self.tau).wait()


        self.kernel_vorticity.omega_accT(self.queue, (NX, NY, NZ), None,
                d_ev_old, d_ev_new,
                d_nb_old, d_nb_new,
                self.d_omega_accT[1],
                self.eos_table,self.tau).wait()
        
        
        self.kernel_vorticity.omega_chemical(self.queue, (NX, NY, NZ), None,
                d_ev_old, d_ev_new, 
                d_nb_old, d_nb_new,
                self.d_omega_chemical[1],
                self.eos_table,self.tau).wait()
        
        if loop == 1:
            cl.enqueue_copy(self.queue,self.d_omega[0],self.d_omega[1]).wait()
            cl.enqueue_copy(self.queue,self.d_omega_shear1[0],self.d_omega_shear1[1]).wait()
            cl.enqueue_copy(self.queue,self.d_omega_shear2[0],self.d_omega_shear2[1]).wait() 
            cl.enqueue_copy(self.queue,self.d_omega_accT[0],self.d_omega_accT[1]).wait() 
            cl.enqueue_copy(self.queue,self.d_omega_chemical[0],self.d_omega_chemical[1]).wait() 



        if save_data:
            self.kernel_vorticity.omega_mu(self.queue, (NX*NY*NZ, ), None,
                self.a_omega_mu.data, self.d_ev[1],
                self.d_omega[1], self.eos_table,
                self.cfg.real(self.efrz), self.tau).wait()

            path_out = os.path.abspath(self.cfg.fPathOut)

            fname = ('omega_mu_tau%s'%self.tau).replace('.', 'p') + '.dat'

            omega_mu = self.a_omega_mu.get()

            np.savetxt(os.path.join(path_out, fname),
                       omega_mu.reshape(NX*NY*NZ, 4),
                       header='omega^{t, x, y, z} = Omega^{mu nu} u_{nu}')



    def update_udiff(self):
        '''get d_udiff = u_{visc}^{n} - u_{visc}^{n-1}, it is possible to
        set d_udiff in analytical solution for viscous gubser test'''
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ


        self.kernel_IS.get_udiff(self.queue, (NX*NY*NZ,), None,
            self.d_udiff,self.d_mubdiff,self.d_ev[0], self.d_ev[1],
            self.d_nb[0],self.d_nb[1],self.eos_table).wait()




    def update_time(self, loop):
        '''update time with TAU0 and loop, convert its type to np.float32 or 
        float64 which can be used directly as parameter in kernel functions'''
        self.tau = self.cfg.real(self.cfg.TAU0 + (loop+1)*self.cfg.DT)


    def visc_stepUpdate(self, step, switch_eloss=False):
        ''' Do step update in kernel with KT algorithm for visc evolution
            Args:
                gpu_ev_old: self.d_ev[1] for the 1st step,
                            self.d_ev[2] for the 2nd step
                step: the 1st or the 2nd step in runge-kutta
        '''
        # upadte d_Src by KT time splitting, along=1,2,3 for 'x','y','z'
        # input: gpu_ev_old, tau, size, along_axis
        # output: self.d_Src

        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ
        tau = self.tau
        self.kernel_visc.kt_src_christoffel(self.queue, (NX*NY*NZ, ), None,
                         self.d_Src, self.d_Src_nb,self.d_ev[step],self.d_nb[step],
                         self.d_pi[step], self.d_bulkpr[step], self.eos_table,tau, np.int32(step)).wait()


        self.kernel_visc.kt_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                self.d_Src, self.d_ev[step],self.d_nb[step],
                self.d_pi[step],self.d_bulkpr[step], self.eos_table,tau).wait()

        self.kernel_visc.kt_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                self.d_Src, self.d_ev[step],self.d_nb[step],
                self.d_pi[step],self.d_bulkpr[step], self.eos_table,tau).wait()

        self.kernel_visc.kt_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                self.d_Src, self.d_ev[step],self.d_nb[step],
                self.d_pi[step],self.d_bulkpr[step], self.eos_table,tau).wait()
        
        if self.cfg.baryon_on:
            self.kernel_visc_nb.kt_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                    self.d_Src_nb, self.d_nb[step],self.d_qb[step],
                    self.d_ev[step],tau, self.eos_table,np.int32(step)).wait()

            self.kernel_visc_nb.kt_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                    self.d_Src_nb, self.d_nb[step],self.d_qb[step],
                    self.d_ev[step],tau, self.eos_table,np.int32(step)).wait()

            self.kernel_visc_nb.kt_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                    self.d_Src_nb, self.d_nb[step],self.d_qb[step],
                    self.d_ev[step],tau, self.eos_table,np.int32(step)).wait()


        if switch_eloss:
            self.jnu.update_src(self.d_Src)

        self.kernel_visc_nb.update_evn(self.queue, (NX*NY*NZ, ), None,
                                  self.d_ev[3-step], self.d_ev[1],self.d_ev[2],
                                  self.d_nb[3-step], self.d_nb[1],
                                  self.d_pi[0], self.d_pi[3-step],
                                  self.d_qb[0], self.d_qb[3-step],
                                  self.d_bulkpr[0], self.d_bulkpr[3-step],self.d_udiff,
                                  self.d_Src,self.d_Src_nb,
                                  self.eos_table, self.tau, np.int32(step),np.float32(self.nbmax)).wait()







    def BES_stepUpdate(self, step):
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

        if self.cfg.qu_on:
            self.kernel_qub.qub_src_christoffel(self.queue, (NX*NY*NZ,), None,
                     self.d_qb_src, self.d_qb[step], self.d_ev[step],
                     self.tau,np.int32(step)).wait()

            self.kernel_qub.qub_src_alongx(self.queue, (BSZ, NY, NZ),(BSZ, 1, 1),
                     self.d_qb_src, self.d_mubdx, self.d_qb[step], self.d_ev[step],
                     self.d_nb[step],self.eos_table,self.tau,np.int32(step)).wait()

            self.kernel_qub.qub_src_alongy(self.queue, (NX, BSZ, NZ),(1, BSZ, 1),
                     self.d_qb_src, self.d_mubdy, self.d_qb[step], self.d_ev[step],
                     self.d_nb[step],self.eos_table,self.tau,np.int32(step)).wait()

            self.kernel_qub.qub_src_alongz(self.queue, (NX, NY, BSZ),(1, 1, BSZ),
                     self.d_qb_src, self.d_mubdz, self.d_qb[step], self.d_ev[step],
                     self.d_nb[step],self.eos_table,self.tau,np.int32(step)).wait()

            self.kernel_qub.update_qub(self.queue, (NX*NY*NZ,), None,
                    self.d_qb[3-step], self.d_goodcell, self.d_qb[1], self.d_qb[step],
                    self.d_ev[1], self.d_ev[2], self.d_nb[1],
                    self.d_nb[2], self.d_udiff,self.d_udx, self.d_udy,
                    self.d_udz, self.d_mubdiff, self.d_mubdx,self.d_mubdy, self.d_mubdz,
                    self.d_qb_src,self.eos_table, self.tau, np.int32(step)).wait()





    def IS_bulkpr_stepupdate(self,step):
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ
        if self.cfg.bulkpr_on:
            self.kernel_bulkpr.bulkpr_src_christoffel(self.queue,(NX*NY*NZ,),None,
                self.d_ISpr_src, self.d_bulkpr[step],np.int32(step)).wait()

            self.kernel_bulkpr.bulkpr_src_alongx(self.queue,(BSZ,NY,NZ),(BSZ, 1, 1),
                self.d_ISpr_src,self.d_bulkpr[step],self.d_ev[step],self.d_nb[step],
                self.eos_table, self.tau).wait()


            self.kernel_bulkpr.bulkpr_src_alongy(self.queue,(NX, BSZ, NZ), (1, BSZ, 1),
                self.d_ISpr_src,self.d_bulkpr[step],self.d_ev[step],self.d_nb[step],
                self.eos_table, self.tau).wait()

            self.kernel_bulkpr.bulkpr_src_alongz(self.queue, (NX, NY, BSZ),(1, 1, BSZ),
                self.d_ISpr_src,self.d_bulkpr[step],self.d_ev[step],self.d_nb[step],
                self.eos_table, self.tau).wait()

            self.kernel_bulkpr.update_bulkpr(self.queue, (NX*NY*NZ,), None,
                        self.d_bulkpr[3-step], self.d_bulkpr[1], self.d_bulkpr[step],self.d_pi[step],
                        self.d_ev[1], self.d_ev[2], self.d_nb[1],
                        self.d_nb[2], self.d_udiff,self.d_udx, self.d_udy,
                        self.d_udz, self.d_ISpr_src,self.eos_table, self.tau,
                        np.int32(step)).wait()




    def IS_stepUpdate(self, step):
        NX, NY, NZ, BSZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ, self.cfg.BSZ

        if self.cfg.pimn_on:
            self.kernel_IS.visc_src_christoffel(self.queue, (NX*NY*NZ,), None,
                    self.d_IS_src, self.d_pi[step], self.d_ev[step],self.d_nb[step],
                    self.tau, np.int32(step)).wait()

            self.kernel_IS.visc_src_alongx(self.queue, (BSZ, NY, NZ), (BSZ, 1, 1),
                    self.d_IS_src, self.d_udx, self.d_pi[step], self.d_ev[step],self.d_nb[step],
                    self.eos_table, self.tau).wait()

            self.kernel_IS.visc_src_alongy(self.queue, (NX, BSZ, NZ), (1, BSZ, 1),
                    self.d_IS_src, self.d_udy, self.d_pi[step], self.d_ev[step],self.d_nb[step],
                    self.eos_table, self.tau).wait()

            self.kernel_IS.visc_src_alongz(self.queue, (NX, NY, BSZ), (1, 1, BSZ),
                    self.d_IS_src, self.d_udz, self.d_pi[step], self.d_ev[step],self.d_nb[step],
                    self.eos_table,self.tau).wait()

            #    # for step==1, d_ev[2] is useless, since u_new = u_old + d_udiff
            #    # for step==2, d_ev[2] is used to calc u_new
            self.kernel_IS.update_pimn(self.queue, (NX*NY*NZ,), None,
                    self.d_pi[3-step], self.d_goodcell, self.d_pi[1], self.d_pi[step],
                    self.d_ev[1], self.d_ev[2], self.d_nb[1],
                    self.d_nb[2], self.d_udiff,self.d_udx, self.d_udy,
                    self.d_udz, self.d_IS_src,self.eos_table, self.tau,
                    np.int32(step)
                    ).wait()

    def load_handcrafted_initial_condition(self):
        print('start to load ini data')
        data = pd.read_csv(self.cfg.Initial_profile,sep=' ', dtype=self.cfg.real,comment="#",header=None)
        data = data.values
        data = data.astype(self.cfg.real)
        self.h_ev1[:,0] = self.cfg.KFACTOR*data[:,0]
        self.h_ev1[:,1] = data[:,1]
        self.h_ev1[:,2] = data[:,2]
        self.h_ev1[:,3] = data[:,3]

        cl.enqueue_copy(self.queue, self.d_ev[1], self.h_ev1).wait()
        if self.cfg.baryon_on:
            data = pd.read_csv(self.cfg.Initial_nb_profile,sep=' ', dtype=self.cfg.real)
            data = data.values
            data = data.astype(self.cfg.real)
            self.h_nb[:] = data[:,0]
            cl.enqueue_copy(self.queue, self.d_nb[1], self.h_nb).wait()

        print('finish ini data')
   

    def hypersf_eos(self,switch_eos="hardon_gas"):
        NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
        sf_nmtp = np.loadtxt(os.path.join(self.cfg.fPathOut,"sf_nbmutp.dat"),dtype= self.cfg.real)
        cwd = os.getcwd()
        os.chdir(self.cfg.fPathOut)
        call(["cp", "sf_nbmutp.dat", "sf_nbmutp.datbk"])
        os.chdir(cwd)

        self.sf_nmtp_eos = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE ,size=sf_nmtp.nbytes)
        cl.enqueue_copy(self.queue,self.sf_nmtp_eos, sf_nmtp).wait()
        nline = len(sf_nmtp[:,0])
        self.kernel_hypersf_eos.get_tpsmu_eos(self.queue, (NX*NY*NZ, ), None, self.sf_nmtp_eos, self.eos_table_hy, np.int32(nline)).wait()
        cl.enqueue_copy(self.queue, sf_nmtp, self.sf_nmtp_eos).wait()
        np.savetxt(os.path.join(self.cfg.fPathOut,"sf_nbmutp.dat"), sf_nmtp ,fmt='%.6e', header="nb, mu, T, PrplusEd of hypersf elements")
    
    def get_bulkinfo_vorticity(self,loop):
        if loop == 0:
            return 0
        else:
            cl.enqueue_copy(self.queue, self.h_omega, self.d_omega[1]).wait()
            cl.enqueue_copy(self.queue, self.h_omega_shear1, self.d_omega_shear1[1]).wait()
            cl.enqueue_copy(self.queue, self.h_omega_shear2, self.d_omega_shear2[1]).wait()
            cl.enqueue_copy(self.queue, self.h_omega_accT, self.d_omega_accT[1]).wait()
            cl.enqueue_copy(self.queue, self.h_omega_chemical, self.d_omega_chemical[1]).wait()

            self.bulkinfo.get_vorticity(self.tau,self.h_omega,self.h_omega_shear1,self.h_omega_shear2,self.h_omega_accT,self.h_omega_chemical)


    def save_pimn_sf(self, set_to_zero=False):
        '''save pimn information on freeze out hyper surface
        Params:
            :param set_to_zero: True to set pimn on surface to 0.0,
            in case eta/s=0, ideal evolution is switch on'''

        num_of_sf = self.num_of_sf
        comment_line = "#pi00 01 02 03 11 12 13 23 33"
        pi_onsf = np.zeros(10*num_of_sf, dtype=self.cfg.real)
        if not set_to_zero:
            cl.enqueue_copy(self.queue, pi_onsf, self.d_pi_sf).wait()
        out_path = os.path.join(self.cfg.fPathOut, 'pimnsf.dat')
        print("pimn on frzsf is saved to ", out_path)
        np.savetxt(out_path, pi_onsf.reshape(num_of_sf, 10), fmt='%.6e',
                   header = comment_line)

    def save_qb_sf(self, set_to_zero=False):

        num_of_sf = self.num_of_sf
        qb_onsf = np.zeros(4*num_of_sf, dtype=self.cfg.real)
        if not set_to_zero:
            cl.enqueue_copy(self.queue, qb_onsf, self.d_qb_sf).wait()
        out_path = os.path.join(self.cfg.fPathOut, 'qbmusf.dat')
        print("qb on frzsf is saved to ", out_path)

        comment_line ='qb0 qb1 qb2 qb3'
        np.savetxt(out_path, qb_onsf.reshape(num_of_sf, 4), fmt='%.6e',
                   header = comment_line)
    def save_bulkpr_sf(self, set_to_zero=False):

        num_of_sf = self.num_of_sf
        bulkpr_onsf = np.zeros(num_of_sf, dtype=self.cfg.real)
        if not set_to_zero:
            cl.enqueue_copy(self.queue, bulkpr_onsf, self.d_bulkpr_sf).wait()
        out_path = os.path.join(self.cfg.fPathOut, 'bulkprsf.dat')
        print("bulk pressure on frzsf is saved to ", out_path)

        comment_line ='bulkpr'
        np.savetxt(out_path, bulkpr_onsf.reshape(num_of_sf, 1), fmt='%.6e',
                   header = comment_line)

    def save_vorticity_sf(self):
        if self.cfg.calc_vorticity:
            
            
            
            num_of_sf = self.num_of_sf
            omega_mu = np.empty(6*num_of_sf, dtype=self.cfg.real)
            cl.enqueue_copy(self.queue, omega_mu, self.d_omega_sf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'omegamu_sf.dat')
            print("vorticity omega_{mu, nu} on surface is saved to", out_path)
            np.savetxt(out_path, omega_mu.reshape(self.num_of_sf, 6),
                       fmt='%.6e', header = 'omega^{01, 02, 03, 12, 13, 23}')


            omega_shear1 = np.empty(16*num_of_sf, dtype=self.cfg.real)
            cl.enqueue_copy(self.queue, omega_shear1, self.d_omega_shear1_sf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'omegamu_shear1_sf.dat')
            print("vorticity omega_{mu, nu} on surface is saved to", out_path)
            np.savetxt(out_path, omega_shear1.reshape(self.num_of_sf, 16),
                       fmt='%.6e', header = 'omega^{01, 02, 03, 12, 13, 23}')

            omega_shear2 = np.empty(4*num_of_sf, dtype=self.cfg.real)
            cl.enqueue_copy(self.queue, omega_shear2, self.d_omega_shear2_sf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'omegamu_shear2_sf.dat')
            print("vorticity omega_{mu, nu} on surface is saved to", out_path)
            np.savetxt(out_path, omega_shear2.reshape(self.num_of_sf, 4),
                       fmt='%.6e', header = 'omega^{01, 02, 03, 12, 13, 23}')

            omega_accT = np.empty(6*num_of_sf, dtype=self.cfg.real)
            cl.enqueue_copy(self.queue, omega_accT, self.d_omega_accT_sf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'omegamu_accT_sf.dat')
            print("vorticity omega_{mu, nu} on surface is saved to", out_path)
            np.savetxt(out_path, omega_accT.reshape(self.num_of_sf, 6),
                       fmt='%.6e', header = 'omega^{01, 02, 03, 12, 13, 23}')
            
            omega_chemical = np.empty(6*num_of_sf, dtype=self.cfg.real)
            cl.enqueue_copy(self.queue, omega_chemical, self.d_omega_chemical_sf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'omegamu_chemical_sf.dat')
            print("vorticity omega_{mu, nu} on surface is saved to", out_path)
            np.savetxt(out_path, omega_chemical.reshape(self.num_of_sf, 6),
                      fmt='%.6e', header = 'omega^{01, 02, 03, 12, 13, 23}')




    def save(self):
        self.num_of_sf = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(self.queue, self.num_of_sf, self.d_num_of_sf).wait()
        # convert the single value array [num_of_sf] to num_of_sf.
        self.num_of_sf = np.squeeze(self.num_of_sf)
        if self.cfg.save_hypersf and (not self.cfg.cornelius):
            print("num of sf=", self.num_of_sf)
            hypersf = np.empty(self.num_of_sf, dtype=self.cfg.real8)
            cl.enqueue_copy(self.queue, hypersf, self.d_hypersf).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'hypersf.dat')
            if not self.cfg.Tfrz_on:
                np.savetxt(out_path, hypersf, fmt='%.6e', header =\
                    'efrz=%.6e ; other rows: dS0, dS1, dS2, dS3, vx, vy, veta, etas'%self.efrz)
            else:
                np.savetxt(out_path, hypersf, fmt='%.6e', header =\
                    'Tfrz=%.6e ; other rows: dS0, dS1, dS2, dS3, vx, vy, veta, etas'%self.cfg.TFRZ)

            sf_txyz = np.empty(self.num_of_sf, dtype=self.cfg.real4)
            cl.enqueue_copy(self.queue, sf_txyz, self.d_sf_txyz).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'sf_txyz.dat')
            np.savetxt(out_path, sf_txyz, fmt='%.6e', header =
              '(t, x, y, z) the time-space coordinates of hypersf elements')
            sf_nbmutp = np.empty(self.num_of_sf, dtype = self.cfg.real4)
            cl.enqueue_copy(self.queue, sf_nbmutp, self.d_sf_nbmutp).wait()
            out_path = os.path.join(self.cfg.fPathOut, 'sf_nbmutp.dat')
            np.savetxt(out_path, sf_nbmutp, fmt='%.6e', header =
                    'nb, mu, T, Pr of hypersf elements')
            self.hypersf_eos()
            self.save_pimn_sf()
            self.save_qb_sf()
            self.save_bulkpr_sf()
            self.save_vorticity_sf()

        if self.cfg.cornelius and self.cfg.save_hypersf:
                self.hypersf_eos()

        if self.cfg.save_bulkinfo:
            self.bulkinfo.save()



    def evolve(self, max_loops=1000, save_pi=False, force_run_to_maxloop = False,debug=False):
        '''The main loop of hydrodynamic evolution
        Args:
            jet_eloss_src: one dictionary stores the jet eloss information '''
        # if etaos<1.0E-6, use ideal hydrodynamics which is much faster
        
    
        self.start = False
        loop = 0
        for loop in np.arange(max_loops):
            t0 = time()
            # stop at max_loops if force_run_to_maxloop set to True
            if force_run_to_maxloop and loop == max_loops:
                break

            if loop % self.cfg.ntskip == 0:
                

                self.edmax = self.max_energy_density()
                
                cl.enqueue_copy(self.queue, self.h_nb,self.d_nb[1]).wait()
                self.nbmax = self.max_baryon_density()
                self.history.append([self.tau, self.edmax,self.nbmax])
                print('tau=', self.tau, ' EdMax= ',self.edmax, ' Nbmax=',self.nbmax)
                
                    
           
            is_finished = False
            
            if self.cfg.source and self.start == True:
                is_finished = self.edmax < self.efrz
            if not self.cfg.source:
                is_finished = self.edmax < self.efrz

            NX, NY, NZ = self.cfg.NX, self.cfg.NY, self.cfg.NZ
            
            
            
            
            # add vorticity calculation here            
            if (loop % self.cfg.ntskip == 0 or loop==1) and self.cfg.calc_vorticity:
                self.get_vorticity(save_data=False,loop=loop)

            if self.cfg.save_hypersf:
                if self.cfg.cornelius:
                    self.get_hypersf_cornelius(loop, self.cfg.ntskip, is_finished )
                else:
                    self.get_hypersf_projection(loop, self.cfg.ntskip, is_finished )
                
                 
            if (self.cfg.save_bulkinfo) and loop % (self.cfg.ntskip) == 0:
                 bulk_t1 = time()
                 self.kernel_visc_nb.get_tpsmu(self.queue, (NX*NY*NZ, ), None, self.d_tpsmu, self.eos_table, self.d_ev[1], self.d_nb[1]).wait()
                 cl.enqueue_copy(self.queue, self.h_tpsmu, self.d_tpsmu).wait()
                 self.bulkinfo.get(self.tau, self.d_ev[1],self.d_nb[1], self.h_tpsmu, self.edmax, self.nbmax,\
                         self.d_qb[1],self.d_pi[1],self.d_bulkpr[1])
                 bulk_t2 = time()
                 print ("Bulk cost %s s"%(bulk_t2-bulk_t1))
            
            if (loop % self.cfg.ntskip == 0 or loop==1) and self.cfg.calc_vorticity and self.cfg.save_bulkinfo:
                 self.get_bulkinfo_vorticity(loop)

            # finish if edmax < freeze out energy density
            
            if is_finished and not force_run_to_maxloop:
                if self.cfg.Colhydro:
                   self.jnu.tc_inf()
                #if self.cfg.source:
                #    self.save_corona_particles()
                #if self.cfg.cornelius:
                #    call(["rm", "%s/bulk_prev.h5"%self.cfg.fPathOut])
                break
            #todo
            # self.start =  self.ideal.edmax > self.ideal.efrz
            # if self.cfg.source and self.start==False:
            #     self.reset_cell()
            
            
                

            # store d_pi[0] for self.visc_stepUpdate()
            cl.enqueue_copy(self.queue, self.d_pi[0],self.d_pi[1]).wait()
            cl.enqueue_copy(self.queue, self.d_bulkpr[0],self.d_bulkpr[1]).wait()
            
            # copy the d_ev[1] to d_ev[0] for umu_new prediction
            cl.enqueue_copy(self.queue, self.d_ev[0],self.d_ev[1]).wait()
            
            cl.enqueue_copy(self.queue, self.d_nb[0],self.d_nb[1]).wait()
            cl.enqueue_copy(self.queue, self.d_qb[0],self.d_qb[1]).wait()

            switch_eloss = False
            
            if self.cfg.Colhydro:
                self.jnu.nextstep(self.tau,self.d_ev[1],self.d_nb[1])
                switch_eloss = True
            
            # update pi[2] with d_ev[0] and u_new=u0+d_udiff
            # where d_udiff is prediction from previous step
            self.IS_stepUpdate(step=1)
            self.IS_bulkpr_stepupdate(step=1)
            self.BES_stepUpdate(step=1)
            self.visc_stepUpdate(step=1,switch_eloss=switch_eloss)
            self.update_time(loop)
            # update pi[1] with d_ev[0] and d_ev[2]_visc*

            self.IS_stepUpdate(step=2)
            self.IS_bulkpr_stepupdate(step=2)
            self.BES_stepUpdate(step=2)
            self.visc_stepUpdate(step=2,switch_eloss=switch_eloss)
            self.update_udiff()


            # save the bad cells at debug stage
            if debug and loop % self.cfg.ntskip == 0:
                self.check_bad_cell()

            loop = loop + 1
            t1 = time()
            print('one step: {dtime}'.format(dtime = t1-t0 ))

        self.save() 
        out_path = os.path.join(self.cfg.fPathOut, 'history.dat')
        np.savetxt(out_path, self.history, fmt='%.6e', header =
              '(t, x, y, z) the time-space coordinates of hypersf elements')




if __name__ == "__main__":
    fconfig = sys.argv[1]
    cfg = read_config(fconfig)
    visc = CLVisc(cfg, gpu_id=0)
