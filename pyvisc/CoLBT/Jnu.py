from __future__ import absolute_import, division, print_function
import os
import math as ma
import pyopencl as cl
import numpy as np
from backend_opencl import OpenCLBackend
import sys
cwd, cwf = os.path.split(__file__)
sys.path.append(cwd)
import LBT.LBT as LBT

acu = 1.0E-7

def GetTV(partonlist):
    medium_info = []
    T = 0.3
    for i in range(len(partonlist)):
        medium_info.append([T, 0.0, 0.0, 0.0])
    return medium_info

class Jnu(object):
    def __init__(self, backend,eos_table,compile_options):
        self.cwd, cwf = os.path.split(__file__)
        # create the fPathOut directory if not exists
        self.backend = backend
        self.ctx = self.backend.ctx
        self.queue = self.backend.default_queue
        self.cfg = self.backend.config
        self.eos_table = eos_table
        path = self.cfg.fPathOut
        
        if not os.path.exists(path):
            os.makedirs(path)

        self.size = self.cfg.NX*self.cfg.NY*self.cfg.NZ
        self.tau = self.cfg.real(self.cfg.TAU0)


        self.trans_model = self.cfg.trans_model
        if self.trans_model=='LBT':
            self.lbt = LBT.PartonEvolution()
            self.Model_setting()

        self.gpu_defines = compile_options   
        self.__loadAndBuildCLPrg()

        mf = cl.mem_flags
        self.h_jetsrc = np.zeros((self.size,4), self.cfg.real)
        self.d_jetsrc = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.h_jetsrc)
        fpath_jet_inf = os.path.abspath(os.path.join(self.cfg.fPathOut, 'parton_info_%s_%s_%s.dat'%(self.cfg.cent,self.cfg.pthatmin,self.cfg.pthatmax)))
        files = {'fjet_inf':fpath_jet_inf}
        self.jet_init(readfile=True, **files)

        print('parton initialization finished')



    
    def jet_init(self, readfile=True, **files):
        init_partons = LBT.PartonInitialization()
        if readfile==True:
            init_partons.SetPath_2( self.cfg.fPathOut)
            if 'fjet_inf' in files.keys():
                fjet_inf = files['fjet_inf']
                init_partons.ReadPartonInfo(fjet_inf)                
            else:
                print('no full jet information from files, please check that')
                exit(-3)
        else:
            init_partons.SetPath_2( self.cfg.fPathOut)
            init_partons.SetPartonInfo()

        if self.trans_model =='LBT':
            self.lbt.PartonInitial(init_partons.par_vec)

    def Model_setting(self):
        if self.trans_model=='LBT':
            self.lbt.SetPath( self.cfg.fPathOut)
            t0 = float(self.tau)
            dt = float(self.cfg.DT)
            Ecut = float(self.cfg.Ecut)
            path_config = self.cfg.LBT_config
            path_table = self.cfg.LBT_table
            alphas = self.cfg.alphas
            self.lbt.Initialization(t0, dt, Ecut, alphas, path_config,path_table)

    def __loadAndBuildCLPrg(self):
        gpu_defines = list(self.gpu_defines)
        gpu_defines.append('-D {key}={value:f}f'.format(key='jet_r_gw', value=self.cfg.jet_r_gw))
        gpu_defines.append('-D {key}={value:f}f'.format(key='jet_eta_gw', value=self.cfg.jet_eta_gw))
        
        with open(os.path.join(self.cwd, '../kernel', 'kernel_jet_eloss.cl'), 'r') as f:
            prg_jetsrc = f.read()
            self.kernel_jet_eloss = cl.Program(self.ctx, prg_jetsrc).build(options=' '.join(gpu_defines))
    def get_medium_func(self,x3,d_ev,d_nb=None):
        mf = cl.mem_flags
        num = np.int32( len(x3))

        h_TVf = np.zeros( (num, 4), dtype=self.cfg.real )
        d_TVf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_TVf)
        d_x3 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x3)

      
        self.kernel_jet_eloss.get_TV(self.queue, (num,), None, d_x3, d_ev, d_nb, d_TVf, self.eos_table, num ).wait()
        #cl.enqueue_copy(self.queue, self.h_ev1, self.d_ev[1]).wait()
        #print('DEV:',self.h_ev1)
        cl.enqueue_copy(self.queue, h_TVf, d_TVf ).wait()
        return h_TVf

    def nextstep(self, tau,d_ev,d_nb=None):
        tau = float(tau)
        
        tx3 = []
        
        for parton in self.lbt.PosParton:
            tx3.append([parton.x, parton.y, parton.etas,0.0])#wenbin
        tx3 = np.array(tx3)
        tx3 = tx3.astype(self.cfg.real)
        medium_info = self.get_medium_func(tx3,d_ev,d_nb)
        #print (medium_info)
        for mi in medium_info:
            mi = np.array(mi).astype('float64')
            self.lbt.vec_T.append(mi[0])
            vf = LBT.double3()
            vf.vx, vf.vy, vf.vz = mi[1:4]
            self.lbt.vec_Vf.append(vf)
        self.lbt.AssignTV(tau)
        self.lbt.PartonCascade(tau)
        self.lbt.SortPartons()
        self.lbt.ClearLBTarrays()
        self.lbt.Propagation(tau)
        self.lbt.Output(1,tau)

        x3 = []
        p4 = []
        for parton in self.lbt.HdrParton:
            x,y,etas = parton.x, parton.y, parton.etas
        # #    print('x, y, etas:',x,y,etas)
            x3.append([x,y,etas,0.0])
            E, px, py, pz = parton.Ep
         #    print('tau',tau,'E',E)
            mt = ma.sqrt( E*E-pz*pz )
            Y = 0.5*( ma.log(E+pz) - ma.log(E-pz) )
            p4.append([ mt*ma.cosh(Y-etas), px, py, mt*ma.sinh(Y-etas)/(tau) ])
        for parton in self.lbt.NegParton:
            x,y,etas = parton.x, parton.y, parton.etas
            x3.append([x,y,etas,0.0])#wenbin
            E, px, py, pz = parton.Ep
            mt = ma.sqrt( E*E-pz*pz)
            Y = 0.5*( ma.log(E+pz) - ma.log(E-pz) )
            p4.append([ -mt*ma.cosh(Y-etas), -px, -py, -mt*ma.sinh(Y-etas)/(tau) ])
        if x3==[] or p4==[]:
            x3 = [[0.0,0.0,0.0]]
            p4 = [[0.0,0.0,0.0,0.0]]
        x3 = np.array(x3)
        p4 = np.array(p4)
        #print('p4:  ', p4)
        return self.Cal_jetsrc(tau, x3, p4)

    def update_src(self,d_src):
        #pass
        self.kernel_jet_eloss.update_src(self.queue, (self.cfg.NX,self.cfg.NY,self.cfg.NZ),None,self.d_jetsrc,d_src)


    def one_src(self, tau, X3, P4):
        x3 = np.array(X3)
        p4 = np.array(P4)
        return self.Cal_jetsrc(tau,x3,p4)

    def tc_inf(self):
        return self.lbt.Tc_Inf()

    def Cal_jetsrc(self, tau, x3, p4):
        x3 = x3.astype(self.cfg.real)
        p4 = p4.astype(self.cfg.real)

        mf = cl.mem_flags
        d_x3 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x3)
        d_p4 = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=p4)

        num = np.int32( len(x3))
        tau = self.cfg.real(tau)
        
        self.kernel_jet_eloss.jetsource(self.queue, (self.cfg.NX,self.cfg.NY,self.cfg.NZ), None, d_x3, d_p4, self.d_jetsrc, tau, num).wait()
        cl.enqueue_copy(self.queue, self.h_jetsrc, self.d_jetsrc).wait()
        #print('srchere=',self.d_jetsrc)
        return self.d_jetsrc

    def ev_to_host(self):
        cl.enqueue_copy(self.queue, self.h_jetsrc, self.d_jetsrc).wait()
        print(self.h_jetsrc[:,2].max())

    def plt_ed(self):
        cl.enqueue_copy(self.queue, self.h_jetsrc, self.d_jetsrc).wait()
        ed3d = (self.h_jetsrc[:,0]).reshape(self.cfg.NX, self.cfg.NY, self.cfg.NZ)
        extent=(-self.cfg.NX/2*self.cfg.DX,self.cfg.NX/2*self.cfg.DX,-self.cfg.NY/2*self.cfg.DY,self.cfg.NY/2*self.cfg.DY)
      


if __name__=='__main__':
    from config import cfg, write_config
    #import pandas as pd
    print('start ...')
    cfg.IEOS = 2
    cfg.NX = 201
    cfg.NY = 201
    cfg.NZ = 105
    cfg.DX = 0.16
    cfg.DY = 0.16
    cfg.DZ = 0.16
    cfg.DT = 0.02
    cfg.ntskip = 16
    cfg.nxskip = 2
    cfg.nyskip = 2
    cfg.nzskip = 2
    cfg.Eta_gw = 0.4
    cfg.ImpactParameter = 2.4
    cfg.ETAOS = 0.0
    cfg.TFRZ = 0.137

    cfg.Edmax = 55
    cfg.TAU0 = 0.2

    cfg.fPathOut = '../results'

    cfg.save_to_hdf5 = False

    cfg.BSZ = 64

    write_config(cfg)

    backend = OpenCLBackend(cfg, gpu_id=0)

    jnu = Jnu(backend,numid=1,fold_id=1)
    for t in np.arange(0.4, 5.0, 0.02):
        jnu.nextstep(t, GetTV)
