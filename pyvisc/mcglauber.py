#/usr/bin/env python


from __future__ import absolute_import, division, print_function
import numpy as np
import pyopencl as cl
from pyopencl import array
import os
import sys
from time import time
from scipy.integrate import quad
#from numba import jit
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
def intep_s_ed(cfg, eos, ev, nb):
    nb_l = np.int32 ( np.floor(nb/cfg.DNB ))
    nb_h = nb_l + 1
    ed_l = np.zeros_like(ev[:,0])
    ed_h = np.zeros_like(ev[:,0])
    for i in range( len(ev[:,0]) ):
        ed_l[i] = eos.f_Sed[nb_l[i]](ev[i,0])
        ed_h[i] = eos.f_Sed[nb_h[i]](ev[i,0])

    x = (nb - nb_l*cfg.DNB)/cfg.DNB
    ed = (1.0 - x)*ed_l + x*(ed_h)
    ev[:,0] = ed[:]
    return ev

def get_cross_section(snn):
    a =29.033455
    b=-0.054979147
    c=0.0605993633
    d=2.29076504
    s = snn**2
    return (a+np.log(s)*b+c*(np.log(s))**d)*0.1

def Woods_Saxon( R, a, r):

    return 1./ (1. + np.exp((r-R)/a))

def unitstep(edge,x):
    if x > edge:
        return 1.0
    else:
        return 0.0


class MCglauber(object):
    def __init__(self, visc):
        self.cwd, cwf = os.path.split(__file__)
        self.cfg  = visc.cfg
        self.visc = visc 
        self.queue = visc.queue
        self.ctx = visc.ctx
        self.eos = visc.eos
        self.set_configuration(self.cfg)
        self.gpu_defines = visc.compile_options
        self.__loadAndBuildCLPrg(self.ctx)
 
        self.d_ev1 = visc.d_ev[1]
        self.d_nb1 = visc.d_nb[1]

        mf = cl.mem_flags 
        
        self.h_phi2_re = np.zeros((self.cfg.NX, self.cfg.NY),dtype=self.cfg.real)
        self.h_phi2_im = np.zeros((self.cfg.NX, self.cfg.NY),dtype=self.cfg.real)
        self.h_phi2_wt = np.zeros((self.cfg.NX, self.cfg.NY),dtype=self.cfg.real)



        self.d_phi2_re = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_phi2_re.nbytes) 
        self.d_phi2_im = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_phi2_im.nbytes) 
        self.d_phi2_wt = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_phi2_wt.nbytes) 
        self.phi2 = np.zeros((4))
        self.e2 = np.zeros((4))

        self.h_Ta = np.zeros((self.cfg.NX*self.cfg.NY),dtype=self.cfg.real)
        self.h_Tb = np.zeros((self.cfg.NX*self.cfg.NY),dtype=self.cfg.real)
        
        self.d_Ta = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_Ta.nbytes) 
        self.d_Tb = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.h_Tb.nbytes) 

        self.read_centrality()
    
    def set_configuration(self,cfg):
        self.FACTOR = cfg.KFACTOR
        self.Eta_flat = cfg.Eta_flat
        self.Eta_gw = cfg.Eta_gw
        self.Eta_flat_c = cfg.Eta_flat_c
        self.Eta_gw_c = cfg.Eta_gw_c
        self.r_gw = cfg.r_gw
        self.SQRTS = cfg.SQRTS
        self.Eta0_nb = cfg.Eta0_nb
        self.Eta_nb_gw_p = cfg.Eta_nb_gw_p
        self.Eta_nb_gw_m = cfg.Eta_nb_gw_m
        self.cent = cfg.cent
        self.Hwn = cfg.Hwn

        
        if cfg.NucleusA == cfg.NucleusB and cfg.NucleusA=="Au":
            self.NumOfNucleons = 197
            self.Ra = 6.38
            self.Eta = 0.535
        elif cfg.NucleusA == cfg.NucleusB and cfg.NucleusA=="Pb":
            self.NumOfNucleons = 208
            self.Ra = 6.62
            self.Eta = 0.546
        else:
            print ("Please check collision system [Au+Au or Pb+Pb")
            exit(1)
        dec_num = np.modf(self.SQRTS)[0]
        if dec_num < 0.00001:
            self.system = self.cfg.NucleusA+self.cfg.NucleusB+str(int(np.floor(self.SQRTS)))
        else:
            self.system = self.cfg.NucleusA+self.cfg.NucleusB+str(self.SQRTS)
            self.system = self.system.replace(".","p")

    def read_centrality(self):
        if self.cent == "random":
            self.centrality = None
        else:
            self.centrality = np.loadtxt("{path}/ini/mcglauber/{system}centrality.dat".format(path=self.cwd,system=self.system)) 

    def sample_nucleon(self):
        self.Rmax = self.Ra + 10.0 * self.Eta
        r=0
        while True:
            r = self.Rmax * (np.random.random())**(1./3.) 
            if np.random.random() < Woods_Saxon(self.Ra, self.Eta, r):
                break
        phi = 2.0*np.pi*np.random.random()
        cos_theta = 2.0*np.random.random()-1.0
        sin_theta = np.sqrt(1.0 - cos_theta*cos_theta)
        
        x = r*sin_theta*np.cos(phi)
        y = r*sin_theta*np.sin(phi)
        z = r*cos_theta
        
        return x,y,z

    def sample_nucleus(self):
        self.tar_x = np.zeros((int(self.NumOfNucleons)),dtype=self.cfg.real)
        self.tar_y = np.zeros((int(self.NumOfNucleons)),dtype=self.cfg.real)
        self.tar_z = np.zeros((int(self.NumOfNucleons)),dtype=self.cfg.real)
        self.pro_x = np.zeros((int(self.NumOfNucleons)),dtype=self.cfg.real)
        self.pro_y = np.zeros((int(self.NumOfNucleons)),dtype=self.cfg.real)
        self.pro_z = np.zeros((int(self.NumOfNucleons)),dtype=self.cfg.real)
        self.tar_flag = np.zeros((int(self.NumOfNucleons)),dtype=int)
        self.pro_flag = np.zeros((int(self.NumOfNucleons)),dtype=int)
        self.Npart = 0
        self.Ncoll = 0
        self.Npart_tar = 0
        self.Npart_pro = 0
        self.xc = 0.0
        self.yc = 0.0
        self.b = 0.0

        for i in range(int(self.NumOfNucleons)):
            while(True):
                x,y,z = self.sample_nucleon()
                if np.min( (self.tar_x[:i+1]-x)**2+(self.tar_y[:i+1]-y)**2+(self.tar_z[:i+1]-z)**2 ) >0.81:
                    self.tar_x[i], self.tar_y[i], self.tar_z[i] = x,y,z
                    break     
        for i in range(int(self.NumOfNucleons)):

            while(True):
                x,y,z = self.sample_nucleon()
                if np.min( (self.pro_x[:i+1]-x)**2+(self.pro_y[:i+1]-y)**2+(self.pro_z[:i+1]-z)**2 ) >0.81:
                    self.pro_x[i], self.pro_y[i], self.pro_z[i] = x,y,z
                    break       




    def collision(self):
        
        self.b = 0.0
        Snn = get_cross_section(self.cfg.SQRTS)
        print (Snn)
        while (True):
            self.sample_nucleus()
            self.Ncoll=0
            self.Npart=0
            self.tar_flag[:]=0
            self.pro_flag[:]=0
            

            if self.cent == "random":
                #self.b = self.Rmax*2.0 * (np.random.random()**(1./2.))
                self.b = self.Rmax*1.6 * (np.random.random()**(1./2.))
            else:
                csplit = self.cent.split("_")
                clow = int(csplit[0])
                chi = int(csplit[-1])
                Rmax1 = self.centrality[chi,1]
                Rmin1 = self.centrality[clow,1]
                while(True):
                    self.b =self.Rmax*1.6 * (np.random.random()**(1./2.))
                    if self.b - Rmin1>1e-12 and Rmax1 - self.b > 1e-12:
                        break
    
            self.tar_x = self.tar_x - self.b/2.
            self.pro_x = self.pro_x + self.b/2.
            particle1 = np.array(list(zip(self.tar_x,self.tar_y)))
            particle2 = np.array(list(zip(self.pro_x,self.pro_y)))
            twocollision = cdist(particle1,particle2,'euclidean')
            collindex = np.argwhere(twocollision<np.sqrt(Snn/np.pi))
            self.Ncoll = len(collindex)
            self.Npart_tar = len(np.unique(collindex[:,0]))
            self.Npart_pro = len(np.unique(collindex[:,1]))
            self.Npart = self.Npart_tar + self.Npart_pro
            self.nbinaryx = 0.5*(self.tar_x[collindex[:,0]] + self.pro_x[collindex[:,1]])
            self.nbinaryy = 0.5*(self.tar_y[collindex[:,0]] + self.pro_y[collindex[:,1]])
            
            self.tar_x = self.tar_x[np.unique(collindex[:,0])]
            self.tar_y = self.tar_y[np.unique(collindex[:,0])]
            self.pro_x = self.pro_x[np.unique(collindex[:,1])]
            self.pro_y = self.pro_y[np.unique(collindex[:,1])]
            self.xc = ( np.sum(self.tar_x)+np.sum(self.pro_x)+ np.sum(self.nbinaryx) )/(self.Npart*1.0+1e-7+self.Ncoll)
            self.yc = ( np.sum(self.tar_y)+np.sum(self.pro_y)+ np.sum(self.nbinaryy) )/(self.Npart*1.0+1e-7+self.Ncoll)
            if self.Ncoll > 0:
                break
    

    def generate_s(self):
        NX, NY = self.cfg.NX, self.cfg.NY
        mf = cl.mem_flags 
        self.d_tar_x = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.tar_x.nbytes) 
        self.d_tar_y = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.tar_y.nbytes) 

        self.d_pro_x = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.pro_x.nbytes) 
        self.d_pro_y = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.pro_y.nbytes)

        self.d_nbinaryx = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.nbinaryx.nbytes) 
        self.d_nbinaryy = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.nbinaryy.nbytes) 

        cl.enqueue_copy(self.queue, self.d_tar_x,self.tar_x).wait()
        cl.enqueue_copy(self.queue, self.d_tar_y,self.tar_y).wait()
        cl.enqueue_copy(self.queue, self.d_pro_x,self.pro_x).wait()
        cl.enqueue_copy(self.queue, self.d_pro_y,self.pro_y).wait()

        cl.enqueue_copy(self.queue, self.d_nbinaryx,self.nbinaryx).wait()
        cl.enqueue_copy(self.queue, self.d_nbinaryy,self.nbinaryy).wait()

        
        self.kernel_mcglauber.generate_s(self.queue,(NX , NY), None, self.d_tar_x,\
            self.d_tar_y,self.d_pro_x,self.d_pro_y, np.int32(self.Npart_tar),\
            np.int32(self.Npart_pro),self.d_nbinaryx,self.d_nbinaryy,np.int32(self.Ncoll),\
            self.d_ev1,self.d_nb1,self.d_Ta,self.d_Tb,self.cfg.real(self.FACTOR),self.visc.eos_table).wait()

        cl.enqueue_copy(self.queue, self.visc.h_ev1,self.d_ev1).wait()
        cl.enqueue_copy(self.queue, self.visc.h_nb,self.d_nb1).wait()


        cl.enqueue_copy(self.queue, self.h_Ta,self.d_Ta).wait()
        cl.enqueue_copy(self.queue, self.h_Tb,self.d_Tb).wait()
       


    def generate_phi2(self):
 
        self.tar_x = self.tar_x - self.xc
        self.tar_y = self.tar_y - self.yc
        self.pro_x = self.pro_x - self.xc
        self.pro_y = self.pro_y - self.yc

        self.nbinaryx = self.nbinaryx - self.xc
        self.nbinaryy = self.nbinaryy - self.yc
        
        self.xc = ( np.sum(self.tar_x)+np.sum(self.pro_x)+ np.sum(self.nbinaryx) )/(self.Npart*1.0+1e-7+self.Ncoll)
        self.yc = ( np.sum(self.tar_y)+np.sum(self.pro_y)+ np.sum(self.nbinaryy) )/(self.Npart*1.0+1e-7+self.Ncoll)
        for nid in range(2,6):
            phi1 = np.arctan2(self.tar_y,self.tar_x)
            r1 = np.sqrt( self.tar_y**2+self.tar_x**2)
            phi2 = np.arctan2(self.pro_y,self.pro_x)
            r2 = np.sqrt( self.pro_y**2+self.pro_x**2)

            phi3 = np.arctan2(self.nbinaryy,self.nbinaryx)
            r3 = np.sqrt( self.nbinaryy**2+self.nbinaryx**2)


            rncos = ( np.sum( (r1**nid)*np.cos(nid*phi1)) + np.sum( (r2**nid)*np.cos(nid*phi2) ) + np.sum( (r3**nid)*np.cos(nid*phi3) ) )/(self.Npart+self.Ncoll)
            rnsin = ( np.sum( (r1**nid)*np.sin(nid*phi1)) + np.sum( (r2**nid)*np.sin(nid*phi2) ) + np.sum( (r3**nid)*np.sin(nid*phi3) ))/(self.Npart+self.Ncoll)
            rn = ( np.sum(r1**nid) + np.sum(r2**nid) + np.sum( (r3**nid) ) ) /(self.Npart+self.Ncoll)
            self.e2[nid-2] = np.sqrt(rncos**2 + rnsin**2)/rn
            self.phi2[nid-2] = np.arctan2(rnsin,rncos)/nid + np.pi/nid

        


    def ebe_event(self,cent=None):
        iniinfo = np.zeros((9))
        #self.sample_nucleus()
        self.collision()
        self.generate_s()
        self.generate_phi2()
        iniinfo[0] =   self.Npart 
        iniinfo[1:5] = self.e2 
        iniinfo[5:9] = self.phi2 
        np.savetxt(os.path.join(self.cfg.fPathOut,"Glauber.dat"),iniinfo.reshape((1,9)),header="#npart e2 e3 e4 e5 phi2 phi3 phi4 phi5")
        np.savetxt(os.path.join(self.cfg.fPathOut,"Ta_and_Tb.dat"),np.array(list(zip(self.h_Ta,self.h_Tb))),header="#Ta Tb [size: %s * %s]"%(self.cfg.NX,self.cfg.NY))
        
    def random_event(self):
        ncolllist=[]
        npartlist=[]
        nblist=[]
        for i in tqdm(range(100000)):
            self.collision()
            npartlist.append( self.Npart )
            ncolllist.append( self.Ncoll )
            nblist.append( self.b )
        np.savetxt(os.path.join(self.cfg.fPathOut,"npart_ncoll_b.dat"),np.array(list(zip(npartlist,ncolllist,nblist))))
        


    def rotate(self):
        tem_tar_x = self.tar_x[:]
        tem_tar_y = self.tar_y[:]
        tem_pro_x = self.pro_x[:]
        tem_pro_y = self.pro_y[:]

        tem_nbinaryx = self.nbinaryx[:]
        tem_nbinaryy = self.nbinaryy[:]
        
        self.tar_x = tem_tar_x *np.cos(self.phi2[0]) + tem_tar_y * np.sin(self.phi2[0]) 
        self.tar_y = - tem_tar_x *np.sin(self.phi2[0]) + tem_tar_y * np.cos(self.phi2[0]) 
        self.pro_x = tem_pro_x *np.cos(self.phi2[0]) + tem_pro_y * np.sin(self.phi2[0]) 
        self.pro_y = - tem_pro_x *np.sin(self.phi2[0]) + tem_pro_y * np.cos(self.phi2[0])
        
        self.nbinaryx = tem_nbinaryx *np.cos(self.phi2[0]) + tem_nbinaryy * np.sin(self.phi2[0]) 
        self.nbinaryy = - tem_nbinaryx *np.sin(self.phi2[0]) + tem_nbinaryy * np.cos(self.phi2[0])
        
        
        nid = 2
        phi1 = np.arctan2(self.tar_y,self.tar_x)
        r1 = np.sqrt( self.tar_y**2+self.tar_x**2)
        phi2 = np.arctan2(self.pro_y,self.pro_x)
        r2 = np.sqrt( self.pro_y**2+self.pro_x**2)
        phi3 = np.arctan2(self.nbinaryy,self.nbinaryx)
        r3 = np.sqrt( self.nbinaryy**2+self.nbinaryx**2)
        rncos = ( np.sum( (r1**nid)*np.cos(nid*phi1)) + np.sum( (r2**nid)*np.cos(nid*phi2) ) + np.sum( (r3**nid)*np.cos(nid*phi3) ) )/(self.Npart+self.Ncoll)
        rnsin = ( np.sum( (r1**nid)*np.sin(nid*phi1)) + np.sum( (r2**nid)*np.sin(nid*phi2) ) + np.sum( (r3**nid)*np.sin(nid*phi3) ))/(self.Npart+self.Ncoll)
        rn = ( np.sum(r1**nid) + np.sum(r2**nid) + np.sum( (r3**nid) ) ) /(self.Npart+self.Ncoll)
        self.e2[nid-2] = np.sqrt(rncos**2 + rnsin**2)/rn
        self.phi2[nid-2] = np.arctan2(rnsin,rncos)/nid + np.pi/nid

        #print ("after: ",self.e2[0],self.phi2[0])


    def ave_event(self,Nave = 1000):
        iniinfo = np.zeros((Nave,9))
        h_ave_ev = np.zeros_like(self.visc.h_ev1)
        h_ave_nb = np.zeros_like(self.visc.h_nb)
        h_ave_Ta = np.zeros_like(self.h_Ta)
        h_ave_Tb = np.zeros_like(self.h_Tb)
        for i in range(Nave):
            print ("Event ", i, " ...")
            self.collision()
            self.generate_phi2()
            self.rotate()
            self.generate_s()
            h_ave_ev += self.visc.h_ev1  
            h_ave_nb += self.visc.h_nb

            h_ave_Ta += self.h_Ta  
            h_ave_Tb += self.h_Tb

            iniinfo[i,0] = iniinfo[i,0] + self.Npart 
            iniinfo[i,1:5] = iniinfo[i,1:5] + self.e2 
            iniinfo[i,5:9] = iniinfo[i,5:9] + self.phi2 
        self.visc.h_ev1 = h_ave_ev / (Nave*1.0)
        self.visc.h_nb  = h_ave_nb / (Nave*1.0)

        h_ave_Ta = h_ave_Ta / (Nave*1.0)
        h_ave_Tb = h_ave_Tb / (Nave*1.0)

        cl.enqueue_copy(self.queue, self.d_ev1,self.visc.h_ev1).wait()
        cl.enqueue_copy(self.queue, self.d_nb1,self.visc.h_nb).wait()

        print("<Npart>: ", np.mean(iniinfo[:,0]), "After smearing <Npart>",np.sum(self.visc.h_nb)*self.cfg.DX*self.cfg.DY*self.cfg.DZ*self.cfg.TAU0)
        np.savetxt(os.path.join(self.cfg.fPathOut,"Glauber.dat"),iniinfo,header="#npart e2 e3 e4 e5 phi2 phi3 phi4 phi5")
        np.savetxt(os.path.join(self.cfg.fPathOut,"Ta_and_Tb.dat"),np.array(list(zip(h_ave_Ta,h_ave_Ta))),header="#Ta Tb [size: %s * %s]"%(self.cfg.NX,self.cfg.NY))
        #np.savetxt(os.path.join(self.cfg.fPathOut,"ev.dat"),self.visc.h_ev1)
        #np.savetxt(os.path.join(self.cfg.fPathOut,"nb.dat"),self.visc.h_nb)
               

    def __loadAndBuildCLPrg(self, ctx):
        #load and build *.cl programs with compile self.gpu_defines
        glauber_defines = list(self.gpu_defines)
        glauber_defines.append('-D {key}={value:f}f'.format(key='NumOfNucleons', value=self.NumOfNucleons))
        glauber_defines.append('-D {key}={value:f}f'.format(key='SQRTS', value=self.SQRTS))
        glauber_defines.append('-D {key}={value:f}f'.format(key='Eta0_s',value=self.Eta_flat/2.0))
        glauber_defines.append('-D {key}={value:f}f'.format(key='Eta_gw',value=self.Eta_gw))
        glauber_defines.append('-D {key}={value:f}f'.format(key='Eta0_nb',value=self.Eta0_nb))
        glauber_defines.append('-D {key}={value:f}f'.format(key='Sigma0_p',value=self.Eta_nb_gw_p))
        glauber_defines.append('-D {key}={value:f}f'.format(key='Sigma0_m',value=self.Eta_nb_gw_m))
        glauber_defines.append('-D {key}={value:f}f'.format(key='sigma_r',value=self.r_gw))
        glauber_defines.append('-D {key}={value:f}f'.format(key='Eta_flat_c',value=self.Eta_flat_c))
        glauber_defines.append('-D {key}={value:f}f'.format(key='Eta_gw_c',value=self.Eta_gw_c))
        glauber_defines.append('-D {key}={value:f}f'.format(key='Hwn',value=self.Hwn))
        
        with open(os.path.join(self.cwd, 'kernel', 'kernel_mcglauber.cl'), 'r') as f:
            prg_src = f.read()
            self.kernel_mcglauber = cl.Program(ctx, prg_src).build(
                                             options=' '.join(glauber_defines))
        
