#/usr/bin/env python
#Original Copyright (c) 2014  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from time import time
import pyopencl as cl
def minmod(a, b):
    if a>0 and b>0:
        return min(a, b)
    elif a<0 and b<0:
        return max(a, b)
    else:
        return 0

class Eos(object):
    '''create eos table for hydrodynamic simulation;
    the (ed, pr, T, s) is stored in image2d buffer
    for fast linear interpolation'''
    def __init__(self, eos_type='ideal_gas'):
        self.eos_type = eos_type
        if eos_type.upper() == 'IDEAL_GAS':
            self.ideal_gas()
        elif eos_type.upper()  == 'LATTICE_PCE165':
            self.lattice_pce165()
        elif eos_type.upper()  =='LATTICE_PCE150':
            self.lattice_pce150()
        elif eos_type.upper()  == 'LATTICE_WB':
            self.lattice_ce()
        elif eos_type.upper()  == 'PURE_GAUGE':
            self.pure_su3()
        elif eos_type.upper()  == 'FIRST_ORDER':
            self.eosq()
        elif eos_type.upper()  == 'IDEAL_GAS_BARYON':
            self.ideal_gas_baryon()
        elif eos_type.upper()  == 'NEOSB':
            self.neosB()
        elif eos_type.upper() == 'NEOSBQS' :
            self.neosBQS()
        elif eos_type.upper() == 'CHIRAL' :
            self.chiral_model()
        elif eos_type.upper()  == 'HARDON_GAS':
            self.hardon_gas()
        elif eos_type.upper()  == 'EOSQ':
            self.EOSQ()
        elif eos_type.upper()  == 'NJL_MODEL':
            self.NJL()
        elif eos_type.upper()  == 'HOTQCD2014':
            self.hotqcd2014()



    def ideal_gas(self):
        '''ideal gas eos, P=ed/3 '''
        hbarc = 0.1973269
        dof = 169.0/4.0
        coef = np.pi*np.pi/30.0
        self.f_P = lambda ed: np.array(ed)/3.0 #GeV*fm^-3
        self.f_T = lambda ed: hbarc*(1.0/(dof*coef)*np.array(ed)/hbarc)**0.25 + 1.0E-10 #GeV
        self.f_S = lambda ed: (np.array(ed) + self.f_P(ed))/self.f_T(ed) #fm^-3
        self.f_ed = lambda T: dof*coef*hbarc*(np.array(T)/hbarc)**4.0  #GeV*fm^-3
        self.f_cs2 = lambda ed: 1.0/3.0 * np.ones_like(ed)      
        self.ed = np.linspace(0, 1999.99, 200000,dtype=np.float32)
        self.pr = self.f_P(self.ed)
        self.T = self.f_T(self.ed)
        self.s = self.f_S(self.ed)
        self.cs2 = self.f_cs2(self.ed)

        self.ed_start = 0.0
        self.ed_step = 0.01
        self.num_of_ed = 200000
        
        self.cs2 = self.cs2.astype(np.float32)
        self.pr = self.pr.astype(np.float32)
        self.T = self.T.astype(np.float32)
        self.s = self.s.astype(np.float32)

        self.h_eos = np.array(([]),np.float32)
        self.h_eos = np.append(self.h_eos,self.cs2)
        self.h_eos = np.append(self.h_eos,self.pr)
        self.h_eos = np.append(self.h_eos,self.T)
        self.h_eos = np.append(self.h_eos,self.s)

    def eosq(self):
        import eosq
        self.ed = eosq.ed
        self.pr = eosq.pr
        self.T = eosq.T
        self.s = eosq.s
        self.ed_start = eosq.ed_start
        self.ed_step = eosq.ed_step
        self.num_of_ed = eosq.num_ed
        #get cs2 using dp/de
        self.eos_func_from_interp1d()
        self.f_P = eosq.f_P
        self.f_T = eosq.f_T
        self.f_S = eosq.f_S
        self.f_ed = eosq.f_ed
        

        self.cs2 = self.cs2.astype(np.float32)
        self.pr = self.pr.astype(np.float32)
        self.T = self.T.astype(np.float32)
        self.s = self.s.astype(np.float32)


        self.h_eos = np.array(([]),np.float32)
        self.h_eos = np.append(self.h_eos,self.cs2)
        self.h_eos = np.append(self.h_eos,self.pr)
        self.h_eos = np.append(self.h_eos,self.T)
        self.h_eos = np.append(self.h_eos,self.s)




    def eos_func_from_interp1d(self, order=1):
        # construct interpolation functions
        self.f_ed = InterpolatedUnivariateSpline(self.T, self.ed, k=order, ext=0)
        self.f_T = InterpolatedUnivariateSpline(self.ed, self.T, k=order, ext=0)
        self.f_P = InterpolatedUnivariateSpline(self.ed, self.pr, k=order, ext=0)
        self.f_S = InterpolatedUnivariateSpline(self.ed, self.s, k=order, ext=0)
        # calc the speed of sound square
        self.cs2 = np.gradient(self.pr, self.ed_step)
        # remove high gradient in dp/de function
        for i, cs2_ in enumerate(self.cs2):
            if abs(cs2_) > 0.34:
                a = self.cs2[i-1]
                b = cs2_
                c = self.cs2[i+1]
                self.cs2[i] = minmod(a, minmod(b, c))
        mask = self.ed >= 30.
        ed_mask = self.ed[mask]

        cs2_mask = self.cs2[mask]
        def exp_func(x, a, b, c):
            '''fit cs2 at ed >30 with a smooth curve'''
            return a / (np.exp(b/x) + c)
        popt, pcov = curve_fit(exp_func, ed_mask, cs2_mask)
        self.cs2[mask] = exp_func(ed_mask, *popt)


    def lattice_pce165(self):
        import os
        cwd, cwf = os.path.split(__file__)
        pce = np.loadtxt(os.path.join(cwd, 'eos_table/s95p-PCE165-v0/EOS_PST.dat'))
        self.ed = np.insert(0.5*(pce[1:, 0] + pce[:-1, 0]), 0, 0.0)
        self.pr = np.insert(0.5*(pce[1:, 1] + pce[:-1, 1]), 0, 0.0)
        self.s = np.insert(0.5*(pce[1:, 2] + pce[:-1, 2]), 0, 0.0)
        self.T = np.insert(0.5*(pce[1:, 3] + pce[:-1, 3]), 0, 0.0)
        self.ed_start = 0.0
        self.ed_step = 0.002
        self.num_of_ed = 155500
        self.eos_func_from_interp1d()

        self.cs2 = self.cs2.astype(np.float32)
        self.pr = self.pr.astype(np.float32)
        self.T = self.T.astype(np.float32)
        self.s = self.s.astype(np.float32)
        
        self.h_eos = np.array(([]),np.float32)
        self.h_eos = np.append(self.h_eos,self.cs2)
        self.h_eos = np.append(self.h_eos,self.pr)
        self.h_eos = np.append(self.h_eos,self.T)
        self.h_eos = np.append(self.h_eos,self.s)




    def lattice_pce150(self):
        import os
        cwd, cwf = os.path.split(__file__)
        pce = np.loadtxt(os.path.join(cwd, 'eos_table/s95p-PCE-v1/EOS_PST.dat'))
        self.ed = np.insert(0.5*(pce[1:, 0] + pce[:-1, 0]), 0, 0.0)
        self.pr = np.insert(0.5*(pce[1:, 1] + pce[:-1, 1]), 0, 0.0)
        self.s = np.insert(0.5*(pce[1:, 2] + pce[:-1, 2]), 0, 0.0)
        self.T = np.insert(0.5*(pce[1:, 3] + pce[:-1, 3]), 0, 0.0)
        self.ed_start = 0.0
        self.ed_step = 0.002
        self.num_of_ed = 155500
        self.eos_func_from_interp1d()


        self.cs2 = self.cs2.astype(np.float32)
        self.pr = self.pr.astype(np.float32)
        self.T = self.T.astype(np.float32)
        self.s = self.s.astype(np.float32)

        self.h_eos = np.array(([]),np.float32)
        self.h_eos = np.append(self.h_eos,self.cs2)
        self.h_eos = np.append(self.h_eos,self.pr)
        self.h_eos = np.append(self.h_eos,self.T)
        self.h_eos = np.append(self.h_eos,self.s)


    def hotqcd2014(self):
        import os
        cwd, cwf = os.path.split(__file__)
        hotepst = np.loadtxt(os.path.join(cwd, 'eos_table/hotqcd/HotQCD.dat'))
        self.ed =np.insert(hotepst[:,0],0, 0.0)
        self.pr = np.insert(hotepst[:,1],0, 0.0)
        self.s = np.insert(hotepst[:,2],0, 0.0)
        self.T = np.insert(hotepst[:,3],0, 0.0)
        self.ed_start = 0.0
        self.ed_step = 0.002
        self.num_of_ed = 155500
        self.eos_func_from_interp1d()
        
        
        self.cs2 = self.cs2.astype(np.float32)
        self.pr = self.pr.astype(np.float32)
        self.T = self.T.astype(np.float32)
        self.s = self.s.astype(np.float32)

        self.h_eos = np.array(([]),np.float32)
        self.h_eos = np.append(self.h_eos,self.cs2)
        self.h_eos = np.append(self.h_eos,self.pr)
        self.h_eos = np.append(self.h_eos,self.T)
        self.h_eos = np.append(self.h_eos,self.s)



    def ideal_gas_baryon(self):
        import os
        hbarc=0.1973269
        # in order to impove the read speed, use pandas
        start = time()
        cwd, cwf = os.path.split(__file__)
        pce = np.zeros((140,155500,5))
        self.ed = np.zeros((140,155500))
        self.pr = np.zeros((140,155500))
        self.s = np.zeros((140,155500))
        self.T = np.zeros((140,155500))
        self.mu = np.zeros((140,155500))
        self.cs2 = np.zeros((140,155500))

        for i in range(0,140,1):
            data = pd.read_csv(os.path.join(cwd,'eos_table/ideal_gas_baryon/table3d_%d.dat'%i),sep=' ',dtype=float)
            pce[i,:,:] = data.values
            self.ed[i] = np.insert(0.5*(pce[i,1:, 0]+pce[i,:-1, 0]), 0 ,0.0)
            self.T[i] = np.insert(0.5*(pce[i,1:, 1]+pce[i,:-1, 1]), 0 ,0.0)
            self.mu[i] = np.insert(0.5*(pce[i,1:, 2]+pce[i,:-1, 2]), 0 ,0.0)
            self.pr[i] = np.insert(0.5*(pce[i,1:, 3]+pce[i,:-1, 3]), 0 ,0.0)
            self.s[i] = np.insert(0.5*(pce[i,1:, 4]+pce[i,:-1, 4]), 0 ,0.0)
            self.cs2[i,:] = 0.333333
        


        self.ed_start = 0.0
        self.ed_step = 0.002
        self.num_of_ed = 155500
        self.nb_start = 0.0
        self.nb_step = 0.1
        self.num_of_nb = 140
       
        elapsed = (time())-start
        print ("Read Eos data finished cost : %s s"%elapsed)
        

    def hardon_gas(self):
        import os
        hbarc=0.1973269
        start = time()
        cwd, cwf = os.path.split(__file__)
        self.ed_start = 0.0
        self.ed_step = 0.01
        self.num_of_ed = 101
        self.nb_start = 0.0
        self.nb_step = 0.01
        self.num_of_nb = 101

        eos_table =  np.loadtxt(os.path.join(cwd,'eos_table/hadgas_eos_with_p.dat'))
        eos_zero = eos_table[eos_table[:,1]<self.nb_step,:]

        ed = eos_zero[:,0]
        pr = eos_zero[:,3]
        T = eos_zero[:,2]
        s = (ed+pr)/(T+1e-8)

        self.f_ed = InterpolatedUnivariateSpline(T, ed, k=1, ext=0)
        self.f_T = InterpolatedUnivariateSpline(ed, T, k=1, ext=0)
        self.f_P = InterpolatedUnivariateSpline(ed, pr, k=1, ext=0)
        self.f_S = InterpolatedUnivariateSpline(ed, s, k=1, ext=0)

        self.h_eos = eos_table[:,[2,3,4]]

        self.h_eos = np.loadtxt(os.path.join(cwd,'eos_table/hadgas_eos_with_p.dat'),usecols=(2,3,4),dtype=np.float32)
        
        elapsed = (time())-start



        print ("Read Eos data finished cost : %s s"%elapsed)
    def neosB(self):
        import os
        hbarc=0.1973269
        start = time()
        cwd, cwf = os.path.split(__file__)
        
        self.h_eos = np.array(([]),np.float32)
        self.ed_length = np.array([13, 20, 31, 42, 200, 200, 200],dtype = np.int32)
        self.nb_length = np.array([500, 300, 180, 250, 350, 250, 200],dtype = np.int32)
        for i in range(1,8):

            table_p   = np.loadtxt(os.path.join(cwd, "./eos_table/neosB/neos%d_p.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_T   = np.loadtxt(os.path.join(cwd,"./eos_table/neosB/neos%d_t.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_mub = np.loadtxt(os.path.join(cwd,"./eos_table/neosB/neos%d_mub.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_p = table_p.T.flatten()
            table_T = table_T.T.flatten()
            table_mub = table_mub.T.flatten()
            
            self.h_eos = np.append(self.h_eos,table_p)   #GeV/fm^3
            self.h_eos = np.append(self.h_eos,table_T)   #GeV
            self.h_eos = np.append(self.h_eos,table_mub) #GeV
        self.h_eos = self.h_eos/hbarc 


        elapsed = (time())-start
        print ("Read neosB Eos data finished cost : %s s"%elapsed)

    def neosBQS(self):
        import os
        hbarc=0.1973269
        start = time()
        cwd, cwf = os.path.split(__file__)
        
        self.h_eos = np.array(([]),np.float32)
        self.ed_length = np.array([13, 20, 31, 42, 200, 200, 200],dtype = np.int32)
        self.nb_length = np.array([500, 300, 180, 250, 350, 250, 200],dtype = np.int32)
        for i in range(1,8):
            table_p   = np.loadtxt(os.path.join(cwd, "./eos_table/neosBQS/neos%dqs_p.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_T   = np.loadtxt(os.path.join(cwd,"./eos_table/neosBQS/neos%dqs_t.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_mub = np.loadtxt(os.path.join(cwd,"./eos_table/neosBQS/neos%dqs_mub.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_mus = np.loadtxt(os.path.join(cwd,"./eos_table/neosBQS/neos%dqs_mus.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_muc = np.loadtxt(os.path.join(cwd,"./eos_table/neosBQS/neos%dqs_muq.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_p = table_p.T.flatten()
            table_T = table_T.T.flatten()
            table_mub = table_mub.T.flatten()
            table_mus = table_mus.T.flatten()
            table_muc = table_muc.T.flatten()
            
            self.h_eos = np.append(self.h_eos,table_p)   #GeV/fm^3
            self.h_eos = np.append(self.h_eos,table_T)   #GeV
            self.h_eos = np.append(self.h_eos,table_mub) #GeV
            self.h_eos = np.append(self.h_eos,table_mus) #GeV
            self.h_eos = np.append(self.h_eos,table_muc) #GeV
        self.h_eos = self.h_eos/hbarc 
        

        

        elapsed = (time())-start
        print ("Read Eos data finished cost : %s s"%elapsed)
    
    def NJL(self):
        import os
        hbarc=0.1973269
        start = time()
        cwd, cwf = os.path.split(__file__)
        
        self.h_eos = np.array(([]),np.float32)
        self.ed_length = np.array([13, 20, 31, 42, 200, 200],dtype = np.int32)
        self.nb_length = np.array([500, 300, 180, 250, 350, 250],dtype = np.int32)
        for i in range(1,7):
            table_p   = np.loadtxt(os.path.join(cwd, "./eos_table/NJL_model/neos%dqs_p.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_T   = np.loadtxt(os.path.join(cwd,"./eos_table/NJL_model/neos%dqs_t.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_mub = np.loadtxt(os.path.join(cwd,"./eos_table/NJL_model/neos%dqs_mub.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_p = table_p.T.flatten()
            table_T = table_T.T.flatten()
            table_mub = table_mub.T.flatten()
            
            self.h_eos = np.append(self.h_eos,table_p)   #GeV/fm^3
            self.h_eos = np.append(self.h_eos,table_T)   #GeV
            self.h_eos = np.append(self.h_eos,table_mub) #GeV
        self.h_eos = self.h_eos/hbarc 

         
        self.ed_start = 0.0
        self.ed_step = 0.02
        self.num_of_ed = 15550
        self.nb_start = 0.0
        self.nb_step = 0.005
        self.num_of_nb = 1500

        elapsed = (time())-start
        print ("Read Eos data finished cost : %s s"%elapsed)

    def EOSQ(self):
        import os
        hbarc=0.1973269
        start = time()
        cwd, cwf = os.path.split(__file__)
        
        self.h_eos = np.array(([]),np.float32)
        self.ed_length = np.array([46, 250],dtype = np.int32)
        self.nb_length = np.array([150, 250],dtype = np.int32)
        for i in range(1,3):

            table_p   = np.loadtxt(os.path.join(cwd, "./eos_table/EOSQ_mb/aa%d_p.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_T   = np.loadtxt(os.path.join(cwd,"./eos_table/EOSQ_mb/aa%d_t.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            table_mub = np.loadtxt(os.path.join(cwd,"./eos_table/EOSQ_mb/aa%d_mb.dat"%i),skiprows =2, dtype = np.float32).reshape(self.ed_length[i-1],self.nb_length[i-1])
            
            table_p = table_p.T.flatten()
            table_T = table_T.T.flatten()
            table_mub = table_mub.T.flatten()
            
            self.h_eos = np.append(self.h_eos,table_p)   #GeV/fm^3
            self.h_eos = np.append(self.h_eos,table_T)   #GeV
            self.h_eos = np.append(self.h_eos,table_mub) #GeV
        self.h_eos = self.h_eos/hbarc 

         
        self.ed_start = 0.0
        self.ed_step = 0.02
        self.num_of_ed = 15550
        self.nb_start = 0.0
        self.nb_step = 0.005
        self.num_of_nb = 1500

        elapsed = (time())-start
        print ("Read Eos data finished cost : %s s"%elapsed)




    def chiral_model(self):
        import os
        hbarc=0.1973269
        start = time()
        cwd, cwf = os.path.split(__file__)
        
        self.h_eos = np.array(([]),np.float32)
        
        eossmall = np.loadtxt(os.path.join(cwd,"./eos_table/chiral/chiralsmall.dat"),dtype = np.float32)
        eosbig = np.loadtxt(os.path.join(cwd,"./eos_table/chiral/chiraleos.dat"),dtype = np.float32)
        
        eossmall[:,0] = eossmall[:,0]/1000.0 #T [GeV]
        eossmall[:,1] = eossmall[:,1]/1000.0 #mub[GeV]
        eossmall[:,2] = eossmall[:,2]*0.146  #ed [GeV/fm3]
        eossmall[:,3] = eossmall[:,3]*0.146  #pt [GeV/fm3]
        eossmall[:,4] = eossmall[:,4]*0.15   #nb [GeV/fm3]
        eossmall[:,5] = eossmall[:,5]*0.15   #s  [1/fm3]
        eossmall[:,6] = eossmall[:,6]/1000.0 #mubs [GeV]
        
        self.edmaxsmall = np.max(eossmall[:,2])
        self.edminsmall = np.min(eossmall[:,2])
        
        self.nbmaxsmall = np.max(eossmall[:,4])
        self.nbminsmall = np.min(eossmall[:,4])

        self.nedsmall = 201
        self.nnbsmall = 201

        eossmall = np.delete(eossmall,[2,4,7],axis=1)

        
        eosbig[:,0] = eosbig[:,0]/1000.0 #T
        eosbig[:,1] = eosbig[:,1]/1000.0 #mub
        eosbig[:,2] = eosbig[:,2]*0.146  #ed
        eosbig[:,3] = eosbig[:,3]*0.146  #pt
        eosbig[:,4] = eosbig[:,4]*0.15   #nb
        eosbig[:,5] = eosbig[:,5]*0.15   #s
        eosbig[:,6] = eosbig[:,6]/1000.0 #mubs
        
        self.edmaxbig = np.max(eosbig[:,2])
        self.edminbig = np.min(eosbig[:,2])
        
        self.nbmaxbig = np.max(eosbig[:,4])
        self.nbminbig = np.min(eosbig[:,4])

        self.nedbig = 2001
        self.nnbbig = 401

        eosbig = np.delete(eosbig,[2,4,7],axis=1)

        self.h_eos=np.append(eossmall.flatten(), eosbig.flatten())


        elapsed = (time())-start
        print ("Read Eos data finished cost : %s s"%elapsed)



    def lattice_ce(self):
        '''lattice qcd EOS from wuppertal budapest group
        2014 with chemical equilibrium EOS'''
        import wb as  wb
        self.ed = wb.ed
        self.pr = wb.pr
        self.T = wb.T
        self.s = (self.ed + self.pr)/(self.T + 1.0E-10)
        self.ed_start = wb.ed_start
        self.ed_step = wb.ed_step
        self.num_of_ed = wb.num_ed
        self.eos_func_from_interp1d()

        self.cs2 = self.cs2.astype(np.float32)
        self.pr = self.pr.astype(np.float32)
        self.T = self.T.astype(np.float32)
        self.s = self.s.astype(np.float32)

        self.h_eos = np.array(([]),np.float32)
        self.h_eos = np.append(self.h_eos,self.cs2)
        self.h_eos = np.append(self.h_eos,self.pr)
        self.h_eos = np.append(self.h_eos,self.T)
        self.h_eos = np.append(self.h_eos,self.s)

    def lattice_ce_mod(self):
        '''lattice qcd EOS from wuppertal budapest group
        2014 with chemical equilibrium EOS
        use T=np.linspace(0.03, 1.13, 1999) to create the table,
        notice that ed_step is not constant'''
        import wb_mod as wb
        self.ed = wb.ed
        self.pr = wb.pr
        self.T = wb.T
        self.s = (self.ed + self.pr)/(self.T + 1.0E-10)
        self.ed_start = wb.ed_start
        self.ed_step = wb.ed_step
        self.num_of_ed = wb.num_ed
        self.eos_func_from_interp1d()


    def pure_su3(self):
        '''pure su3 gauge EOS'''
        import glueball
        self.ed = glueball.ed
        self.pr = glueball.pr
        self.T = glueball.T
        self.s = (self.ed + self.pr)/(self.T + 1.0E-10)
        self.ed_start = glueball.ed_start
        self.ed_step = glueball.ed_step
        self.num_of_ed = glueball.num_ed
        self.eos_func_from_interp1d()


        self.cs2 = self.cs2.astype(np.float32)
        self.pr = self.pr.astype(np.float32)
        self.T = self.T.astype(np.float32)
        self.s = self.s.astype(np.float32)


        self.h_eos = np.array(([]),np.float32)
        self.h_eos = np.append(self.h_eos,self.cs2)
        self.h_eos = np.append(self.h_eos,self.pr)
        self.h_eos = np.append(self.h_eos,self.T)
        self.h_eos = np.append(self.h_eos,self.s)


    def create_table(self, ctx, compile_options):
        '''store the eos (ed, pr, T, s) in image2d_t table for fast
        linear interpolation,
        add some information to compile_options for EOS table'''
        
        mf = cl.mem_flags
        self.d_eos = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h_eos)

        
        compile_options.append('-D EOS_ED_START={value}f'.format(
                                                 value=self.ed_start))
        compile_options.append('-D EOS_ED_STEP={value}f'.format(
                                                 value=self.ed_step))
        compile_options.append('-D EOS_NUM_ED={value}'.format(
                                                 value=self.num_of_ed))
        compile_options.append('-D {value}'.format(
                                                 value=self.eos_type.upper()))
        self.compile_options = compile_options
        return  self.d_eos


   



##########################################################################



    def create_table_nb(self, ctx, compile_options, nrow=200, ncol=1000):
        '''store the eos (ed, pr, T, s) in image2d_t table for fast
        linear interpolation,
        add some information to compile_options for EOS table'''
        
        mf = cl.mem_flags
        self.size_eos= self.num_of_nb*nrow*ncol
        
        self.h_eos = np.zeros((5*self.size_eos),np.float32)
        
        for i in range(0,self.num_of_nb):
            l = 0
            start_id = l*self.num_of_nb*nrow*ncol + i*nrow*ncol
            end_id = l*self.num_of_nb*nrow*ncol + (i+1)*nrow*ncol
            self.h_eos[start_id:end_id] = self.cs2[i,:]
          

            l = 1
            start_id = l*self.num_of_nb*nrow*ncol + i*nrow*ncol
            end_id = l*self.num_of_nb*nrow*ncol + (i+1)*nrow*ncol
            self.h_eos[start_id:end_id] = self.pr[i,:]

            l = 2 
            start_id = l*self.num_of_nb*nrow*ncol + i*nrow*ncol
            end_id = l*self.num_of_nb*nrow*ncol + (i+1)*nrow*ncol
            self.h_eos[start_id:end_id] = self.T[i,:]


            l = 3 
            start_id = l*self.num_of_nb*nrow*ncol + i*nrow*ncol
            end_id = l*self.num_of_nb*nrow*ncol + (i+1)*nrow*ncol
            self.h_eos[start_id:end_id] = self.mu[i,:]

            
            l = 4 
            start_id = l*self.num_of_nb*nrow*ncol + i*nrow*ncol
            end_id = l*self.num_of_nb*nrow*ncol + (i+1)*nrow*ncol
            self.h_eos[start_id:end_id] = self.s[i,:]


            
        self.d_eos = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h_eos)

        compile_options.append('-D EOS_ED_START={value}f'.format(
                                                 value=self.ed_start))
        compile_options.append('-D EOS_ED_STEP={value}f'.format(
                                                 value=self.ed_step))
        compile_options.append('-D EOS_NUM_ED={value}'.format(
                                                 value=self.num_of_ed))
        compile_options.append('-D EOS_NUM_OF_ROWS=%s'%nrow)
        compile_options.append('-D EOS_NUM_OF_COLS=%s'%ncol)

        compile_options.append('-D EOS_NB_START={value}f'.format(
                                                 value=self.nb_start))
        compile_options.append('-D EOS_NB_STEP={value}f'.format(
                                                 value=self.nb_step))
        compile_options.append('-D EOS_NUM_NB={value}'.format(
                                                 value=self.num_of_nb))
        compile_options.append('-D EOS_NUM_OF_WIDTHS={value}'.format(
                                                 value=self.num_of_nb))
        compile_options.append('-D {value}'.format(
                                                 value=self.eos_type.upper()))
        self.compile_options = compile_options
        return self.d_eos 

    def create_table_hardon_gas(self, ctx, compile_options, nrow=200, ncol=1000):
        '''store the eos (ed, pr, T, s) in image2d_t table for fast
        linear interpolation,
        add some information to compile_options for EOS table'''
        
        mf = cl.mem_flags

        self.d_eos = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h_eos)

        compile_options.append('-D EOS_ED_START={value}f'.format(
                                                 value=self.ed_start))
        compile_options.append('-D EOS_ED_STEP={value}f'.format(
                                                 value=self.ed_step))
        compile_options.append('-D EOS_NUM_ED={value}'.format(
                                                 value=self.num_of_ed))
        compile_options.append('-D EOS_NUM_OF_ROWS=%s'%nrow)
        compile_options.append('-D EOS_NUM_OF_COLS=%s'%ncol)

        compile_options.append('-D EOS_NB_START={value}f'.format(
                                                 value=self.nb_start))
        compile_options.append('-D EOS_NB_STEP={value}f'.format(
                                                 value=self.nb_step))
        compile_options.append('-D EOS_NUM_NB={value}'.format(
                                                 value=self.num_of_nb))
        compile_options.append('-D EOS_NUM_OF_WIDTHS={value}'.format(
                                                 value=self.num_of_nb))
        compile_options.append('-D {value}'.format(
                                                 value=self.eos_type.upper()))
        self.compile_options = compile_options
        return self.d_eos 

    def create_table_neosB(self, ctx, compile_options):
        '''store the eos (ed, pr, T, s) in image2d_t table for fast
        linear interpolation,
        add some information to compile_options for EOS table'''
        
        mf = cl.mem_flags
        self.d_eos = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h_eos)
        compile_options.append('-D {value}'.format(
                                                 value=self.eos_type.upper()))

        self.compile_options = compile_options
        return self.d_eos 
    
    
    def create_table_chiral(self, ctx, compile_options):
        '''store the eos (ed, pr, T, s) in image2d_t table for fast
        linear interpolation,
        add some information to compile_options for EOS table'''
        
        mf = cl.mem_flags
        
        self.d_eos = cl.Buffer(ctx,mf.READ_ONLY | mf.COPY_HOST_PTR,hostbuf=self.h_eos)
        
        compile_options.append("-D EDMAXSMALL={value}f".format(value = self.edmaxsmall))
        compile_options.append("-D NBMAXSMALL={value}f".format(value = self.nbmaxsmall))
        compile_options.append("-D EDMINSMALL={value}f".format(value = self.edminsmall))
        compile_options.append("-D NBMINSMALL={value}f".format(value = self.nbminsmall))
        compile_options.append("-D NESMALL={value}".format(value = self.nedsmall))
        compile_options.append("-D NBSMALL={value}".format(value = self.nnbsmall))

        compile_options.append("-D EDMAXBIG={value}f".format(value = self.edmaxbig))
        compile_options.append("-D NBMAXBIG={value}f".format(value = self.nbmaxbig))
        compile_options.append("-D EDMINBIG={value}f".format(value = self.edminbig))
        compile_options.append("-D NBMINBIG={value}f".format(value = self.nbminbig))
        compile_options.append("-D NEBIG={value}".format(value = self.nedbig))
        compile_options.append("-D NBBIG={value}".format(value = self.nnbbig))


        compile_options.append('-D {value}'.format(
                                                 value=self.eos_type.upper()))

        self.compile_options = compile_options
        return self.d_eos 

   
if __name__ == '__main__':
    
    
    #eos = Eos('IDEAL_GAS')
    #eos = Eos('lattice_pce165')
    #eos = Eos('lattice_pce150')
    #eos = Eos('lattice_wb')
    eos = Eos('PURE_GAUGE')
    #eos = Eos('FIRST_ORDER')
    #eos = Eos('IDEAL_GAS_BARYON')
    #eos = Eos('NEOSB')
    #eos = Eos('NEOSBQS')
    #eos = Eos('EOSQ')
    #eos = Eos('NJL_MODEL')
    #eos = Eos('HARDON_GAS')
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    compile_options = []
    eos_table = eos.create_table(ctx, compile_options)
    #eos_table = eos.create_table_nb(ctx, compile_options, nrow=100, ncol=1555)
    #eos_table = eos.create_table_neosB(ctx,compile_options)
    #eos_table = eos.create_table_chiral(ctx,compile_options)
    #eos_table = eos.create_table_hardon_gas(ctx,compile_options)
    print (eos.compile_options)
    print (eos.f_P(1e-15*0.19733))
    print (eos.f_S(1e-15*0.19733))
    print (eos.f_T(1e-15*0.19733))

