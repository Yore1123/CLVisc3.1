#/usr/bin/env python

#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2024- Jun-Qi Tao <taojunqi@mails.ccnu.edu.cn>
#Copyright (c) 2024- Xiang Fan <xfan@mails.ccnu.edu.cn>


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from time import time
import math 
import cmath

try:
    # used in python 2.*
    from StringIO import StringIO as fstring
except ImportError:
    # used in python 3.*
    from io import StringIO as fstring



class mcspec(object):
    def __init__(self, events_str, rapidity_kind='eta', fpath='./',nsampling=2000):
        self.rapidity_col = 4
        if rapidity_kind == 'eta':
            self.rapidity_col = 6
        
        if type(events_str).__name__=="unicode":
            self.events = [np.genfromtxt(fstring(event)) for event
                                    in events_str.split('#finished')[:-1]]

        #self.events = [pd.read_csv(fstring(event), sep=' ',
        #               header=None, dtype=np.float64).values
        #               for event in events_str.split('#finished')[:-1]]

            self.num_of_events = len(self.events)
            print('in mcspec, num of events=', self.num_of_events)
            print(self.events[9999])

            self.fpath = fpath
        else:
            self.events = events_str
            self.num_of_events = nsampling 
            self.fpath = fpath
            self.oversampleptc = self.events 
            self.pid = self.oversampleptc[:,5]
            self.px = self.oversampleptc[:, 1]
            self.py = self.oversampleptc[:, 2]
            self.pt = np.sqrt(self.px*self.px + self.py*self.py)
            self.phi = np.arctan2(self.py, self.px)
            


    def pt_differential_vn(self, n=2, pid='211', pt_min=0.3, pt_max=2.5, eta_max=2.0):
        pts = np.linspace(0.3, 2.5, 11)

        avg2_list = np.zeros(self.num_of_events)
        avg4_list = np.zeros(self.num_of_events)
        avg2_prime_list = np.zeros((self.num_of_events, len(pts)))
        avg4_prime_list = np.zeros((self.num_of_events, len(pts)))

        for idx, spec in enumerate(self.events):
            self.rapidity = spec[:, self.rapidity_col]
            # pion, kaon, proton, ... pid
            self.pid = spec[:, 5]

            self.px = spec[:, 1]
            self.py = spec[:, 2]
            self.pt = np.sqrt(self.px*self.px + self.py*self.py)
            self.phi = np.arctan2(self.py, self.px)

            # use charged particle for reference flow
            avg2_list[idx], avg4_list[idx] = self.avg(n, pid, pt1=0, pt2=4.2,
                                                  rapidity1=-5.5, rapidity2=5.5)

            for ipt, pt in enumerate(pts):
                avg2_prime_list[idx, ipt], avg4_prime_list[idx, ipt] = self.avg_prime(
                        n, pid, pt_min=pt-0.1, pt_max=pt+0.1, eta_min=-eta_max, eta_max=eta_max)

        vn2, vn4 = self.differential_flow(avg2_list, avg4_list,
                                               avg2_prime_list, avg4_prime_list)

        return pts, vn2, vn4

    def vn_vs_eta(self, pid='charged', make_plot=False):
        # <<2>> as a function of eta
        avg2_vs_eta = np.zeros(20)

        eta = np.linspace(-5, 5, 20)
        avg22_prime_list = np.zeros((self.num_of_events, 20))
        avg32_prime_list = np.zeros((self.num_of_events, 20))
        avg42_prime_list = np.zeros((self.num_of_events, 20))

        avg24_prime_list = np.zeros((self.num_of_events, 20))
        avg34_prime_list = np.zeros((self.num_of_events, 20))
        avg44_prime_list = np.zeros((self.num_of_events, 20))

        avg22_ref_list = np.zeros(self.num_of_events)
        avg32_ref_list = np.zeros(self.num_of_events)
        avg42_ref_list = np.zeros(self.num_of_events)

        avg24_ref_list = np.zeros(self.num_of_events)
        avg34_ref_list = np.zeros(self.num_of_events)
        avg44_ref_list = np.zeros(self.num_of_events)

        for idx, spec in enumerate(self.events):
            self.rapidity = spec[:, self.rapidity_col]
            # pion, kaon, proton, ... pid
            self.pid = spec[:, 5]

            self.px = spec[:, 1]
            self.py = spec[:, 2]
            self.pt = np.sqrt(self.px*self.px + self.py*self.py)
            self.phi = np.arctan2(self.py, self.px)

            # The reference particles
            avg22_ref_list[idx], avg24_ref_list[idx] = self.avg(2, pid,
                    pt1=0, pt2=5.0, rapidity1=-0.8, rapidity2=0.8)
            avg32_ref_list[idx], avg34_ref_list[idx] = self.avg(3, pid,
                    pt1=0, pt2=5.0, rapidity1=-0.8, rapidity2=0.8)
            avg42_ref_list[idx], avg44_ref_list[idx] = self.avg(4, pid,
                    pt1=0, pt2=5.0, rapidity1=-0.8, rapidity2=0.8)

            for ih in range(20):
                hmin = eta[ih] - 0.25
                hmax = eta[ih] + 0.25
                avg22_prime_list[idx, ih], avg24_prime_list[idx, ih] = self.avg_prime(2, pid,
                        pt_min=0, pt_max=5.0, eta_min=hmin, eta_max=hmax)
                avg32_prime_list[idx, ih], avg34_prime_list[idx, ih] = self.avg_prime(3, pid,
                        pt_min=0, pt_max=5.0, eta_min=hmin, eta_max=hmax)
                avg42_prime_list[idx, ih], avg44_prime_list[idx, ih] = self.avg_prime(4, pid,
                        pt_min=0, pt_max=5.0, eta_min=hmin, eta_max=hmax)

        # ignore the NAN in mean calculation if there is no particles in one rapidity bin in a event
        v22, v24 = self.differential_flow(avg22_ref_list, avg24_ref_list, avg22_prime_list, avg24_prime_list)
        v32, v34 = self.differential_flow(avg32_ref_list, avg34_ref_list, avg32_prime_list, avg34_prime_list)
        v42, v44 = self.differential_flow(avg42_ref_list, avg44_ref_list, avg42_prime_list,avg44_prime_list)
        np.savetxt(os.path.join(self.fpath, 'vn24_vs_eta.txt'), list(zip(eta, v22, v32, v42, v24, v34, v44)))

        if make_plot:
            plt.plot(eta, v22, label='v2{2}')
            plt.plot(eta, v32, label='v3{2}')
            plt.plot(eta, v42, label='v4{2}')

            plt.legend(loc='best')
            plt.show()



    def qn(self, n, pid='211', pt1=0.0, pt2=3.0, rapidity1=-1, rapidity2=1):
        '''return the Qn cumulant vector for particles with pid in
        pt range [pt1, pt2] and rapidity range [rapidity1, rapidity2]
        Params:
            :param n: int, order of the Qn cumulant vector
            :param pid: string, 'charged', '211', '321', '2212' for
                charged particles, pion+, kaon+ and proton respectively
            :param pt1: float, lower boundary for transverse momentum
            :param pt2: float, upper boundary for transverse momentum
            :param rapidity1: float, lower boundary for rapidity
            :param rapidity2: float, upper boundary for rapidity
        Return:
            particle_of_interest, multiplicity, Qn = sum_i^m exp(i n phi) '''

        # poi stands for particle of interest
        particle_of_interest = None

        def multi_and(*args):
            '''select elements of one numpy array that satisfing multiple situations'''
            selected = np.ones_like(args[0], dtype=np.bool_)
            for array_i in args:
                selected = np.logical_and(selected, array_i)
            return selected

        if pid == 'charged':
            particle_of_interest = multi_and(self.pt>pt1, self.pt<pt2,
                                             self.rapidity > rapidity1,
                                             self.rapidity < rapidity2)
        else:
            particle_of_interest = multi_and(self.pid == int(pid), self.pt>pt1,
                                             self.pt<pt2, self.rapidity > rapidity1,
                                             self.rapidity < rapidity2)

        multiplicity = np.count_nonzero(particle_of_interest)

        return  particle_of_interest, multiplicity, np.exp(1j*n*self.phi[particle_of_interest]).sum()


    def avg(self, n, pid='211', pt1=0.0, pt2=3.0, rapidity1=-1, rapidity2=1,mergedata=True):
        '''return the 2- and 4- particle cumulants for particles with pid in
        pt range [pt1, pt2] and rapidity range [rapidity1, rapidity2]
        Params:
            :param n: int, order of the Qn cumulant vector
            :param pid: string, 'charged', '211', '321', '2212' for
                charged particles, pion+, kaon+ and proton respectively
            :param pt1: float, lower boundary for transverse momentum
            :param pt2: float, upper boundary for transverse momentum
            :param rapidity1: float, lower boundary for rapidity
            :param rapidity2: float, upper boundary for rapidity
        Return:
            cn{2} = <<2>>
            cn{4} = <<4>> - 2<<2>>**2 '''
        POI_0, M, Qn = self.qn(n, pid, pt1, pt2, rapidity1, rapidity2)
        Qn_square = Qn * Qn.conjugate()
        avg2 = ((Qn_square - M)/float(M*(M-1))).real
        weight = float(M*(M-1))
        POI_1, M2, Q2n = self.qn(2*n, pid, pt1, pt2, rapidity1, rapidity2)
        Q2n_square = Q2n * Q2n.conjugate()
        
        weight2 =  float(M*(M-1)*(M-2)*(M-3))

        term1 = (Qn_square**2 + Q2n_square - 2*(Q2n*Qn.conjugate()**2).real
                )/float(M*(M-1)*(M-2)*(M-3))
        term2 = 2*(2*(M-2)*Qn_square - M*(M-3))/float(M*(M-1)*(M-2)*(M-3))
        avg4 = term1 - term2
        #print (term1,term2,Qn_square**2,Q2n_square,weight2) 
        if mergedata == True:
            return avg2.real, weight, avg4.real, weight2
        else:
            return avg2.real, avg4.real


    def avg_prime(self, n, pid, pt_min, pt_max, eta_min, eta_max,
                  eta_ref_min = -0.8, eta_ref_max = 0.8,
                  pt_ref_min = 0.0, pt_ref_max = 5.0,mergedata=True):
        '''return <2'> and <4'> for particles with pid in the 
        range ( pt_min < pt < pt_max ) and ( eta_min < eta < eta_max )'''
        # Qn from reference flow particles
        REF_0, M, Qn = self.qn(n,"charged", pt_ref_min, pt_ref_max, eta_ref_min, eta_ref_max)
        REF_1, M2, Q2n = self.qn(2*n, "charged", pt_ref_min, pt_ref_max, eta_ref_min, eta_ref_max)

        # particle of interest
        POI_0, mp, pn = self.qn(n, pid, pt_min, pt_max, eta_min, eta_max)
 
        # mq, qn: labeled as both POI and REF
        POI_AND_REF = REF_0 & POI_0
        mq = np.count_nonzero(POI_AND_REF)
        qn = np.exp(1j*n*self.phi[POI_AND_REF]).sum()

        #POI_1, mq2, q2n = self.qn(2*n, pid, pt_min, pt_max, eta_min, eta_max)
        q2n = np.exp(2j*n*self.phi[POI_AND_REF]).sum()

        avg2_prime = (pn * Qn.conjugate() - mq)/(mp * M - mq)
        avg4_prime = (pn * Qn * Qn.conjugate()**2 - q2n * Qn.conjugate()**2 - pn * Qn * Q2n.conjugate()
                - 2 * M * pn * Qn.conjugate() - 2 * mq * Qn * Qn.conjugate() +
                7 * qn * Qn.conjugate() - Qn * qn.conjugate() + q2n * Q2n.conjugate() 
                + 2 * pn * Qn.conjugate() + 2 * mq * M - 6 * mq ) / (
                        (mp * M - 3 * mq) * (M - 1) * (M - 2))
        weight1 = mp*M-mq
        weight2 = (mp * M - 3 * mq) * (M - 1) * (M - 2)
        if mergedata == True:
            return avg2_prime.real,weight1, avg4_prime.real, weight2
        else:
            return avg2_prime.real, avg4_prime.real


                               
    def differential_flow(self, avg2_list, avg4_list,
                                avg2_prime_list, avg4_prime_list):
        '''return the differential flow vs transverse momentum
        Params: 
            :param avg2_list: ebe <2>
            :param avg4_list: ebe <4>

        Returns:
            pt_array, vn{2} array, vn{4} array '''
        avg2 = np.nanmean(avg2_list)
        avg4 = np.nanmean(avg4_list)
        avg2_prime = np.nanmean(avg2_prime_list, axis=0)
        avg4_prime = np.nanmean(avg4_prime_list, axis=0)

        cn2 = avg2
        cn4 = avg4 - 2 * avg2 * avg2

        dn2 = avg2_prime
        dn4 = avg4_prime - 2 * avg2_prime * avg2

        vn2 = dn2 / np.sqrt(cn2)
        vn4 = - dn4 / np.power(-cn4, 0.75)

        return vn2, vn4
    
    def integrated_flow(self, n=2, pid='211', pt1=0.2, pt2=4.0, rapidity1=-1.0, rapidity2=-0.2,kind="Eta"):
        #sub-event method

        rapidity_col = 4
        if kind == 'Eta':
            rapidity_col = 6

        self.rapidity = self.oversampleptc[:,rapidity_col]

        POI_0,M,Qn = self.qn(n,pid,pt1,pt2,rapidity1,rapidity2)
        POI_prime,M_prime,Qn_prime = self.qn(n,pid,pt1,pt2,-rapidity2,-rapidity1)
        cn2 = Qn*Qn_prime.conjugate()/(M*M_prime*1.0)
        vn2 = np.sqrt(cn2)
        

        return vn2
    
    def get_integrated_flow(self, pid='211', pt1=0.2, pt2=4.0, rapidity1=-1.0, rapidity2=-0.5,kind="Eta"):
        vn = np.zeros(3)
        vn[0] = self.integrated_flow(n=2,pid ="charged",kind=kind)
        vn[1] = self.integrated_flow(n=3,pid ="charged",kind=kind)
        vn[2] = self.integrated_flow(n=4,pid ="charged",kind=kind)

        np.savetxt(os.path.join(self.fpath, 'vn_2_intgeted_%s.dat'%pid), vn)
    
    
    def get_pt_differential_cn(self, n=2, pid='211', pt_min=0.3, pt_max=2.5, eta_max=2.0,eta_ref_max=4.0,kind="Eta"):
        pts = np.linspace(0.3,3.1,15,dtype=np.float64)
        dpt = pts[1]-pts[0]

        avg2_prime = np.zeros((len(pts)))
        avg4_prime = np.zeros((len(pts)))
        
        avg2_prime_weight = np.zeros((len(pts)))
        avg4_prime_weight = np.zeros((len(pts)))

        rapidity_col = 4
        if kind == 'Eta':
            rapidity_col = 6

        self.rapidity = self.oversampleptc[:,rapidity_col]
        
        

        # use charged particle for reference flow
        avg2,avg2_weight,avg4,avg4_weight = self.avg(n, pid="charged", pt1=0.2, pt2=3.2,
                                                  rapidity1=-eta_ref_max, rapidity2=eta_ref_max,mergedata=True)
        for ipt, pt in enumerate(pts):
            avg2_prime[ipt],avg2_prime_weight[ipt],avg4_prime[ipt], avg4_prime_weight[ipt] \
                    = self.avg_prime(n, pid, pt_min=pt-0.5*dpt, pt_max=pt+0.5*dpt, eta_min=-eta_max,\
                    eta_max=eta_max,eta_ref_min=-eta_ref_max,eta_ref_max=eta_ref_max,pt_ref_min = 0.2, pt_ref_max = 3.2)
        
        avg2_prime = np.append(avg2_prime,avg2)
        avg2_prime_weight = np.append(avg2_prime_weight,avg2_weight)
        avg4_prime = np.append(avg4_prime,avg4)
        avg4_prime_weight = np.append(avg4_prime_weight,avg4_weight)
        pts = np.append(pts,0.0)


        np.savetxt(os.path.join(self.fpath, 'cn%s_pt_%s.dat'%(n,pid)),\
                list(zip(pts,avg2_prime,avg2_prime_weight,\
                avg4_prime,avg4_prime_weight)))

    def get_integrated_cn(self, n=2, pid='211', pt_min=0.3, pt_max=2.5, rapidity1=-1.0, rapidity2=1.0,kind="Eta"):  
        rapidity_col = 4
        if kind == 'Eta':
            rapidity_col = 6

        self.rapidity = self.oversampleptc[:,rapidity_col]

        
        avg2,avg2_weight,avg4,avg4_weight = self.avg(n, pid, pt1=pt_min, pt2=pt_max,
                                                  rapidity1=rapidity1, rapidity2=rapidity2,mergedata=True)
        

        np.savetxt(os.path.join(self.fpath, 'cn%s_integrated_%s.dat'%(n,pid)),\
                [avg2,avg2_weight,avg4,avg4_weight])
    

    def get_dNdY(self, pid=211, kind='Y'):
        rapidity_col = 4
        if kind == 'Eta':
            rapidity_col = 6
        Yi = self.oversampleptc[:, rapidity_col]
        
        dN, Y = None, None
        if pid == 'charged':
            dN, Y = np.histogram(Yi, bins=40,range=[-8,8])
        else:
            dN, Y = np.histogram(Yi[np.isclose(self.oversampleptc[:, 5],pid)], bins=40,range=[-8,8])

        #print (len(Yi[np.isclose(self.oversampleptc[:, 5],pid)]),pid)
        dY = (Y[1:]-Y[:-1])
        Y = 0.5*(Y[:-1]+Y[1:])
        res = np.array([Y, dN/(dY*float(self.num_of_events))]).T
        np.savetxt(os.path.join(self.fpath, 'dNd%s_mc_%s.dat'%(kind, pid)), res)
        return res[:, 0], res[:, 1]
    def get_ptspec(self, pid=211, kind='Y', rapidity_window=1.6):
        E = self.oversampleptc[:,0]
        pz = self.oversampleptc[:,3]
        rapidity_col = 4
        if kind == 'Eta':
            rapidity_col = 6
    
        particle_type = None
    
        if pid == 'charged':
            particle_type = (np.isclose(self.oversampleptc[:, 5],self.oversampleptc[:, 5]))
        else:
            particle_type = (np.isclose(self.oversampleptc[:, 5],pid))
    
        Yi = self.oversampleptc[particle_type, rapidity_col]
        dN, Y = np.histogram(Yi, bins=20)
    
        dY = (Y[1:]-Y[:-1])
        Y = 0.5*(Y[:-1]+Y[1:])
    
        pti = np.sqrt(self.oversampleptc[particle_type, 1]**2+self.oversampleptc[particle_type, 2]**2)
    
        pti = pti[np.abs(Yi)<0.5*rapidity_window]
    
        dN, pt = np.histogram(pti, bins=20)
        print (np.max(pti))
    
        dpt = pt[1:]-pt[:-1]
        pt = 0.5*(pt[1:]+pt[:-1])
    
        res = np.array([pt, dN/(2*np.pi*float(self.num_of_events)*pt*dpt*rapidity_window)]).T
        fname = os.path.join(self.fpath, 'dN_over_2pid%sptdpt_mc_%s.dat'%(kind, pid))
        
        np.savetxt(fname, res)

    def get_dNdphi(self, pid=211, kind='Y', rapidity_window=1.6):
        E = self.oversampleptc[:,0]
        pz = self.oversampleptc[:,3]
        rapidity_col = 4
        if kind == 'Eta':
            rapidity_col = 6
    
        particle_type = None
    
        if pid == 'charged':
            particle_type = (np.isclose(self.oversampleptc[:, 5],self.oversampleptc[:, 5]))
        else:
            particle_type = (np.isclose(self.oversampleptc[:, 5],pid))
    
        Yi = self.oversampleptc[particle_type, rapidity_col]
        
        phi_p = np.arctan2(self.oversampleptc[particle_type,2], self.oversampleptc[particle_type,1])
    
        Phi = phi_p[np.abs(Yi)<0.5*rapidity_window]
    
        dN, phi = np.histogram(Phi, range=[-np.pi, np.pi], bins=20)
    
        dphi = phi[1:]-phi[:-1]
        phi = 0.5*(phi[1:]+phi[:-1])
    
        res = np.array([phi, dN/(float(self.num_of_events)*dphi)]).T
    
        #fname = os.path.join(self.fpath, 'dN_over_2pid%sptdpt_mc_%s.dat'%(kind, pid))
        fname = os.path.join(self.fpath, 'dNdphi_%s_mc_%s.dat'%(kind, pid))
        np.savetxt(fname, res)

    def get_ptmean(self, pid=211, kind='Y', rapidity_window=1.6):
        E = self.oversampleptc[:,0]
        pz = self.oversampleptc[:,3]
        pt = np.sqrt(self.oversampleptc[:,1]**2 + self.oversampleptc[:,2]**2)
        rapidity_col = 4
        if kind == 'Eta':
            rapidity_col = 6
    
        particle_type = None
    
        if pid == 'charged':
            particle_type = (np.isclose(self.oversampleptc[:, 5],self.oversampleptc[:, 5]))
        else:
            particle_type = (np.isclose(self.oversampleptc[:, 5],pid))
    
        Yi = self.oversampleptc[particle_type, rapidity_col]
    
        pti = np.sqrt(self.oversampleptc[particle_type, 1]**2+self.oversampleptc[particle_type, 2]**2)
        pti = pti[np.abs(Yi)<0.5*rapidity_window]
    
        #print (pti) 
        meanpt = np.mean(pti)
    

     
    
        fname = os.path.join(self.fpath, 'mean_pt_%s_%s.dat'%(kind, pid))
        np.savetxt(fname, [meanpt])

    def get_event_planes(self, pid = "211", kind='Y',Ylo=3.3, Yhi=3.9, total_n=6,num_ptbins=15, num_phibins=50):
        rapidity_col = 4
        if kind == 'Eta':
            rapidity_col = 6

        particle_type = None
        if pid == 'charged':
            particle_type = (np.isclose(self.oversampleptc[:, 5],self.oversampleptc[:, 5]))
        else:
            particle_type = (np.isclose(self.oversampleptc[:, 5],pid))

        Yi = self.oversampleptc[particle_type, rapidity_col]
        phi_p = np.arctan2(self.oversampleptc[particle_type,2], self.oversampleptc[particle_type,1])
        pti = np.sqrt(self.oversampleptc[particle_type, 1]**2 + self.oversampleptc[particle_type, 2]**2)

        pti = pti[(Yi>Ylo)*(Yi<Yhi)]
        phi_p = phi_p[(Yi>Ylo)*(Yi<Yhi)]
        d2N, pt, Phi = np.histogram2d(pti, phi_p, range=[[0, 4.0], [-np.pi, np.pi]], bins=[num_ptbins, num_phibins])
        dpt = pt[1:]-pt[:-1]
        dphi_p = Phi[1:]-Phi[:-1]
        pt = 0.5*(pt[1:]+pt[:-1])
        Phi = 0.5*(Phi[1:]+Phi[:-1])

        d2N=d2N.flatten()
        Phi = np.repeat(Phi, num_ptbins)

        Vn = np.zeros(total_n+1)
        event_plane = np.zeros(total_n+1)
        total_vn = np.zeros(total_n+1, dtype=complex)
        Norm = np.sum(d2N)
        print('event_plane_window',Norm,pid)
        for n in range(1, total_n+1):
            total_vn[n] = (d2N * np.exp(1j*n*Phi)).sum()/float(Norm)
            Vn[n], event_plane[n]=cmath.polar(total_vn[n])
            event_plane[n] /= float(n)
        return event_plane




    
    def get_vn_pt_event_plane_method(self,pid="211",kind="Y",Ylo=-0.35,Yhi=0.35,Y_event_plane=[3.3,3.9], total_n=6, num_ptbins=15, num_phibins=50):
        rapidity_col = 4
        if kind == "Eta":
            rapidity_col = 6

        particle_type = None
        if pid == 'charged':
            particle_type = (np.isclose(self.oversampleptc[:, 5],self.oversampleptc[:, 5]))
        else:
            particle_type = (np.isclose(self.oversampleptc[:, 5],pid))
        Yi = self.oversampleptc[particle_type, rapidity_col]
        phi_p = np.arctan2(self.oversampleptc[particle_type,2], self.oversampleptc[particle_type,1])
        pti = np.sqrt(self.oversampleptc[particle_type, 1]**2 + self.oversampleptc[particle_type, 2]**2)
        pti = pti[np.abs(Yi)<Yhi]
        phi_p = phi_p[np.abs(Yi)<Yhi]
        d2N, pt, Phi = np.histogram2d(pti, phi_p, range=[[0, 4.0], [-np.pi, np.pi]], bins=[num_ptbins, num_phibins])
        
        dpt = pt[1:]-pt[:-1]
        dphi_p = Phi[1:]-Phi[:-1]
        pt = 0.5*(pt[1:]+pt[:-1])
        Phi = 0.5*(Phi[1:]+Phi[:-1])
        total_n = 6
        Vn_pt = np.zeros(shape=(num_ptbins, total_n+1))
        Vn_vec = np.zeros(shape=(num_ptbins, total_n+1), dtype=complex)
        angles = np.zeros(shape=(num_ptbins, total_n+1))
        event_plane = np.zeros(total_n+1)
        Norm = 0
        event_plane = self.get_event_planes(pid,kind, Ylo=Y_event_plane[0], Yhi=Y_event_plane[1])
        for i in range(num_ptbins):
            norm_factor = np.sum(d2N[i,:])
            if norm_factor<1e-2: continue
            print('norm,pti',norm_factor,pt[i])
            for n in range(1, 7):
                Vn_vec[i, n] = (d2N[i]*np.exp(1j*n*(Phi-event_plane[n]))).sum()/float(norm_factor)
                Vn_pt[i, n], angles[i, n]=cmath.polar(Vn_vec[i, n])
        fout_name = os.path.join(self.fpath,'vn_mc_%s.dat'%pid)
        np.savetxt(fout_name, np.array(list(zip(pt, Vn_pt[:,1],Vn_pt[:,2],Vn_pt[:,3],Vn_pt[:,4],Vn_pt[:,5], Vn_pt[:,6]))))




    def plot_vn_pt(self, n=2, make_plot=False):
        '''plot vn as a function of pt '''
        pts, vn2_pion,    vn4_pion = self.pt_differential_vn(n=n, pid='211')
        pts, vn2_kaon,    vn4_kaon = self.pt_differential_vn(n=n, pid='321')
        pts, vn2_proton,  vn4_proton = self.pt_differential_vn(n=n, pid='2212')
        pts, vn2_charged, vn4_charged = self.pt_differential_vn(n=n, pid='charged')

        np.savetxt(os.path.join(self.fpath, 'v%s_2_vs_pt.txt'%n), list(zip(pts,
                   vn2_pion, vn2_kaon, vn2_proton, vn2_charged)))

        np.savetxt(os.path.join(self.fpath, 'v%s_4_vs_pt.txt'%n), list(zip(pts,
                   vn4_pion, vn4_kaon, vn4_proton, vn4_charged)))

        print("v%s finished!"%n)

        if make_plot:
            plt.plot(pts, vn4_pion, label='v%s{2} pion'%n)
            plt.plot(pts, vn4_kaon, label='v%s{2} kaon'%n)
            plt.plot(pts, vn4_proton, label='v%s{2} proton'%n)

            plt.legend(loc='best')
            plt.show()



from subprocess import call, check_output

def calc_vn(fpath, over_sampling=1000, make_plot=False, viscous_on='true', decay='true'):
    cwd = os.getcwd()
    os.chdir('../build')
    call(['cmake', '..'])
    call(['make'])
    cmd = ['./main', fpath, viscous_on, decay, '%s'%over_sampling]

    proc = check_output(cmd)

    mc = mcspec(proc.decode('utf-8'), fpath=fpath)

    mc.vn_vs_eta(make_plot = make_plot)
    mc.plot_vn_pt(n=4)
    mc.plot_vn_pt(n=3)
    mc.plot_vn_pt(n=2, make_plot=make_plot)

    os.chdir(cwd)



if __name__=='__main__':
    t1 = time()

    import sys

    #fpath = '/lustre/nyx/hyihp/lpang/trento_ini/bin/pbpb2p76/20_30/n2/mean'
    fpath = '/lustre/nyx/hyihp/lpang/trento_ebe_hydro/pbpb2p76_results_ampt/etas0p16_k1p4/20_30/event1/'

    if len(sys.argv) == 2:
        fpath = sys.argv[1]

    calc_vn(fpath, over_sampling=10000, make_plot=True, viscous_on='false')
