#/usr/bin/env python
from __future__ import absolute_import, division, print_function
import numpy as np
import os
import sys
from time import time
from subprocess import call
import matplotlib.pyplot as plt

NPT=16
pts = pts = np.linspace(0.3,3.1,15)

def average_vn_pt(path0,NEVENT,pid="211",n=2):
    cn0=np.zeros((NPT,5,NEVENT))
    for i in range(NEVENT):
        fpath = os.path.join(path0,"event%d/final/cn%s_pt_%s.dat"%(i,n,pid))
        data = np.loadtxt(fpath)
        cn0[:,:,i] = cn0[:,:,i]+data
        #print (i)

    vn2 = np.nanmean(cn0[:,1,:]*cn0[:,2,:],axis=1)/np.nanmean(cn0[:,2,:],axis=1)
    dn2= vn2[:-1]
    cn2= np.sqrt(vn2[-1])
    
    vn2 = dn2/cn2
    #print (vn2)
    

    vn4 = np.nanmean(cn0[:,3,:]*cn0[:,4,:],axis=1)/np.nanmean(cn0[:,4,:],axis=1)
    dn4 = vn4[:-1]-2.0*dn2*cn2**2
    cn4 = vn4[-1] - 2.0*cn2**4
    
    vn4 = - dn4/np.power(-cn4,0.75)
    #print (vn4)
    
    outputpath = os.path.join(path0,"ave_event")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    np.savetxt(os.path.join(outputpath,"vn%s_pt_%s.dat"%(n,pid)),zip(pts,vn2,vn4))

def average_vn_integrate_Qn(path0,NEVENT,pid="211",n=2):
    cn0=np.zeros((4,NEVENT))
    v4=[]
    for i in range(NEVENT):
        
        fpath = os.path.join(path0,"event%d/final/cn%s_integrated_%s.dat"%(i,n,pid))
        #try:
        data = np.loadtxt(fpath)
        cn0[:,i] = cn0[:,i]+data
        cn2 = np.nansum(cn0[0,:]*cn0[1,:])/np.nansum(cn0[1,:])
        vn2 = np.sqrt(cn2)
        cn4 = np.nansum(cn0[2,:]*cn0[3,:])/np.nansum(cn0[3,:])
    
        cn4 = cn4 - 2.0*cn2*cn2
        print(i,pid,cn4,2.0*cn2*cn2,cn2,data[0],data[2],NEVENT)
        #except:
        #    pass

        
        

    cn2 = np.nansum(cn0[0,:]*cn0[1,:])/np.nansum(cn0[1,:])
    vn2 = np.sqrt(cn2)
    
    #print (np.mean(cn0[0,:]),np.std(cn0[0,:]),np.max(cn0[0,:]))
    cn4 = np.nansum(cn0[2,:]*cn0[3,:])/np.nansum(cn0[3,:])
    
    cn4 = cn4 - 2.0*cn2*cn2
    
    if cn4 < 0:
        vn4 = np.power(-cn4,0.25)
    else:
        vn4 = 0.0
    print (vn2,vn4,cn4,pid,n)
    outputpath = os.path.join(path0,"ave_event")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    np.savetxt(os.path.join(outputpath,"vn%s_integrate_Qn_%s.dat"%(n,pid)),[vn2,vn4])


def average_mean_pt(path0,NEVENT,pid="211"):
    pt_list = []
    for i in range(NEVENT):
        fpath = os.path.join(path0,"event%d/final/mean_pt_Y_%s.dat"%(i,pid))
        data = np.loadtxt(fpath)
        pt_list.append(data)    
    outputpath = os.path.join(path0,"ave_event")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    np.savetxt(os.path.join(outputpath,"mean_pt_Y_%s.dat"%(pid)),[np.nanmean(pt_list)])



def average_vn_integrate(path0,NEVENT,pid="211"):
    data = np.loadtxt(os.path.join(path0,"event0/final/vn_2_intgeted_%s.dat"%(pid)))
    for i in range(1,NEVENT):
        data_tep = np.loadtxt(os.path.join(path0,"event%i/final/vn_2_intgeted_%s.dat"%(i,pid)))
        data = data+data_tep
    data = data/float(NEVENT)

    outputpath = os.path.join(path0,"ave_event")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    np.savetxt(os.path.join(outputpath,"vn_2_intgeted_%s.dat"%(pid)),data)


def average_dNdeta(path0,NEVENT,pid="211",rapidity_kind="Eta"):
    data = np.loadtxt(os.path.join(path0,"event0/final/dNd%s_mc_%s.dat"%(rapidity_kind,pid)))
    for i in range(1,NEVENT):
        data_tep = np.loadtxt(os.path.join(path0,"event%i/final/dNd%s_mc_%s.dat"%(i,rapidity_kind,pid)))
        data = data+data_tep
    data = data/float(NEVENT)

    outputpath = os.path.join(path0,"ave_event")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    np.savetxt(os.path.join(outputpath,"dNd%s_mc_%s.dat"%(rapidity_kind,pid)),data)

def average_ptspec(path0,NEVENT,pid="211",rapidity_kind="Eta"):
    data = np.loadtxt(os.path.join(path0,"event0/final/dN_over_2pid%sptdpt_mc_%s.dat"%(rapidity_kind,pid)))
    for i in range(1,NEVENT):
        data_tep = np.loadtxt(os.path.join(path0,"event%i/final/dN_over_2pid%sptdpt_mc_%s.dat"%(i,rapidity_kind,pid)))
        data = data+data_tep
    data = data/float(NEVENT)

    outputpath = os.path.join(path0,"ave_event")
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    np.savetxt(os.path.join(outputpath,"dN_over_2pid%sptdpt_mc_%s.dat"%(rapidity_kind,pid)),data)


def average_event(path,NEVENT):
    #average_vn_pt(path,NEVENT,pid="211",n=2)
    #average_vn_pt(path,NEVENT,pid="211",n=3)
    #average_vn_pt(path,NEVENT,pid="2212",n=2)
    #average_vn_pt(path,NEVENT,pid="2212",n=3)

    #average_dNdeta(path,NEVENT,pid="211",rapidity_kind="Y")
    #average_dNdeta(path,NEVENT,pid="2212",rapidity_kind="Y")
    #average_dNdeta(path,NEVENT,pid="321",rapidity_kind="Y")
    #average_dNdeta(path,NEVENT,pid="-2212",rapidity_kind="Y")

    #average_ptspec(path,NEVENT,pid="211",rapidity_kind="Y")
    #average_ptspec(path,NEVENT,pid="2212",rapidity_kind="Y")
    #average_ptspec(path,NEVENT,pid="321",rapidity_kind="Y")
    #average_ptspec(path,NEVENT,pid="-2212",rapidity_kind="Y")

    # average_vn_integrate(path,NEVENT,pid="charged")
    #average_vn_integrate_Qn(path,NEVENT,pid="charged",n=3)
    average_vn_integrate_Qn(path,NEVENT,pid="charged",n=2)
    # average_vn_integrate_Qn(path,NEVENT,pid="211",n=3)
    # average_vn_integrate_Qn(path,NEVENT,pid="211",n=2)
    # average_vn_integrate_Qn(path,NEVENT,pid="2212",n=3)
    # average_vn_integrate_Qn(path,NEVENT,pid="2212",n=2)
    # average_vn_integrate_Qn(path,NEVENT,pid="-2212",n=2)
    # average_vn_integrate_Qn(path,NEVENT,pid="-2212",n=3)
    # average_mean_pt(path,NEVENT,pid="211")
    # average_mean_pt(path,NEVENT,pid="321")
    # average_mean_pt(path,NEVENT,pid="2212")









if __name__ =="__main__":
    if len(sys.argv) != 3:
        print ("usage: python average_event.py path num")
        exit()
    path=sys.argv[1]
    path = os.path.abspath(path)
    NEVENT = int(sys.argv[2])
    average_event(path,NEVENT)

