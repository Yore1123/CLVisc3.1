import numpy as np
from subprocess import call
from mcspec import mcspec
import os
import glob
import gc
import argparse
from time import time
import pandas as pd
import h5py



def calc_obervables(path,infortem,nsampling,reso_decay,run_afterburner):
    mc = mcspec(infortem,rapidity_kind="eta",fpath=path,nsampling=nsampling)
    mc.get_integrated_flow( pid='charged',kind="Y")
    
    if reso_decay or run_afterburner:
        mc.get_dNdY(pid='charged', kind='Eta')
    mc.get_dNdY(pid=211, kind='Y')
    mc.get_dNdY(pid=2212, kind='Y')
    mc.get_dNdY(pid=321, kind='Y')
    mc.get_dNdY(pid=-2212, kind='Y')
    mc.get_dNdY(pid=9000211, kind='Y')
    mc.get_dNdY(pid=221, kind='Y')


    mc.get_ptspec(pid='charged',rapidity_window=1.6,kind='Y')
    mc.get_ptspec(pid=211,rapidity_window=0.5,kind='Y')
    mc.get_ptspec(pid=2212,rapidity_window=0.5,kind='Y')
    mc.get_ptspec(pid=321,rapidity_window=0.5,kind='Y')
    mc.get_ptspec(pid=-2212,rapidity_window=0.5,kind='Y')
    
    if reso_decay or run_afterburner:
        mc.get_dNdphi(pid='charged',rapidity_window=0.5,kind='Y')
    mc.get_dNdphi(pid=211,rapidity_window=0.5,kind='Y')

    # mc.get_pt_differential_cn(n=2,pid="211",kind='Y',eta_max=1.0)
    # mc.get_pt_differential_cn(n=3,pid="211",kind='Y',eta_max=1.0)
    # mc.get_pt_differential_cn(n=2,pid="2212",kind='Y',eta_max=1.0)
    # mc.get_pt_differential_cn(n=3,pid="2212",kind='Y',eta_max=1.0)
    mc.get_pt_differential_cn(n=2,pid="charged",kind='Y',eta_max=1.0)
    mc.get_pt_differential_cn(n=3,pid="charged",kind='Y',eta_max=1.0)
    # mc.get_pt_differential_cn(n=2,pid="-2212",kind='Y',eta_max=1.0)
    # mc.get_pt_differential_cn(n=3,pid="-2212",kind='Y',eta_max=1.0)
    # mc.get_pt_differential_cn(n=2,pid="321",kind='Y',eta_max=1.0)
    # mc.get_pt_differential_cn(n=3,pid="321",kind='Y',eta_max=1.0)
    mc.get_vn_pt_event_plane_method(pid = 211,kind="Eta",Ylo=-0.5,Yhi=0.5,Y_event_plane=[2.0,3.0])    
    mc.get_vn_pt_event_plane_method(pid = 321,kind="Eta",Ylo=-0.5,Yhi=0.5,Y_event_plane=[2.0,3.0])    
    mc.get_vn_pt_event_plane_method(pid = 2212,kind="Eta",Ylo=-0.5,Yhi=0.5,Y_event_plane=[2.0,3.0])    
    mc.get_vn_pt_event_plane_method(pid = -2212,kind="Eta",Ylo=-0.5,Yhi=0.5,Y_event_plane=[2.0,3.0])
    mc.get_vn_pt_event_plane_method(pid = "charged",kind="Eta",Ylo=-0.5,Yhi=0.5,Y_event_plane=[2.0,3.0])        
    if reso_decay or run_afterburner:
        mc.get_integrated_cn(n=2,pid="charged",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
        mc.get_integrated_cn(n=3,pid="charged",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
    
    mc.get_integrated_cn(n=2,pid="211",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
    mc.get_integrated_cn(n=3,pid="211",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
    mc.get_integrated_cn(n=2,pid="2212",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
    mc.get_integrated_cn(n=3,pid="2212",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
    mc.get_integrated_cn(n=2,pid="321",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
    mc.get_integrated_cn(n=3,pid="321",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
    mc.get_integrated_cn(n=2,pid="-2212",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)
    mc.get_integrated_cn(n=3,pid="-2212",kind='Eta',pt_min=0.2, pt_max=4.0, rapidity1=-1.0, rapidity2=1.0)

    mc.get_ptmean(pid=211, kind='Y', rapidity_window=0.2)
    mc.get_ptmean(pid=321, kind='Y', rapidity_window=0.2)
    mc.get_ptmean(pid=2212, kind='Y', rapidity_window=0.2)
    mc.get_ptmean(pid=-2212, kind='Y', rapidity_window=0.2)


  

def format_SMASH_h5(path):
    path = os.path.join("{event_dir}/final".format(event_dir=path))
    infoall=0
    acu=1e-6
    with h5py.File(os.path.join(path,"particle_lists.h5"),"r") as f:
        data = f["particle_list"]            
        nprt  = len(data["p0"][...])
        infoall = np.zeros((nprt,8))
        infoall[:,0] = data["p0"][...]
        infoall[:,1] = data["px"][...]
        infoall[:,2] = data["py"][...]
        infoall[:,3] = data["pz"][...]
      
      
        pmag = np.sqrt(infoall[:,3]**2+infoall[:,1]**2+infoall[:,2]**2)
        infoall[:,4] = 0.5*(np.log(np.maximum(infoall[:,0]+infoall[:,3],acu))-np.log(np.maximum(infoall[:,0]-infoall[:,3],acu)))
        infoall[:,5] = data["pdg"][...]
        infoall[:,6] = 0.5*(np.log(np.maximum(pmag+infoall[:,3],acu))-np.log(np.maximum(pmag-infoall[:,3],acu)))
        infoall[:,7] = data["charge"][...]
    select_charged = np.fabs(infoall[:,7]) > 0
    infotem = np.zeros((len(infoall[select_charged,0]),7))
    infotem[:,:] = infoall[select_charged,:-1]
    return infotem

def format_URQMD_h5(path):
    path = os.path.join("{event_dir}/final".format(event_dir=path))
    infoall=0
    with h5py.File(os.path.join(path,"particle_lists.h5"),"r") as f:
        data = f["particle_list"]            
        nprt  = len(data["p0"][...])
        infoall = np.zeros((nprt,8))
        infoall[:,0] = data["p0"][...]
        infoall[:,1] = data["px"][...]
        infoall[:,2] = data["py"][...]
        infoall[:,3] = data["pz"][...]
      
        infoall[:,4] = data["rap"][...]
        infoall[:,5] = data["pdg"][...]
        infoall[:,6] = data["eta"][...]
        infoall[:,7] = data["charge"][...]
    select_charged = np.fabs(infoall[:,7]) > 0
    infotem = np.zeros((len(infoall[select_charged,0]),7))
    infotem[:,:] = infoall[select_charged,:-1]
    return infotem

def format_SMASH_dat(path):
    acu=1e-6
    path = os.path.join("{event_dir}/before".format(event_dir=cfg.event_dir))
    fpaths = glob.glob(os.path.abspath("{path}/mc_particle_list*".format(path=path)))
    data = pd.read_csv(fpaths[0],sep=" ",comment="#",header=None)
    data = data.values
    if len(fpaths)>1:
        for fpathid in fpaths[1:]:
            data_tep = pd.read_csv(fpathid,sep=" ",comment="#",header=None)
            data = np.concatenate((data,data_tep))
    select_charged = np.fabs(data[:,11]) > 0
    num_charged = len(data[select_charged,0])
    infotem = np.zeros((num_charged,7))
    infotem[:,0] = data[select_charged,5]
    infotem[:,1] = data[select_charged,6]
    infotem[:,2] = data[select_charged,7]
    infotem[:,3] = data[select_charged,8]
    pmag = np.sqrt(infotem[:,3]**2+infotem[:,1]**2+infotem[:,2]**2)
    infotem[:,4] = 0.5*(np.log(np.maximum(infotem[:,0]+infotem[:,3],acu))-np.log(np.maximum(infotem[:,0]-infotem[:,3],acu)))
    infotem[:,5] = np.int32(data[select_charged,9])
    infotem[:,6] = 0.5*(np.log(np.maximum(pmag+infotem[:,3],acu))-np.log(np.maximum(pmag-infotem[:,3],acu)))

    return infotem


def format_URQMD_dat(path):
    acu=1e-6
    path = os.path.join("{event_dir}/before".format(event_dir=cfg.event_dir))
    fpaths = glob.glob(os.path.abspath("{path}/mc_particle_list*".format(path=path)))
    data = pd.read_csv(fpaths[0],sep=" ",comment="#",header=None)
    data = data.values
    if len(fpaths)>1:
        for fpathid in fpaths[1:]:
            data_tep = pd.read_csv(fpathid,sep=" ",comment="#",header=None)
            data = np.concatenate((data,data_tep))

    
    infotem = np.zeros((len(data[:,0]),7))
    infotem[:,0] = data[:,5]
    infotem[:,1] = data[:,6]
    infotem[:,2] = data[:,7]
    infotem[:,3] = data[:,8]
    pmag = np.sqrt(infotem[:,3]**2+infotem[:,1]**2+infotem[:,2]**2)
    infotem[:,4] = 0.5*(np.log(np.maximum(infotem[:,0]+infotem[:,3],acu))-np.log(np.maximum(infotem[:,0]-infotem[:,3],acu)))
    infotem[:,5] = np.int32(data[:,0])
    infotem[:,6] = 0.5*(np.log(np.maximum(pmag+infotem[:,3],acu))-np.log(np.maximum(pmag-infotem[:,3],acu)))
    return infotem




def format_output(path,reso_decay,nsampling,run_afterburner,model):


    
    if run_afterburner :
        if model == "SMASH":
            infortem = format_SMASH_h5(path)
        elif model == "URQMD":
            infortem = format_URQMD_h5(path)
    else:
        if reso_decay:
            infortem = format_SMASH_h5(path)
        else:
            if model == "SMASH":
                infortem = format_SMASH_dat(path)
            elif model == "URQMD":
                infortem = format_URQMD_dat(path)
    return infortem
    #calc_obervables(infortem,nsampling,reso_decay,run_afterburner)     



    # if reso_decay == "TRUE":
    #     path = os.path.join("{event_dir}/final".format(event_dir=path))
    #     infoall=0
    #     with h5py.File(os.path.join(path,"particle_lists.h5"),"r") as f:
    #         data = f["particle_list"]            
    #         nprt  = len(data["p0"][...])
    #         infoall = np.zeros((nprt,8))

    #         infoall[:,0] = data["p0"][...]
    #         infoall[:,1] = data["px"][...]
    #         infoall[:,2] = data["py"][...]
    #         infoall[:,3] = data["pz"][...]
          
          
    #         pmag = np.sqrt(infoall[:,3]**2+infoall[:,1]**2+infoall[:,2]**2)
    #         infoall[:,4] = 0.5*(np.log(infoall[:,0]+infoall[:,3])-np.log(infoall[:,0]-infoall[:,3]))
    #         infoall[:,5] = data["pdg"][...]
    #         infoall[:,6] = 0.5*(np.log(pmag+infoall[:,3])-np.log(pmag-infoall[:,3]))
    #         infoall[:,7] = data["charge"][...]

    #     select_charged = np.fabs(infoall[:,7]) > 0
    #     infotem = np.zeros((len(infoall[select_charged,0]),7))
    #     infotem[:,:] = infoall[select_charged,:-1]   

    # else:
    #     path = os.path.join("{event_dir}/before".format(event_dir=cfg.event_dir))
    #     fpaths = glob.glob(os.path.abspath("{path}/mc_particle_list*".format(path=path)))
    #     data = pd.read_csv(fpaths[0],sep=" ",comment="#",header=None)
    #     data = data.values
    #     if len(fpaths)>1:
    #         for fpathid in fpaths[1:]:
    #             data_tep = pd.read_csv(fpathid,sep=" ",comment="#",header=None)
    #             data = np.concatenate((data,data_tep))

    #     select_charged = np.fabs(data[:,11]) > 0
    #     num_charged = len(data[select_charged,0])
    #     infotem = np.zeros((num_charged,7))
    #     infotem[:,0] = data[select_charged,5]
    #     infotem[:,1] = data[select_charged,6]
    #     infotem[:,2] = data[select_charged,7]
    #     infotem[:,3] = data[select_charged,8]
    #     pmag = np.sqrt(infotem[:,3]**2+infotem[:,1]**2+infotem[:,2]**2)
    #     infotem[:,4] = 0.5*(np.log(infotem[:,0]+infotem[:,3])-np.log(infotem[:,0]-infotem[:,3]))
    #     infotem[:,5] = np.int32(data[select_charged,9])
    #     infotem[:,6] = 0.5*(np.log(pmag+infotem[:,3])-np.log(pmag-infotem[:,3]))
     
     

   
 

    
    
    

if __name__ =="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--event_dir", required=True, help="path to folder containing clvisc output")
    parser.add_argument("--reso_decay", default=False, help="path to folder containing clvisc output")
    parser.add_argument("--nsampling", default='2000', help="number of over-sampling from one hyper-surface")
    parser.add_argument("--run_afterburner", default=True, help="Option to switch on/off afterburner ")
    parser.add_argument("--model", default='SMASH', help="afterburner model")

    cfg = parser.parse_args()
    print (cfg)
 
    if cfg.reso_decay.upper()=="FALSE" or cfg.reso_decay.upper()=="NO":
        cfg.reso_decay=False
    else:
        cfg.reso_decay=True
    if cfg.run_afterburner.upper()=="FALSE" or cfg.run_afterburner.upper()=="NO":
        cfg.run_afterburner=False
    else:
        cfg.run_afterburner=True


    infortem = format_output(path=cfg.event_dir,nsampling=cfg.nsampling,\
        reso_decay=cfg.reso_decay,run_afterburner=cfg.run_afterburner,model=cfg.model)
    if (not cfg.reso_decay) and (not cfg.run_afterburner):
        fpath = os.path.join(cfg.event_dir,"before")
    else:
        fpath = os.path.join(cfg.event_dir,"final")
    calc_obervables(fpath, infortem,cfg.nsampling,cfg.reso_decay,cfg.run_afterburner)    
