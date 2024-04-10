#/usr/bin/env python
#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>


import numpy as np
from subprocess import call
import os
import argparse
import pandas as pd
import glob
import h5py
parser = argparse.ArgumentParser()
parser.add_argument("--event_dir", required= True, help="path to folder containing clvisc output")
parser.add_argument("--eos_type", default= 'lattice_pce165', help="the eos type")
parser.add_argument("--reso_decay", default= 'True', help="true to switch_on resonance decay")
parser.add_argument("--nsampling", default='2000', help="number of over-sampling from one hyper-surface")
parser.add_argument("--mode", default='smooth', help="options:[smooth, mc]")
parser.add_argument("--gpu_id", default='0', help="for smooth spectra, one can choose gpu for parallel running")
parser.add_argument("--afterburner", default='SMASH', help="ouput format for afterburner")
parser.add_argument("--vorticity_on", default='0', help="true to switch on voritcity sample")

cfg = parser.parse_args()

print(cfg.event_dir)
event_dir = os.path.abspath(cfg.event_dir)

src_dir = os.path.dirname(os.path.realpath(__file__))

if cfg.mode.upper() == 'SMOOTH':
    dir_smooth_spec = os.path.join(src_dir, '../CLSmoothSpec/build')
    if not os.path.exists(dir_smooth_spec):
        os.makedirs(dir_smooth_spec)
    os.chdir(dir_smooth_spec)
    
    call(['./spec', event_dir,  cfg.reso_decay, cfg.eos_type, cfg.gpu_id])
    #print(['./spec', event_dir, cfg.Tfrz_on, cfg.reso_decay, cfg.eos_type, cfg.gpu_id])
elif cfg.mode.upper() == 'MC':
    dir_mc_spec = os.path.join(src_dir, '../sampler/sampler_cpu/build')
    os.chdir(dir_mc_spec)
    if cfg.reso_decay.upper() == 'TRUE':
        call(['./main','%s'%event_dir,cfg.reso_decay,cfg.eos_type,cfg.nsampling,"0",cfg.afterburner.upper()])
        os.chdir(event_dir)
        if not os.path.exists(os.path.join("./","final")):
            os.makedirs(os.path.join("./","final"))
        fpaths = glob.glob("mc_particle_list*")
        particle_list = pd.read_csv(fpaths[0],sep=" ",comment="#",header=None)
        particle_list = particle_list.values
        if len(fpaths)>1:
            for fpath in fpaths[1:]:
                particle_list_tep = pd.read_csv(fpath,sep=" ",comments="#",header=None)
                particle_list = np.concatenate((particle_list,particle_list_tep))
        names = "t x y z mass p0 px py pz pdg ID charge"
        names =  names.split()
        types = ["f","f","f","f","f","f8","f8","f8","f8","i","i","i"]

        with h5py.File("particle_lists.h5","w") as f:
            for id in range(len(names)):
                f.create_dataset("particle_list/%s"%names[id],data=particle_list[:,id],dtype=types[id], compression='gzip')
            

        call("rm mc_particle_list*",shell=True)
        call("mv particle_lists.h5 ./final",shell=True)

    else:
        call(['./main','%s'%event_dir,cfg.reso_decay,cfg.eos_type,cfg.nsampling,"0",cfg.afterburner.upper()])
        os.chdir(event_dir)
        if not os.path.exists(os.path.join("./","before")):
            os.makedirs(os.path.join("./","before"))
        call("mv mc_particle_list* ./before",shell=True)        
    os.chdir(src_dir)
elif cfg.mode.upper() == "MCGPU":
    dir_mc_spec = os.path.join(src_dir, '../sampler/sampler_gpu/build')
    os.chdir(dir_mc_spec)
    if cfg.reso_decay.upper() == 'TRUE':
        cfg.afterburner = "SMASH"
        call(['./main','%s'%event_dir,'1',cfg.eos_type,cfg.nsampling,cfg.gpu_id,cfg.afterburner.upper(),"0"])
        os.chdir(event_dir)
        if not os.path.exists(os.path.join("./","before")):
            os.makedirs(os.path.join("./","before"))
        call("mv mc_particle_list* ./before",shell=True)
        os.chdir(src_dir)
        dir_decay = os.path.join(src_dir,"../sampler/sampler_cpu/build")
        if not os.path.exists(dir_decay):
            os.makedirs(dir_decay)
        os.chdir(dir_decay)
        call(['./main', event_dir, cfg.reso_decay,cfg.eos_type,cfg.nsampling,"1",cfg.afterburner.upper()])
        
        os.chdir(event_dir)
        if not os.path.exists(os.path.join("./","final")):
            os.makedirs(os.path.join("./","final"))
        
        fpaths = glob.glob("mc_particle_list*")
        particle_list = pd.read_csv(fpaths[0],sep=" ",comment="#",header=None)
        particle_list = particle_list.values
        if len(fpaths)>1:
            for fpath in fpaths[1:]:
                particle_list_tep = pd.read_csv(fpath,sep=" ",comment="#",header=None)
                particle_list = np.concatenate((particle_list,particle_list_tep))

        names = "t x y z mass p0 px py pz pdg ID charge"
        names =  names.split()
        types = ["f","f","f","f","f","f8","f8","f8","f8","i","i","i"]

        with h5py.File("particle_lists.h5","w") as f:
            for id in range(len(names)):
                f.create_dataset("particle_list/%s"%names[id],data=particle_list[:,id],dtype=types[id], compression='gzip')

        call("rm mc_particle_list*",shell=True)
        call("mv particle_lists.h5 ./final",shell=True)
        #call("mv mc_particle_list* ./final",shell=True)
        os.chdir(src_dir)
    else:
        call(['./main','%s'%event_dir,'0',cfg.eos_type,cfg.nsampling,cfg.gpu_id,cfg.afterburner.upper(),cfg.vorticity_on.upper()])
        os.chdir(event_dir)
        if not os.path.exists(os.path.join("./","before")):
            os.makedirs(os.path.join("./","before"))
        call("mv mc_particle_list* ./before",shell=True)
        os.chdir(src_dir)
    os.chdir(src_dir)
else:
    print("Choose from 'smooth' and 'mc' for spectra calc")
