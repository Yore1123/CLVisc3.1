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
print (src_dir)

def copy_pdgfile(mode):

    if mode.upper() == "SMASH":
        call(["cp",os.path.join(src_dir,"../eos/eos_table/pdg_smash.dat"),os.path.join(event_dir,"pdgfile.dat")])
    elif mode.upper() == "URQMD":
        call(["cp",os.path.join(src_dir,"../eos/eos_table/pdg-urqmd.dat"),os.path.join(event_dir,"pdgfile.dat")])
    elif mode.upper() == "QUARK":
        call(["cp",os.path.join(src_dir,"../eos/eos_table/pdg_quark.dat"),os.path.join(event_dir,"pdgfile.dat")])
    else:
        call(["cp",os.path.join(src_dir,"../eos/eos_table/pdg05.dat"),os.path.join(event_dir,"pdgfile.dat")])




if cfg.mode.upper() == "MCGPU":
    dir_mc_spec = os.path.abspath(os.path.join(src_dir, '../../sampler/sampler_gpu/build'))
    dir_coal_spec = os.path.abspath(os.path.join(src_dir, 'Coal_Frag/Coal'))
    dir_frag_spec = os.path.abspath(os.path.join(src_dir, 'Coal_Frag/Fragmentation'))
    dir_convert_spec = os.path.abspath(os.path.join(src_dir, 'Coal_Frag/to_afterburner'))
    
    
    if cfg.reso_decay.upper() == 'TRUE':
        print("Please set the reso_decay to False! ")
        exit(1)
    else:
        
        copy_pdgfile("QUARK")
        os.chdir(dir_mc_spec)
        call(['./main','%s'%event_dir,'0',cfg.eos_type,cfg.nsampling,cfg.gpu_id,cfg.afterburner.upper(),cfg.vorticity_on.upper()])
        os.chdir(event_dir)
        if not os.path.exists(os.path.join("./","before")):
            os.makedirs(os.path.join("./","before"))
        call("mv mc_particle_list0 ./before/thermal_parton.dat",shell=True)
        call("mv tc_inf.dat ./before/shower_parton.dat",shell=True)
        
        os.chdir(dir_mc_spec)
        copy_pdgfile(cfg.afterburner.upper())
        call(['./main','%s'%event_dir,'0',cfg.eos_type,cfg.nsampling,cfg.gpu_id,cfg.afterburner.upper(),cfg.vorticity_on.upper()])
        os.chdir(event_dir)
        call("mv mc_particle_list0 ./before",shell=True)
        
        os.chdir(dir_coal_spec)
        call(['./main','%s'%os.path.join(event_dir,"before"),"1"])
        
        os.chdir(dir_frag_spec)
        call(['./main_string_fragmentation','0','1','%s'%os.path.join(event_dir,"before"),'%s'%os.path.join(event_dir,"before")])
        
        os.chdir(dir_convert_spec)
        call(['./main','1','%s'%os.path.join(event_dir,"before"),cfg.afterburner.upper()])

        os.chdir(event_dir)
        call("mv ./before/oscar.dat ./before/mc_particle_list0 ",shell=True)
        


        
        os.chdir(src_dir)
    os.chdir(src_dir)
else:
    print("Choose from 'smooth' and 'mc' for spectra calc")
