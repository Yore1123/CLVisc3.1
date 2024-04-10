#/usr/bin/env python

import numpy as np
from subprocess import call
import os
from time import time
from glob import glob
import pyopencl as cl
import matplotlib.pyplot as plt
import h5py

import os, sys
cwd, cwf = os.path.split(__file__)

sys.path.append(os.path.abspath(os.path.join(cwd, '../pyvisc')))
from config import read_config, write_config
from bridge import set_initial_condition,calc_freezeout_and_afterburner
from logo import print_logo


print (sys.version_info)



if __name__ == '__main__':
    
    if len(sys.argv)!= 5:
        print (" usage: python config.py hydro.info gpu_id start_id end_id")
    
    fconfig = sys.argv[1]
    gpu_id = int(sys.argv[2])
    start_id = int(sys.argv[3])
    end_id = int(sys.argv[4])


    cfg = read_config(fconfig)
    fPathROOT = cfg.fPathOut
    jetinfo_ROOT = cfg.parton_dat
    
    print(cfg.initial_type)

    print_logo()
    for idx in range(start_id,end_id):
        
        cfg.fPathOut = os.path.abspath(os.path.join(fPathROOT, 'event%s'%idx))
        #if cfg.Colhydro:
        #    cfg.parton_dat = os.path.abspath(os.path.join(jetinfo_ROOT, 'parton_info_%s_%s_%s_%s.dat'%(cfg.cent,idx,cfg.ptmin,cfg.ptmax)))
        
        if cfg.run_evolution:
            hydro = set_initial_condition(cfg,gpu_id,idx)
            if hydro == "finish":
                continue
            write_config(cfg)
            hydro.evolve(max_loops=2000)
        calc_freezeout_and_afterburner(cfg,gpu_id)

        


    
