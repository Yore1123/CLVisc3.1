#/usr/bin/env python




import numpy as np
from subprocess import call,Popen
import os
import argparse
import glob
import pandas as pd
import h5py
import numpy as np
src_dir = os.path.dirname(os.path.realpath(__file__))

def run_smash(fpathout = "./", shift_id = 0, nsampling=2000):
    fout1 = os.path.join(fpathout,"before")
    config_src = '''Version: 1.8
Logging:
    default: INFO

General:
    Modus:         List
    Time_Step_Mode: None
    Delta_Time:    0.1
    End_Time:      5000 
    Randomseed:    -1
    Nevents:       {nsampling}

#Collision_Term:
    #Force_Decays_At_End: False
    #Two_to_One: False
#    No_Collisions: True

Output:
    Output_Interval: 100.0
    Particles:
        Format:          ["Oscar2013"]
        Extended: True
Modi:
    List:
        File_Directory: "{fpathout}"
        File_Prefix: "mc_particle_list"
        Shift_Id: {shift_id} 
    
    '''.format(fpathout =fout1,shift_id = shift_id,nsampling=nsampling)
    
    #print (fpathout)
    fscript = os.path.join(fpathout, "script")
    #print (fscript)
    if not os.path.exists(fscript):
        os.makedirs(fscript)

    job_name = "config_%s.yaml"%(shift_id)
    config_path = os.path.join(fscript,job_name)
    with open(config_path, "w") as fout:
        fout.write(config_src)
    smash_path = os.path.join(src_dir,"../3rdparty/smash_afterburner/smash/build")
    os.chdir(smash_path)
    
    final_path = os.path.join(fpathout,"final/%d"%shift_id)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    
    call(['./smash', "-i",config_path,"-o",final_path])
    #Popen(['./smash', "-i",config_path,"-o",final_path])
    #print('./smash', "-i",config_path,"-o",final_path)



def run_URQMD(fpathout = "./", shift_id = 0):
    urqmd_path = os.path.join(src_dir,"../3rdparty/urqmd_afterburner/")
    os.chdir(urqmd_path)
    finput = os.path.join(fpathout,"before/mc_particle_list%d"%shift_id)
    fintermediate = os.path.join(fpathout,"before/urqmd_input%d.dat"%shift_id)
    final_path = os.path.join(fpathout,"final/%d"%shift_id)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    foutput = os.path.join(final_path,"%d.dat"%shift_id)
    print (foutput)
    call(['./afterburner',finput,fintermediate,foutput])
    call('rm %s'%fintermediate,shell=True)




__cwd__,__cwf__ =os.path.split(__file__)


parser = argparse.ArgumentParser()
parser.add_argument("--event_dir", required=True, help="path to folder containing clvisc output")
parser.add_argument("--nsampling", required=True, help="the number of oversample particles")
parser.add_argument("--model", default="SMASH", help="the number of oversample particles")


cfg = parser.parse_args()

if cfg.model.upper() == "SMASH":

    for i in range(1):
        run_smash(fpathout = cfg.event_dir,shift_id = i,nsampling=cfg.nsampling)
    os.chdir(cfg.event_dir)


    fpaths = glob.glob(os.path.join(cfg.event_dir,"final/*/particle_lists.oscar"))
    particle_list = pd.read_csv(fpaths[0],sep=" ",comment="#",header=None)
    particle_list = particle_list.values
    if len(fpaths)>1:
        for fpath in fpaths[1:]:
            particle_list_tep = pd.read_csv(fpath,sep=" ",comment="#",header=None)
            particle_list_tep = particle_list_tep.values
            particle_list = np.concatenate((particle_list,particle_list_tep))
    
    names = "t x y z mass p0 px py pz pdg ID charge ncoll form_time xsecfac proc_id_origin proc_type_origin time_last_coll pdg_mother1 pdg_mother2"
    names =  names.split()
    types = ["i","f","f","f","f","f8","f8","f8","f8","i","i","i","i","f","f","i","i","f","i","i"]

    with h5py.File("particle_lists.h5","w") as f:
        for id in range(len(names)):
            f.create_dataset("particle_list/%s"%names[id],data=particle_list[:,id],dtype=types[id], compression='gzip')

            

    call("rm ./final/*/particle_lists.oscar",shell=True)

    


elif cfg.model.upper()== "URQMD":
    for i in range(1):
        run_URQMD(fpathout = cfg.event_dir,shift_id = i)
    os.chdir(cfg.event_dir)

    fpaths = glob.glob(os.path.join(cfg.event_dir,"final/*/*.dat"))
    particle_list = np.loadtxt(fpaths[0])
    
    if len(fpaths)>1:
        for fpath in fpaths[1:]:
            particle_list_tep = np.loadtxt(fpath)
            particle_list = np.concatenate((particle_list,particle_list_tep))
    
    with h5py.File("particle_lists.h5","w") as f:
        f.create_dataset("particle_list/p0",data=particle_list[:,4]*np.cosh(particle_list[:,6]),dtype='f8', compression='gzip')
        f.create_dataset("particle_list/px",data=particle_list[:,2]*np.cos(particle_list[:,5]),dtype='f8', compression='gzip')
        f.create_dataset("particle_list/py",data=particle_list[:,2]*np.sin(particle_list[:,5]),dtype='f8', compression='gzip')
        f.create_dataset("particle_list/pz",data=particle_list[:,4]*np.sinh(particle_list[:,6]),dtype='f8', compression='gzip')
        f.create_dataset("particle_list/rap",data=particle_list[:,6],dtype='f8', compression='gzip')
        f.create_dataset("particle_list/eta",data=particle_list[:,7],dtype='f8', compression='gzip')
        f.create_dataset("particle_list/pdg",data=particle_list[:,0],dtype='i', compression='gzip')
        f.create_dataset("particle_list/charge",data=particle_list[:,1],dtype='i', compression='gzip')

    call("rm ./final/*/*.dat",shell=True)

call("mv particle_lists.h5 ./final",shell=True)    
os.chdir(src_dir)








