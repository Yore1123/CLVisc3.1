#/usr/bin/env python

#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>
#Copyright (c) 2024- Jun-Qi Tao <taojunqi@mails.ccnu.edu.cn>


import numpy as np
from subprocess import call
import os
from time import time
from glob import glob
import pyopencl as cl
import matplotlib.pyplot as plt
import h5py
from scipy.interpolate import InterpolatedUnivariateSpline
import os, sys

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.abspath(os.path.join(cwd, '../pyvisc')))
from visc import CLVisc

def from_sd_to_ed(entropy, eos):
    '''using eos to  convert the entropy density to energy density'''
    s = eos.s
    ed = eos.ed
    # the InterpolatedUnivariateSpline works for both interpolation
    # and extrapolation
    f_ed = InterpolatedUnivariateSpline(s, ed, k=1)
    return f_ed(entropy)





def read_p4x4(cent='30_35', idx=0,
        fname='/u/lpang/hdf5_data/auau200_run1.h5'):
    '''read 4-momentum and 4-coordiantes from h5 file,
    return: np.array with shape (num_of_partons, 8)
    the first 4 columns store: E, px, py, pz
    the last 4 columns store: t, x, y, z'''
    with h5py.File(fname, 'r') as f:
        grp = f['cent']
        event_id = grp[cent][:, 0].astype(np.int)

        impact = grp[cent][:, 1]
        nw = grp[cent][:, 2]
        nparton = grp[cent][:, 3]
        key = 'event%s'%event_id[idx]
        p4x4 = f[key]
        return p4x4[...], event_id[idx], impact[idx], nw[idx], nparton[idx]

def get_jet_info(cfg,idx):
    fname = cfg.parton_dat
    print(fname)
    with h5py.File(fname,"r") as f:
        print (f.keys(),'cent_%s/pt_min_max_%d_%d/event%d'%(cfg.cent,cfg.pthatmin,cfg.pthatmax,idx))
        grp = f['cent_%s/pt_min_max_%d_%d/event%d'%(cfg.cent,cfg.pthatmin,cfg.pthatmax,idx)][...]
        nptc = f['cent_%s/pt_min_max_%d_%d/nptc'%(cfg.cent,cfg.pthatmin,cfg.pthatmax)][...]
        ppsigma = f['cent_%s/pt_min_max_%d_%d/sigma'%(cfg.cent,cfg.pthatmin,cfg.pthatmax)][...]
        random_seed = f['cent_%s/pt_min_max_%d_%d/random_seed'%(cfg.cent,cfg.pthatmin,cfg.pthatmax)][...]
    
        print ("www" , nptc[idx],ppsigma[0,0],random_seed[idx])
        header0 = """num_of_parton cross_section[mb] random_seed  
pid px py pz mass x y formation_time 
%s %s %s """%(nptc[idx],ppsigma[0,0],random_seed[idx])

    if not os.path.exists(cfg.fPathOut):
        os.makedirs(cfg.fPathOut)
    fout = os.path.abspath(os.path.join(cfg.fPathOut, 'parton_info_%s_%s_%s.dat'%(cfg.cent,cfg.pthatmin,cfg.pthatmax)))
    np.savetxt(fout,grp,fmt="%d %f %f %f %f %f %f %f ",header=header0)



def get_ampt(cfg,gpu_id,idx):
    visc = CLVisc(cfg, gpu_id=gpu_id)

    parton_list, eid, imp_b, nwound, npartons = read_p4x4(cfg.cent, idx, cfg.Initial_profile)
    parton_list = np.array(parton_list) 
    NEVENT = 1    
    if cfg.oneshot:
        NEVENT = 250    
        for i in range(1,NEVENT):
            print (i)
            parton_list1, eid, imp_b, nwound, npartons = read_p4x4(cfg.cent, i, cfg.Initial_profile) 
            parton_list = np.concatenate((parton_list,np.array(parton_list1))) 
 
    comments = 'cent=%s, eventid=%s, impact parameter=%s, nw=%s, npartons=%s'%(
    cfg.cent, eid, imp_b, nwound, npartons)
    visc.smear_from_parton_list_ampt(parton_list,NEVENT)

        
    return visc


def read_pdg(fname):
    '''
    read the paticles list in SMASH and get the baryon PDG code
    '''
    pdgpath = os.path.join(fname, "particles.txt")
    baryon_code = ["N","\xce\x94","\xce\x9b","\xce\xa3","\xce\xa9","\xce\x9e"]
    with open(pdgpath,"r") as f:
        lines = f.readlines()
    PDGbaryon = []

    for line in lines[1:]:
        for PDGid in baryon_code:
            if PDGid in line and "####" not in line:
                for baryonid in line.split()[4:]:
                    try:
                        PDGbaryon.append(int(baryonid))
                    except:
                        pass
    
    return PDGbaryon

def read_hadron_list(cent='30_35', idx=0,
        fname='/u/lpang/hdf5_data/auau200_run1.h5'):
    # '''read 4-momentum and 4-coordiantes from h5 file,
    # return: np.array with shape (num_of_partons, 8)
    # the first 4 columns store: E, px, py, pz
    # the last 4 columns store: t, x, y, z'''

    with h5py.File(fname, 'r') as f:
        grp = f['cent']['%s'%cent]["event%d"%idx][...]
    
    return grp



def get_smash(cfg,gpu_id,idx):

    visc = CLVisc(cfg, gpu_id=gpu_id)
    PDG_CODE = read_pdg(os.path.join(cwd,"eos/eos_table"))

    hadron_list = read_hadron_list(cfg.cent, idx, cfg.Initial_profile)
    hadron_list = np.array(hadron_list)
    print ( np.sqrt(hadron_list[:,0]**2 - hadron_list[:,3]**2))


    NEVENT = 1
    if cfg.oneshot:
        NEVENT =400 
        for i in range(1,NEVENT):
            print (i)
            hadron_list1 = read_hadron_list(cfg.cent, i, cfg.Initial_profile)
            hadron_list = np.concatenate((hadron_list,np.array(hadron_list1)))



    visc.smear_from_hadron_list_smash(cfg,hadron_list,PDG_CODE,NEVENT)
    return visc

        
def get_trento_1_3(cfg,gpu_id,idx):
    from ini.trento import AuAu200, PbPb2760, PbPb5020, RuRu200, Ru2Ru2200, Ru3Ru3200, Ru4Ru4200, ZrZr200, AuAu165
    dec_num = np.modf(cfg.SQRTS)[0]
    if dec_num < 0.00001:
        system = cfg.NucleusA + cfg.NucleusB+str(int(np.floor(cfg.SQRTS)))
    else:
        system = cfg.NucleusA+cfg.NucleusB+str(cfg.SQRTS)
        system = system.replace(".","p")
    
    if system == 'PbPb2760':
        collision = PbPb2760()
    elif system == 'PbPb5020':
        collision = PbPb5020()
    elif system == 'AuAu200':
        collision = AuAu200()
    elif system == 'RuRu200':
        collision = RuRu200()
    elif system == 'Ru2Ru2200':
        collision = Ru2Ru2200()
    elif system == 'Ru3Ru3200':
        collision = Ru3Ru3200()
    elif system == 'Ru4Ru4200':
        collision = Ru4Ru4200()
    elif system == 'ZrZr200':
        collision = ZrZr200()
    elif system == 'AuAu165':
        collision = AuAu165()
    else:
        print ("ERROR: Please check collision system !!!", system)
        exit(1)
        
    grid_max = cfg.NX/2 * cfg.DX
   

    fini = os.path.join(cfg.fPathOut , 'trento_ini/')
    
    if os.path.exists(fini):
       call(['rm', '-r', fini])
    
    cwd = os.getcwd()
    os.chdir("../3rdparty/trento_1_3_with_participant_plane/build/src/")
    if cfg.oneshot:
        collision.create_ini_1_3(cfg.cent, fini, num_of_events=200,
                                 grid_max=grid_max, grid_step=cfg.DX,
                                 one_shot_ini=cfg.oneshot,
                                 reduced_thickness=cfg.reduced_thickness, fluctuation=cfg.fluctuation, nucleon_width=cfg.nucleon_width, 
                                 normalization=cfg.normalization)
        s = np.loadtxt(os.path.join(fini, 'one_shot_ini.dat'))
    else:
        collision.create_ini_1_3(cfg.cent, fini, num_of_events=1,
                                 grid_max=grid_max, grid_step=cfg.DX,
                                 one_shot_ini=cfg.oneshot, 
                                 reduced_thickness=cfg.reduced_thickness, fluctuation=cfg.fluctuation, nucleon_width=cfg.nucleon_width, 
                                 normalization=cfg.normalization)
        s = np.loadtxt(os.path.join(fini, '0.dat'))
    
    os.chdir(cwd)
    
    s_scale = s * cfg.KFACTOR
    visc = CLVisc(cfg, gpu_id=gpu_id)
    visc.trento2D_ini(s_scale)
     
    return  visc

def get_trento_2_0(cfg,gpu_id,idx):
    from ini.trento import AuAu200, PbPb2760, PbPb5020, RuRu200, Ru2Ru2200, Ru3Ru3200, Ru4Ru4200, ZrZr200, AuAu165
    dec_num = np.modf(cfg.SQRTS)[0]
    if dec_num < 0.00001:
        system = cfg.NucleusA + cfg.NucleusB+str(int(np.floor(cfg.SQRTS)))
    else:
        system = cfg.NucleusA+cfg.NucleusB+str(cfg.SQRTS)
        system = system.replace(".","p")
    
    if system == 'PbPb2760':
        collision = PbPb2760()
    elif system == 'PbPb5020':
        collision = PbPb5020()
    elif system == 'AuAu200':
        collision = AuAu200()
    elif system == 'RuRu200':
        collision = RuRu200()
    elif system == 'Ru2Ru2200':
        collision = Ru2Ru2200()
    elif system == 'Ru3Ru3200':
        collision = Ru3Ru3200()
    elif system == 'Ru4Ru4200':
        collision = Ru4Ru4200()
    elif system == 'ZrZr200':
        collision = ZrZr200()
    elif system == 'AuAu165':
        collision = AuAu165()
    else:
        print ("ERROR: Please check collision system !!!", system)
        exit(1)
        
    grid_max = cfg.NX/2 * cfg.DX
   

    fini = os.path.join(cfg.fPathOut , 'trento_ini/')

    if os.path.exists(fini):
       call(['rm', '-r', fini])
    
    cwd = os.getcwd()
    os.chdir("../3rdparty/trento_2_0_with_participant_plane/build/src/")
    if cfg.oneshot:
        collision.create_ini_2_0(cfg.cent, fini, num_of_events=200,
                                 grid_max=grid_max, grid_step=cfg.DX,
                                 one_shot_ini=cfg.oneshot,
                                 reduced_thickness=cfg.reduced_thickness, fluctuation=cfg.fluctuation, nucleon_width=cfg.nucleon_width, 
                                 constit_width=cfg.constit_width, constit_number=cfg.constit_number, nucleon_min_dist=cfg.nucleon_min_dist, normalization=cfg.normalization)
        s = np.loadtxt(os.path.join(fini, 'one_shot_ini.dat'))
    else:
        collision.create_ini_2_0(cfg.cent, fini, num_of_events=1,
                                 grid_max=grid_max, grid_step=cfg.DX,
                                 one_shot_ini=cfg.oneshot, 
                                 reduced_thickness=cfg.reduced_thickness, fluctuation=cfg.fluctuation, nucleon_width=cfg.nucleon_width, 
                                 constit_width=cfg.constit_width, constit_number=cfg.constit_number, nucleon_min_dist=cfg.nucleon_min_dist, normalization=cfg.normalization)
        s = np.loadtxt(os.path.join(fini, '0.dat'))
    os.chdir(cwd)
    
    s_scale = s * cfg.KFACTOR
    visc = CLVisc(cfg, gpu_id=gpu_id)
    visc.trento2D_ini(s_scale)
     
    return  visc      



def get_glauber(cfg,gpu_id):
    visc = CLVisc(cfg, gpu_id=gpu_id)
    visc.optical_glauber_ini()
    return visc


def get_mcglauber(cfg,gpu_id):
    visc = CLVisc(cfg, gpu_id=gpu_id)
    from mcglauber import MCglauber
    ini = MCglauber(visc)
    if cfg.oneshot:
        ini.ave_event(Nave=200)
    else:
        ini.ebe_event()
    
    return visc

def get_handcrafted(cfg,gpu_id,idx):
    visc = CLVisc(cfg, gpu_id=gpu_id)
    visc.load_handcrafted_initial_condition()
    return visc

def Colhydro_initial(cfg,gpu_id,idx):
    get_jet_info(cfg,idx)


def set_initial_condition(cfg,gpu_id,idx=0):

    if ( os.path.exists(os.path.join(cfg.fPathOut,"dNdEtaPtdPtdPhi_Charged.dat")) or \
    os.path.exists(os.path.join(cfg.fPathOut,"final/particle_lists.h5")) ):
        print(cfg.fPathOut," exists, skip it .....")
        return "finish"
    #elif os.path.exists(os.path.abspath(cfg.fPathOut)):
    #    call("rm -r %s"%(os.path.abspath(cfg.fPathOut)),shell = True)

    if cfg.Colhydro:
        Colhydro_initial(cfg,gpu_id,idx)


    if cfg.initial_type.upper() == "GLAUBER":
        return get_glauber(cfg,gpu_id)
    if cfg.initial_type.upper() == 'MCGLAUBER':
        return get_mcglauber(cfg,gpu_id)
    if cfg.initial_type.upper() == 'AMPT':
        return get_ampt(cfg,gpu_id,idx)
    if cfg.initial_type.upper() == 'SMASH':
        return get_smash(cfg,gpu_id,idx)
    if cfg.initial_type.upper() == 'TRENTO_1_3':
        return get_trento_1_3(cfg,gpu_id,idx)
    if cfg.initial_type.upper() == 'TRENTO_2_0':
        return get_trento_2_0(cfg,gpu_id,idx)
    if cfg.initial_type.upper() == 'HANDCRAFTED':
        return get_handcrafted(cfg,gpu_id,idx)
    


    
    



def calc_freezeout_and_afterburner(cfg,gpu_id):
    cwd, cwf = os.path.split(__file__)
    t0 = time() 
    if cfg.Colhydro:
        colbt_path  = os.path.join(cwd,"CoLBT")
        os.chdir(colbt_path)
        if cfg.run_cooperfrye :
            call(['python', 'spec_colbt.py', '--event_dir', cfg.fPathOut,\
            "--reso_decay", str(cfg.reso_decay).upper(),\
            "--eos_type","%s"%cfg.eos_type.upper(), '--mode', cfg.sample_type,\
            "--gpu_id",'%s'%gpu_id,"--nsampling",cfg.nsample,"--afterburner",cfg.afterburner,"--vorticity_on",str(cfg.calc_vorticity)])
        os.chdir(cwd)

        if not cfg.reso_decay and cfg.run_afterbuner:
            call(['python', 'after_burner.py','--event_dir', cfg.fPathOut,'--nsampling',cfg.nsample,'--model',cfg.afterburner.upper()])   

    else:

        if cfg.run_cooperfrye :
            call(['python', 'spec.py', '--event_dir', cfg.fPathOut,\
            "--reso_decay", str(cfg.reso_decay).upper(),\
            "--eos_type","%s"%cfg.eos_type.upper(), '--mode', cfg.sample_type,\
            "--gpu_id",'%s'%gpu_id,"--nsampling",cfg.nsample,"--afterburner",cfg.afterburner,"--vorticity_on",str(cfg.calc_vorticity)])
        
        if not cfg.reso_decay and cfg.run_afterbuner:
            call(['python', 'after_burner.py','--event_dir', cfg.fPathOut,'--nsampling',cfg.nsample,'--model',cfg.afterburner.upper()])   
        
    if  cfg.run_cooperfrye:
        if cfg.sample_type.upper() != "SMOOTH":
            cwd = os.path.abspath(os.getcwd())
            dir_mc_spec = os.path.join(cwd, '../sampler/analysis_code/')
            os.chdir(dir_mc_spec)
            call(['python','Calcspec.py','--event_dir',cfg.fPathOut, \
                "--reso_decay",str(cfg.reso_decay),"--nsampling",cfg.nsample,\
                "--run_afterburner",str(cfg.run_afterbuner),"--model",cfg.afterburner.upper()])
            os.chdir(cwd)
        else:
            if str(cfg.reso_decay).upper() == 'TRUE':
                call(['python', '../spec/main.py', cfg.fPathOut, '1'])
            else:
               call(['python', '../spec/main.py', cfg.fPathOut, '0'])

    t1 = time() 
    print ("It costs %s s"%(t1-t0))
    
        
