#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>
#Copyright (c) 2024- Jun-Qi Tao <taojunqi@mails.ccnu.edu.cn>

##Read default configeration from hydro.info
##Update it with input from command line options

import numpy as np
from pyopencl import array
import argparse
import os, sys

if sys.version_info <= (3, 0):
    import ConfigParser as configparser
else:
    import configparser

def write_config(configs, comments=''):
    '''write the current setting to hydro.info in the output directory'''
    fPathOut = configs.fPathOut
    if not os.path.exists(fPathOut):
        os.makedirs(fPathOut)

    configfile_name = os.path.join(fPathOut, 'hydro.info')
    #if not os.path.isfile(configfile_name):
    with open(configfile_name, 'w') as cfgfile:
        # Create the configuration file as it doesn't exist yet
        # Add content to the file
        Config = configparser.ConfigParser()
        Config.add_section('General')
        Config.set('General', 'fPathOut', configs.fPathOut)
        Config.add_section('initial_condition')
        Config.set('initial_condition', 'initial_type', configs.initial_type)





        Config.set('initial_condition', 'reduced_thickness', configs.reduced_thickness)
        Config.set('initial_condition', 'fluctuation', configs.fluctuation)
        Config.set('initial_condition', 'nucleon_width', configs.nucleon_width)
        Config.set('initial_condition', 'constit_width', configs.constit_width)
        Config.set('initial_condition', 'constit_number', configs.constit_number)
        Config.set('initial_condition', 'nucleon_min_dist', configs.nucleon_min_dist)
        Config.set('initial_condition', 'normalization', configs.normalization)





        Config.set('initial_condition', 'Initial_profile', configs.Initial_profile)
        Config.set('initial_condition', 'Initial_nb_profile', configs.Initial_nb_profile)
        Config.set('initial_condition', 'oneshot', str(configs.oneshot))
        Config.set('initial_condition', 'SQRTS', str(configs.SQRTS))
        Config.set('initial_condition', 'NucleusA', configs.NucleusA)
        Config.set('initial_condition', 'NucleusB', configs.NucleusB)
        Config.set('initial_condition', 'KFACTOR', str(configs.KFACTOR))
        Config.set('initial_condition', 'cent', configs.cent)
        Config.set('initial_condition', 'Hwn', str(configs.Hwn))
        Config.set('initial_condition', 'Eta_flat', str(configs.Eta_flat))
        Config.set('initial_condition', 'Eta_gw', str(configs.Eta_gw))
        Config.set('initial_condition', 'r_gw', str(configs.r_gw))
        Config.set('initial_condition', 'Eta0_nb', str(configs.Eta0_nb))
        Config.set('initial_condition', 'Eta_nb_gw_p', str(configs.Eta_nb_gw_p))
        Config.set('initial_condition', 'Eta_nb_gw_m', str(configs.Eta_nb_gw_m))

        Config.add_section('hydrodynamics')
        Config.set('hydrodynamics', 'TAU0', str(configs.TAU0))
        Config.set('hydrodynamics', 'NX', str(configs.NX))
        Config.set('hydrodynamics', 'NY', str(configs.NY))
        Config.set('hydrodynamics', 'NZ', str(configs.NZ))
        Config.set('hydrodynamics', 'ntskip', str(configs.ntskip))
        Config.set('hydrodynamics', 'nxskip', str(configs.nxskip))
        Config.set('hydrodynamics', 'nyskip', str(configs.nyskip))
        Config.set('hydrodynamics', 'nzskip', str(configs.nzskip))
        Config.set('hydrodynamics', 'DT', str(configs.DT))
        Config.set('hydrodynamics', 'DX', str(configs.DX))
        Config.set('hydrodynamics', 'DY', str(configs.DY))
        Config.set('hydrodynamics', 'DZ', str(configs.DZ))

        Config.set('hydrodynamics', 'pimn_on', str(configs.pimn_on))
        Config.set('hydrodynamics', 'CETA_YMIN', str(configs.CETA_YMIN))
        Config.set('hydrodynamics', 'CETA_XMIN', str(configs.CETA_XMIN))
        Config.set('hydrodynamics', 'CETA_LEFT_SLOP', str(configs.CETA_LEFT_SLOP))
        Config.set('hydrodynamics', 'CETA_RIGHT_SLOP', str(configs.CETA_RIGHT_SLOP))

        Config.set('hydrodynamics', 'bulkpr_on', str(configs.bulkpr_on))
        Config.set('hydrodynamics', 'CZETA_duke', str(configs.CZETA_duke))
        Config.set('hydrodynamics', 'CZETA', str(configs.CZETA))


        Config.set('hydrodynamics', 'baryon_on', str(configs.baryon_on))
        Config.set('hydrodynamics', 'qu_on', str(configs.qu_on))
        Config.set('hydrodynamics', 'CB_XMIN', str(configs.CB_XMIN))


        Config.set('hydrodynamics', 'source', str(configs.source))
        Config.set('hydrodynamics', 'calc_vorticity', str(configs.calc_vorticity))
        Config.set('hydrodynamics', 'save_bulkinfo', str(configs.save_bulkinfo))
        Config.set('hydrodynamics', 'save_pimn', str(configs.save_pimn))
        Config.set('hydrodynamics', 'save_qb', str(configs.save_qb))
        Config.set('hydrodynamics', 'save_bulkpr', str(configs.save_bulkpr))





        Config.add_section('EOS')
        Config.set('EOS', 'eos_type', str(configs.eos_type))
        Config.set('EOS', 'switch_eos', str(configs.switch_eos))
        
        Config.add_section('freezeeout_and_afterburner') 
        Config.set('freezeeout_and_afterburner', 'cornelius', str(configs.cornelius))
        Config.set('freezeeout_and_afterburner', 'corona', str(configs.corona))
        Config.set('freezeeout_and_afterburner', 'Tfrz_on', str(configs.Tfrz_on))
        Config.set('freezeeout_and_afterburner', 'TFRZ', str(configs.TFRZ))
        Config.set('freezeeout_and_afterburner', 'Edfrz', str(configs.Edfrz))
        Config.set('freezeeout_and_afterburner', 'sample_type', str(configs.sample_type))
        Config.set('freezeeout_and_afterburner', 'reso_decay', str(configs.reso_decay))
        Config.set('freezeeout_and_afterburner', 'nsample', str(configs.nsample))
        Config.set('freezeeout_and_afterburner', 'afterburner', str(configs.afterburner))
        
        Config.add_section('coLBT_hydro') 
        Config.set('coLBT_hydro', 'Colhydro', str(configs.Colhydro))
        Config.set('coLBT_hydro', 'trans_model', str(configs.trans_model))
        Config.set('coLBT_hydro', 'parton_dat', str(configs.parton_dat))
        Config.set('coLBT_hydro', 'jet_r_gw', str(configs.jet_r_gw))
        Config.set('coLBT_hydro', 'jet_eta_gw', str(configs.jet_eta_gw))
        Config.set('coLBT_hydro', 'Ecut', str(configs.Ecut))
        Config.set('coLBT_hydro', 'alphas', str(configs.alphas))
        Config.set('coLBT_hydro', 'pthatmin', str(configs.pthatmin))
        Config.set('coLBT_hydro', 'pthatmax', str(configs.pthatmax))
               
        Config.add_section('Test') 
        Config.set('Test', 'LAM1', str(configs.LAM1))
        Config.add_section('opencl') 
        Config.set('opencl', 'BSZ', str(configs.BSZ))

        Config.write(cfgfile)
        cfgfile.write('#comments: '+comments)


def read_config(config_file):
    '''read configeration from file, then update the value 
    with command line input if there is any'''
    
    if sys.version_info <= (3, 0):
        _parser = configparser.ConfigParser()
    else:
        _parser = configparser.ConfigParser(inline_comment_prefixes=';' )
    
    
    cwd, cwf = os.path.split(__file__)
    
    _parser.read(config_file)
    config = {}
    
    
    print (_parser.sections())

    try:
        config['fPathOut'] = (_parser.get('General', 'fPathOut'), 
            'The absolute path for output directory')
    except configparser.Error:
        config['fPathOut'] = ("../results/", 
            'The absolute path for output directory')
    
    
    try:
        config['initial_type'] = (_parser.get( 'initial_condition', 'initial_type'),
            'initial condition, choices = [MCGlauber, Glauber, Trento_1_3, Trento_2_0, ampt, handcradfted] for vanishing baryon density')
    
    except configparser.Error:
        config['initial_type'] = ("Glauber",
            'initial condition, choices = [MCGlauber, Glauber, Trento_1_3, Trento_2_0, ampt, handcradfted] for vanishing baryon density')
        


        
        
    try:
        config['reduced_thickness'] = (_parser.get( 'initial_condition', 'reduced_thickness'),
            '(For Trento) Reduced thickness parameter, and the default value is 0')
    
    except configparser.Error:
        config['reduced_thickness'] = (0,
            '(For Trento) Reduced thickness parameter, and the default value is 0')
        
        
    try:
        config['fluctuation'] = (_parser.get( 'initial_condition', 'fluctuation'),
            '(For Trento) Gamma distribution shape parameter k for nucleon fluctuations, and the default value is 1')
    
    except configparser.Error:
        config['fluctuation'] = (1,
            '(For Trento) Gamma distribution shape parameter k for nucleon fluctuations, and the default value is 1')
        
        
    try:
        config['nucleon_width'] = (_parser.get( 'initial_condition', 'nucleon_width'),
            '(For Trento) Gaussian nucleon width in fm, and the default value is 0.5')
    
    except configparser.Error:
        config['nucleon_width'] = (0.5,
            '(For Trento) Gaussian nucleon width in fm, and the default value is 0.5')
        

    try:
        config['constit_width'] = (_parser.get( 'initial_condition', 'constit_width'),
            '(For Trento) *** ONLY FOR TRENTO_2_0 *** Gaussian constituent width in fm, and the default value is set equal to the nucleon_width')
    
    except configparser.Error:
        config['constit_width'] = (0.5,
            '(For Trento) *** ONLY FOR TRENTO_2_0 *** Gaussian constituent width in fm, and the default value is set equal to the nucleon_width')
        

    try:
        config['constit_number'] = (_parser.get( 'initial_condition', 'constit_number'),
            '(For Trento) *** ONLY FOR TRENTO_2_0 *** Number of constituents inside the nucleon, and the default value is 1')
    
    except configparser.Error:
        config['constit_number'] = (1,
            '(For Trento) *** ONLY FOR TRENTO_2_0 *** Number of constituents inside the nucleon, and the default value is 1')
        

    try:
        config['nucleon_min_dist'] = (_parser.get( 'initial_condition', 'nucleon_min_dist'),
            '(For Trento) *** ONLY FOR TRENTO_2_0 *** Minimum nucleon-nucleon distance (fm) for Woods-Saxon nuclei (spherical and deformed), and the default value is 0')
    
    except configparser.Error:
        config['nucleon_min_dist'] = (0,
            '(For Trento) *** ONLY FOR TRENTO_2_0 *** Minimum nucleon-nucleon distance (fm) for Woods-Saxon nuclei (spherical and deformed), and the default value is 0')
        

    try:
        config['normalization'] = (_parser.get( 'initial_condition', 'normalization'),
            '(For Trento) Overall normalization factor, and the default value is 1')
    
    except configparser.Error:
        config['normalization'] = (1,
            '(For Trento) Overall normalization factor, and the default value is 1')
        


        

    try:
        config['Initial_profile'] = (_parser.get( 'initial_condition', 'Initial_profile'),
            'The path to initial energy density profile or the path to h5 file ')
    
    except configparser.Error:
        config['Initial_profile'] = ("Glauber",
            'The path to initial energy density profile or the path to h5 file ')


    try:
        config['Initial_nb_profile'] = (_parser.get( 'initial_condition', 'Initial_nb_profile'),
            'The path to initial baryon density profile')
    
    except configparser.Error:
        config['Initial_nb_profile'] = ("./playground/nb.dat",
            'The path to initial baryon density profile ')


    try:
        config['oneshot'] = (_parser.getboolean( 'initial_condition', 'oneshot'),
            'Create a smooth initital condition ')
    except configparser.Error:
        config['oneshot'] = (False,
            'Create a smooth initital condition ')
    

    try:
        config['SQRTS'] = (_parser.getfloat( 'initial_condition', 'SQRTS'),
            'Beam energy in units of GeV/n; like Au+Au 200 GeV; Pb+Pb 2760 GeV, SQRTS=2760')
    except configparser.Error:
        config['SQRTS'] = (np.float32(200),'Beam energy in units of GeV/n; like Au+Au 200 GeV; Pb+Pb 2760 GeV, SQRTS=2760')
        

    try:
        config['NucleusA'] = (_parser.get( 'initial_condition', 'NucleusA'),
            'The species projectile A [Au,Pb]')
    except configparser.Error:
        config['NucleusA'] = ("Au",
            'The species projectile A [Au,Pb]')

    try:
        config['NucleusB'] = (_parser.get( 'initial_condition', 'NucleusB'),
            'The species projectile B [Au,Pb] ')
    except configparser.Error:
        config['NucleusB'] = ("Au",
            'The species projectile B [Au,Pb] ')


    try:
        config['KFACTOR'] = (_parser.getfloat( 'initial_condition', 'KFACTOR'),
            'The normlization factor for initial energy profile')
    except configparser.Error:
        config['KFACTOR'] = (1.0,
            'The normlization factor for initial energy profile')
        
     
    try:
        config['cent'] = (_parser.get( 'initial_condition', 'cent'),
                'Collision centraility')
    except configparser.Error:
        config['cent'] = ("0_5",
                'Collision centraility')


    try:
        config['Hwn'] = (_parser.getfloat( 'initial_condition', 'Hwn'),
                'in range [0,1), energy density contribution from number of wounded nucleons')
    except configparser.Error:
        config['Hwn'] = (0.95,
                'in range [0,1), energy density contribution from number of wounded nucleons')


    try:
        config['Eta_flat'] = (_parser.getfloat( 'initial_condition', 'Eta_flat'),
                'The width of the plateau along etas at mid rapidity')
    except configparser.Error:
        config['Eta_flat'] = (1.3,
                'The width of the plateau along etas at mid rapidity')


    try:
        config['Eta_gw'] = (_parser.getfloat( 'initial_condition', 'Eta_gw'),
                'The gaussian weight along etas for energy deposition')
    except configparser.Error:
        config['Eta_gw'] = (1.5,
                'The gaussian weight along etas for energy deposition')

    try:
        config['Eta_flat_c'] = (_parser.getfloat( 'initial_condition', 'Eta_flat_c'),
                'The width of the plateau along etas at mid rapidity (nbinary)')
    except configparser.Error:
        config['Eta_flat_c'] = (1.5,
                'The width of the plateau along etas at mid rapidity (nbinary)')


    try:
        config['Eta_gw_c'] = (_parser.getfloat( 'initial_condition', 'Eta_gw_c'),
                'The gaussian weight along etas for energy deposition (nbinary)')
    except configparser.Error:
        config['Eta_gw_c'] = (1.3,
                'The gaussian weight along etas for energy deposition (nbinary)')
        
    try:
        config['r_gw'] = (_parser.getfloat( 'initial_condition', 'r_gw'),
                'The gaussian weight along r for energy deposition')
    except configparser.Error:
        config['r_gw'] = (0.5,
                'The gaussian weight along r for energy deposition')

        

    try:
        config['Eta0_nb'] = (_parser.getfloat( 'initial_condition', 'Eta0_nb'),
                'Eta0_nb in  envelope function of baryon number density')
    except configparser.Error:
        config['Eta0_nb'] = (1.5,
                'Eta0_nb in  envelope function of baryon number density')

    try:
        config['Eta_nb_gw_p'] = (_parser.getfloat( 'initial_condition', 'Eta_nb_gw_p'),
                'sigma0^+ in  envelope function of baryon number density')
    except configparser.Error:
        config['Eta_nb_gw_p'] = (0.2,
                'sigma0^+ in  envelope function of baryon number density')


    try:
        config['Eta_nb_gw_m'] = (_parser.getfloat( 'initial_condition', 'Eta_nb_gw_m'),
                'sigma0^- in  envelope function of baryon number density')
    except configparser.Error:
        config['Eta_nb_gw_m'] = (1.0,
                'sigma0^- in  envelope function of baryon number density')



    try:
        config['run_evolution'] = (_parser.getboolean('hydrodynamics', 'run_evolution'),
                'True to switch on hydro evolution')
    except configparser.Error:
        config['run_evolution'] = (True,
                'True to switch on hydro evolution')  

    try:
        config['TAU0'] = (_parser.getfloat('hydrodynamics', 'TAU0'),
                'time when hydro starts')
    except configparser.Error:
        config['TAU0'] = (1.0,
                'time when hydro starts')

            # Grid sizes, hyper surface grain
    try:
        config['NX'] = (_parser.getint( 'hydrodynamics', 'NX'),
                'Grid size along x direction')
    except configparser.Error:
        config['NX'] = (67,
                'Grid size along x direction')


    try:
        config['NY'] = (_parser.getint( 'hydrodynamics', 'NY'),
                'Grid size along y direction')
    except configparser.Error:
        config['NY'] = (67,
                'Grid size along y direction')


    try:
        config['NZ'] = (_parser.getint( 'hydrodynamics', 'NZ'),
                'Grid size along z direction')
    except configparser.Error:
        config['NZ'] = (67,
                'Grid size along z direction')

        
    try:
        config['DT'] = (_parser.getfloat( 'hydrodynamics', 'DT'),
                'time step for hydro evolution' )
    except configparser.Error:
        config['DT'] = (0.01,
                'time step for hydro evolution' )


    try:
        config['DX'] = (_parser.getfloat( 'hydrodynamics', 'DX'),
                'x step for hydro evolution' )
    except configparser.Error:
        config['DX'] = (0.3,
                'x step for hydro evolution' )


    try:
        config['DY'] = (_parser.getfloat( 'hydrodynamics', 'DY'),
                'y step for hydro evolution' )
    except configparser.Error:
        config['DY'] = (0.3,
                'y step for hydro evolution' )


    try:
        config['DZ'] = (_parser.getfloat( 'hydrodynamics', 'DZ'),
                'z step for hydro evolution' )
    except configparser.Error:
        config['DZ'] = (0.3,
                'z step for hydro evolution' )

        
        
    try:
        config['ntskip'] = ( _parser.getint( 'hydrodynamics', 'ntskip'), 
                'Skip time steps for bulk finding hypersurface '   )
    except configparser.Error:
        config['ntskip'] = ( 10, 
                'Skip time steps for bulk finding hypersurface '   )


    try:
        config['nxskip'] = ( _parser.getint( 'hydrodynamics', 'nxskip'), 
                'Skip steps along x for bulk finding hypersurface ')
    except configparser.Error:
        config['nxskip'] = ( 4, 
                'Skip steps along x for bulk finding hypersurface ')


    try:
        config['nyskip'] = ( _parser.getint( 'hydrodynamics', 'nyskip'), 
                'Skip steps along y for bulk finding hypersurface ')
    except configparser.Error:
        config['nyskip'] = ( 4, 
                'Skip steps along y for bulk finding hypersurface ')


    try:
        config['nzskip'] = ( _parser.getint( 'hydrodynamics', 'nzskip'), 
                'Skip steps along z for bulk finding hypersurface ')
    except configparser.Error:
        config['nzskip'] = ( 4, 
                'Skip steps along z for bulk finding hypersurface ')




    try:
        config['pimn_on']= (_parser.getboolean('hydrodynamics', 'pimn_on'), 
                'True to switch on the evolution for shear tensor pimn')
    except configparser.Error:
        config['pimn_on']= (True, 
                'True to switch on the evolution for shear tensor pimn')
        
    try:
        config['CETA_YMIN']= (_parser.getfloat('hydrodynamics', 'CETA_YMIN'), 
                'minimum eta/s(T)')
    except configparser.Error:
        config['CETA_YMIN']= (0.16, 
                'minimum eta/s(T)')
    try:
        config['CETA_XMIN']= (_parser.getfloat('hydrodynamics', 'CETA_XMIN'), 
                'temperature for minimum eta/s(T)')
    except configparser.Error:
        config['CETA_XMIN']= (0.16, 
                'temperature for minimum eta/s(T)')


    try:
        config['CETA_LEFT_SLOP']= (_parser.getfloat('hydrodynamics', 'CETA_LEFT_SLOP'), 
                'slop of eta/s(T) when T < CETA_XMIN')
    except configparser.Error:
        config['CETA_LEFT_SLOP']= (0.0, 
                'slop of eta/s(T) when T < CETA_XMIN')

    try:
        config['CETA_RIGHT_SLOP']= (_parser.getfloat('hydrodynamics', 'CETA_RIGHT_SLOP'), 
                'slop of eta/s(T) when T > CETA_XMIN')
    except configparser.Error:
        config['CETA_RIGHT_SLOP']= (0.0, 
                'slop of eta/s(T) when T > CETA_XMIN')


    try:
        config['bulkpr_on']= (_parser.getboolean('hydrodynamics', 'bulkpr_on'), 
                'True to switch on the evolution for bulk pressure PI')
    except configparser.Error:
        config['bulkpr_on']= (True, 
                'True to switch on the evolution for bulk pressure PI')
    try:
        config['CZETA_duke']= (_parser.getboolean('hydrodynamics', 'CZETA_duke'), 
                'the temperature dependence of bulk pressure transport coefficient.')
    except configparser.Error:
        config['CZETA_duke']= (True, 
                'the temperature dependence of bulk pressure transport coefficient.')

    try:
        config['CZETA']= (_parser.getfloat('hydrodynamics', 'CZETA'), 
                'the constant bulk pressure transport coefficient.')
    except configparser.Error:
        config['CZETA']= (0.0, 
                'the constant bulk pressure transport coefficient.')
    try:
        config['baryon_on']= (_parser.getboolean('hydrodynamics', 'baryon_on'), 
                'True to switch on the evolution for baryon current nb')
    except configparser.Error:
        config['baryon_on']= (True, 
                'True to switch on the evolution for baryon current nb')
    try:
        config['qu_on']= (_parser.getboolean('hydrodynamics', 'qu_on'), 
                'True to switch on the evolution for baryon diffusion current qmu')
    except configparser.Error:
        config['qu_on']= (True, 
                'True to switch on the evolution for baryon diffusion current qmu')

    try:
        config['CB_XMIN']= (_parser.getfloat('hydrodynamics', 'CB_XMIN'), 
                'temperature for minimum CB(T)')
    except configparser.Error:
        config['CB_XMIN']= (0.4, 
                'temperature for minimum CB(T)')


    try:
        config['source'] = (_parser.getboolean( 'hydrodynamics', 'source'),
                'True to switch on source terms (dynamical initial condition)')
    except configparser.Error:
        config['source'] = (False,
                'True to switch on source terms (dynamical initial condition)')
     
    try:
        config['save_hypersf'] = (_parser.getboolean( 'hydrodynamics', 'save_hypersf'),
                'True to save hypersf')  
    except configparser.Error:
        config['save_hypersf'] = (True,
                'True to save hypersf')  
        

        
    try:
        config['calc_vorticity'] = (_parser.getboolean( 'hydrodynamics', 'calc_vorticity'),
                'True to calculate vorticity')  
    except configparser.Error:
        config['calc_vorticity'] = (True,
                'True to calculate vorticity')  

    try:
        config['save_bulkinfo'] = (_parser.getboolean( 'hydrodynamics', 'save_bulkinfo'),
                'True to save  bulk information during evolution ')
    except configparser.Error:
        config['save_bulkinfo'] = (False,
                'True to save bulk information during evolution ')
    try:
        config['save_pimn'] = (_parser.getboolean( 'hydrodynamics', 'save_pimn'),
                'True to save  shear tensor during evolution ')
    except configparser.Error:
        config['save_pimn'] = (False,
                'True to save shear tensor during evolution ')

    try:
        config['save_qb'] = (_parser.getboolean( 'hydrodynamics', 'save_qb'),
                'True to save  baryon diffusion current during evolution ')
    except configparser.Error:
        config['save_qb'] = (False,
                'True to save baryon diffusion current during evolution ')

    try:
        config['save_bulkpr'] = (_parser.getboolean( 'hydrodynamics', 'save_bulkpr'),
                'True to save bulk pressure during evolution ')
    except configparser.Error:
        config['save_bulkpr'] = (False,
                'True to save bulk pressure during evolution ')
   


    try:
        config['eos_type']  = (_parser.get('EOS', 'eos_type'), 
                'choices = [ideal_gas, first_order, lattice_wb,\
                lattice_pce150, lattice_pce165, pure_gauge, HotQCD2014, neosB,neosBQS,chiral,eosq,njl_model]')
    except configparser.Error:
        config['eos_type']  = ("lattice_pce165", 
                'choices = [ideal_gas, first_order, lattice_wb,\
                lattice_pce150, lattice_pce165, pure_gauge, HotQCD2014, neosB,neosBQS,chiral,eosq,njl_model]')


        
    try:
        config['switch_eos']  = (_parser.get('EOS', 'switch_eos'), 
                'switch to different eos at hypersurface, choices = [ideal_gas, first_order, lattice_wb,\
                 lattice_pce150, lattice_pce165, pure_gauge, HotQCD2014, neosB,neosBQS,chiral,eosq,njl_model,HARDON_GAS]')
    except configparser.Error:
        config['switch_eos']  = (config['eos_type'], 
                'switch to different eos at hypersurface, choices = [ideal_gas, first_order, lattice_wb,\
                 lattice_pce150, lattice_pce165, pure_gauge, HotQCD2014, neosB,neosBQS,chiral,eosq,njl_model,HARDON_GAS]')
    
    try:
        config['run_cooperfrye'] = (_parser.getboolean( 'freezeeout_and_afterburner', 'run_cooperfrye'),
                'True to compute cooperfrye samping or integration')
    except configparser.Error:
        config['run_cooperfrye'] = (True,
                'True to compute cooperfrye samping or integration')


    try:
        config['run_afterbuner'] = (_parser.getboolean( 'freezeeout_and_afterburner', 'run_afterbuner'),
                'True to switch on afterbuner')
    except configparser.Error:
        config['run_afterbuner'] = (True,
                'True to switch on afterbuner')


        
    try:
        config['cornelius'] = (_parser.getboolean( 'freezeeout_and_afterburner', 'cornelius'),
                'True to switch on cornelius method to get hypersuface element')
    except configparser.Error:
        config['cornelius'] = (True,
                'True to switch on cornelius method to get hypersuface element')

        
    try:
        config['corona'] = (_parser.getboolean( 'freezeeout_and_afterburner', 'corona'),
                'True to compute equal tau freezeout at initial proper time')
    except configparser.Error:
        config['corona'] = (True,
                'True to compute equal tau freezeout at initial proper time')


    try:
        config['Tfrz_on'] = (_parser.getboolean( 'freezeeout_and_afterburner', 'Tfrz_on'),
                'True to switch on the constant temperature freezeout ')
    except configparser.Error:
        config['Tfrz_on'] = (True,
                'True to switch on the constant temperature freezeout ')

    

    try:
        config['TFRZ'] = (_parser.getfloat('freezeeout_and_afterburner', 'TFRZ'), 
                'Freeze out temperature, default=0.137')
    except configparser.Error:
        config['TFRZ'] = (0.137, 
                'Freeze out temperature, default=0.137')


    try:
        config['Edfrz'] = (_parser.getfloat('freezeeout_and_afterburner', 'Edfrz'), 
                'Freeze out energy density, default=0.4 Gev/fm^3')
    except configparser.Error:
        config['Edfrz'] = (0.4,  
                'Freeze out energy density, default=0.4 Gev/fm^3')

        
    try:
        config['sample_type']  = (_parser.get('freezeeout_and_afterburner', 'sample_type'), 
                'Sample method, choices = [mc,mcgpu,smooth] , hint: mcgpu only for finite baryon density ')
    except configparser.Error:
        config['sample_type']  = ("mc", 
                'Sample method, choices = [mc,mcgpu,smooth] , hint: mcgpu only for finite baryon density ')

        
    try:
        config['reso_decay'] = (_parser.getboolean( 'freezeeout_and_afterburner', 'reso_decay'),
                'True to switch on resonance decay, if you want to run afterburner, please set it False')
    except configparser.Error:
        config['reso_decay'] = (False,
                'True to switch on resonance decay, if you want to run afterburner, please set it False')

        
    try:
        config['nsample']  = (_parser.get('freezeeout_and_afterburner', 'nsample'), 
                'The number of sampling')
    except configparser.Error:
        config['nsample']  = ("2000", 
                'The number of sampling')

        
    try:
        config['afterburner']  = (_parser.get('freezeeout_and_afterburner', 'afterburner'), 
                'Afterburner type: SMASH and URQMD')    
    except configparser.Error:
        config['afterburner']  = ("None", 
                'Afterburner type: SMASH and URQMD')    
    
    try:
        config['Colhydro']  = (_parser.getboolean('coLBT_hydro', 'Colhydro'), 
                'switch on CoLBT-hydro')    
    except configparser.Error:
        config['Colhydro']  = (False, 
                'switch on CoLBT-hydro')    

    try:
        config['trans_model']  = (_parser.get('coLBT_hydro', 'trans_model'), 
                'Transport model for CoLBT-hydro, defaut: LBT')    
    except configparser.Error:
        config['trans_model']  = ('LBT', 
                'Transport model for CoLBT-hydro, defaut: LBT')
    
    try:
        config['LBT_config']  = (_parser.get('coLBT_hydro', 'LBT_config'), 
                'The path to LBT configuration file')    
    except configparser.Error:
        config['LBT_config']  = (os.path.abspath(os.path.join(cwd,'./LBT/conf_LBT')) , 
                'The path to LBT configuration file')     

    try:
        config['LBT_table']  = (_parser.get('coLBT_hydro', 'LBT_table'), 
                'The path to LBT table file')    
    
    except configparser.Error:
        config['LBT_table']  = (os.path.abspath(os.path.join(cwd,'./LBT/readindatafile')), 
                'The path to LBT table file')    
    try:
        config['parton_dat']  = (_parser.get('coLBT_hydro', 'parton_dat'), 
                'The path to the file including momentum information of initial parton')    
    except configparser.Error:
        config['parton_dat']  = ('./playground/pythia_parton', 
                'The path to the file including momentum information of initial parton')     


    try:
        config['jet_r_gw']  = (_parser.getfloat('coLBT_hydro', 'jet_r_gw'), 
                'The gaussion weight along r in source deposition')    
    except configparser.Error:
        config['jet_r_gw']  = (0.2, 
                'The gaussion weight along r in source deposition')  

    try:
        config['jet_eta_gw']  = (_parser.getfloat('coLBT_hydro', 'jet_eta_gw'), 
                'The gaussion weight along etas in source deposition')    
    except configparser.Error:
        config['jet_eta_gw']  = (0.2, 
                'The gaussion weight along etas in source deposition')      
    
    try:
        config['Ecut']  = (_parser.getfloat('coLBT_hydro', 'Ecut'), 
                'energy cut for themal parton to medium ')    
    except configparser.Error:
        config['Ecut']  = (3.0, 
                'energy cut for themal parton to medium ')  
    
    try:
        config['alphas']  = (_parser.getfloat('coLBT_hydro', 'alphas'), 
                'strong coupling coefficient')    
    except configparser.Error:
        config['alphas']  = (3.0, 
                'strong coupling coefficient')  


    try:
        config['pthatmin']  = (_parser.getint('coLBT_hydro', 'pthatmin'), 
                'The minimum invariant pT')    
    except configparser.Error:
        config['pthatmin']  = (20.0, 
                'The minimum invariant pT')  


    try:
        config['pthatmax']  = (_parser.getint('coLBT_hydro', 'pthatmax'), 
                'The maximum invariant pT')    
    except configparser.Error:
        config['pthatmax']  = (30.0, 
                'The maximum invariant pT')  



    try:
        config['LAM1']= (_parser.getfloat('Test', 'LAM1'), 
                'coefficient for pimn^2 term')
    except configparser.Error:
        config['LAM1']= (-10.0, 
                'coefficient for pimn^2 term')


    try:
        config['BSZ'] = (_parser.getint('opencl', 'local_workgroup_size'), 
                'Local workgroup size in one dimension')
    except configparser.Error:
        config['BSZ'] = (64, 
                'Local workgroup size in one dimension')
   

    parser = argparse.ArgumentParser(description=\
        'Input parameters for hydrodynamic simulations')
    
    for key, value in list(config.items()):
        parser.add_argument('--{key}'.format(key=key), nargs='?', const=1, 
                type=type(value[0]), default=value[0], help=value[1] )

    parser.add_argument('--riemann_test', nargs='?', const=1, type=bool, 
            default=False, help='true to switch on riemann test for expansion to vacuum problem')
    parser.add_argument('--DNB', nargs='?', const=0.1, type=np.float32, 
            default=False, help='true to switch on riemann test for expansion to vacuum problem')
    parser.add_argument('--NNB', nargs='?', const=140, type=np.float32, 
            default=False, help='true to switch on riemann test for expansion to vacuum problem')

    parser.add_argument('--gubser_visc_test', nargs='?', const=1, type=bool, 
            default=False, help='true to switch to 2nd order gubser visc test')

    parser.add_argument('--pimn_omega_coupling', nargs='?', const=1, type=bool, 
            default=False, help='true to switch on pi^{mu nu} and vorticity coupling term')

    parser.add_argument('--omega_omega_coupling', nargs='?', const=1, type=bool, 
            default=False, help='true to switch on vorticity and vorticity coupling term')

    parser.add_argument('--use_float32', nargs='?', const=1, type=bool, 
            default=True, help='true for float and false for double precision')

    parser.add_argument('--opencl_interactive', nargs='?', const=1, type=bool, 
            default=False, help='true to choose device type and device id at run time')

    
    args, unknown = parser.parse_known_args()


    args.sz_int = np.dtype('int32').itemsize   #==sizeof(int) in c
    if args.use_float32 == True :
        args.real = np.float32
        args.real2 = array.vec.float2
        args.real4 = array.vec.float4
        args.real8 = array.vec.float8
        args.sz_real = np.dtype('float32').itemsize   #==sizeof(float) in c
        args.sz_real2 = array.vec.float2.itemsize
        args.sz_real4 = array.vec.float4.itemsize
        args.sz_real8 = array.vec.float8.itemsize
    else :
        args.real = np.float64
        args.real2 = array.vec.double2
        args.real4 = array.vec.double4
        args.real8 = array.vec.double8
        args.sz_real = np.dtype('float64').itemsize   #==sizeof(double) in c
        args.sz_real2= array.vec.double2.itemsize
        args.sz_real4= array.vec.double4.itemsize
        args.sz_real8= array.vec.double8.itemsize

    return args

if __name__ == '__main__':


    if len(sys.argv)!= 2:
        print (" usage: python config.py hydro.info")
        
        
    
    path = sys.argv[1]
    cfg1 = read_config(path)    
#cfg = read_config("./hydro.info")
