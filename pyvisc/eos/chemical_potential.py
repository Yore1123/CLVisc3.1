#/usr/bin/env python
#Original Copyright (c) 2014  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>


from __future__ import print_function

import numpy as np
from math import floor
import logging
import os

#logging.basicConfig(level=logging.DEBUG)

'''This module is used to calculate the effective chemical potential
   for all the resonances at freeze out temperature, from the eos table
   s95p-PCE-v1. This EOS assumes chemical freeze out happens at T=150 MeV,
   from chemical freeze out to kinetic freeze out, the number of stable
   particles is unchanged if all the resonance decay.
   For this purpose, the temperature is changed and effective chemical
   potential is introduced to fix the particle ratio. '''


class ChemicalPotential(object):
    def __init__(self, efrz, eos_type='lattice_pce150'):
        '''generate chemical potential file for freeze out usage'''
        self.efrz = efrz
        self.eos_type = eos_type

        cwd, cwf = os.path.split(__file__)

        self.path = os.path.join(cwd, 'eos_table/s95p-PCE165-v0/')

        if eos_type == 'lattice_pce150':
            self.path = os.path.join(cwd, 'eos_table/s95p-PCE-v1/')

        mu_for_stable = self.get_chemical_potential_for_stable(efrz)

        pids = self.get_pid_for_stable()

        logging.debug("len of chem = %s"%len(mu_for_stable))
        logging.debug("len of pids = %s"%len(pids))

    def get_pid_for_stable(self):
        """Get the stable particles pid from particles.dat, Gamma is exculded,
        return the array of stable particles pid"""
        fname = os.path.join(self.path,  'particles.dat')
        particles = open(fname, "r").readlines()
        stables = []
        for particle in particles:
            info = particle.split()
            if info[-1] == "Stable" or info[-1] == '********':
                if info[1] != "Gamma":
                    stables.append(info[1])
        return stables


    def get_chemical_potential_for_stable(self, efrz):
        ''' interpolate to get the chemical potential for stable particles
            at freeze out energy density '''
        fname = os.path.join(self.path, "s95p-PCE165-v0_pichem1.dat")
        if self.eos_type == 'lattice_pce150':
            fname = os.path.join(self.path, "s95p-PCE-v1_pichem1.dat")
    
        mu_for_stable = None
        with open(fname, 'r') as fchemical:
            e0 = float(fchemical.readline())
            de,ne = fchemical.readline().split()
            de,ne = float(de), int(ne)
            nstable = int(fchemical.readline())

            logging.debug('e0,de,ne,nstable=%s, %s, %s, %s'%(e0, de, ne, nstable))

            chemical_potential = np.loadtxt(fname, skiprows=3)[::-1]
            energy_density = np.array([e0 + i * de for i in range(ne)])

            idx = int(floor((efrz - e0) / de))

            # when energy density is too big, return 0.0 chemical potential

            if idx > 500:
                zero_mu_for_stable = np.zeros_like(chemical_potential[0])
                return zero_mu_for_stable

            ed0, ed1 = energy_density[idx], energy_density[idx+1]
            mu0, mu1 = chemical_potential[idx], chemical_potential[idx+1]
            # linear interpolation
            w0 = (efrz - ed0) / de
            w1 = (ed1 - efrz) / de
            mu_for_stable = w0 * mu1 + w1 * mu0
        return mu_for_stable

    def get_chemical_potential_for_resonance(self, save_path):
        """Calc Resonances chemical potential from stable particles 
        chemical potential and the decay chain listed in pdg05.dat """
        pid_for_stable = self.get_pid_for_stable()
        mu_for_stable  = self.get_chemical_potential_for_stable(self.efrz)
        mu_for_all = {}

        set_to_zero = True
        if self.eos_type == 'lattice_pce165' or self.eos_type == 'lattice_pce150':
            set_to_zero = False

        for i in range(len(pid_for_stable)):
            mu_for_all[pid_for_stable[i]] = mu_for_stable[i]
        mu_for_all['22'] = 0.0

        pids = []

        fname = os.path.join(self.path, 'pdg05.dat')
        with open(fname, "r") as f_pdg:
            lines = f_pdg.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                particle = line.split()
                pid = particle[0]
                ndecays = int(particle[-1])
                mu_reso = 0.0
                for j in range(ndecays):
                    decay = lines[i+j+1].split()
                    ndaughter = int(decay[1])
                    branch_ratio = float(decay[2])

                    # mu_i = sum_j mu_j * n_ij
                    for k in range(ndaughter):
                        daughter_pid = decay[3 + k]
                        mu_reso += branch_ratio * mu_for_all[daughter_pid]

                mu_for_all[pid] = mu_reso
                i = i + ndecays + 1
                pids.append(pid)
    
        save_fname = os.path.join(save_path, "chemical_potential.dat")
        with open(save_fname, "w") as fout:
            for pid in pids:
                if not set_to_zero:
                    print(pid, mu_for_all[pid], file=fout)
                else:
                    print(pid, 0.0, file=fout)
    
 

def create_table(efrz = 0.27, output_path='.', eos_type='lattice_pce150'):
    '''save the chemical potential for different EOS
    eos_type='lattice_pce165', and 'lattice_pce150', save effective mu
    eos_type='ideal_gas', 'first_order', 'pure_gauge', 'lattice_wb'
    save 0.0 for all the resonance'''
    import sys
    from eos import Eos
    from subprocess import call
    
    #if eos_type == "ideal_gas_baryon":
    #    eos_type = 'ideal_gas'
    eos_list = ['IDEAL_GAS','LATTICE_PCE165','LATTICE_PCE150','LATTICE_WB',\
        'PURE_GAUGE','FIRST_ORDER','HOTQCD2014']
    if eos_type.upper() in eos_list:
        eos = Eos(eos_type)
        chem = ChemicalPotential(efrz, eos_type=eos_type)
        chem.get_chemical_potential_for_resonance(output_path)
        save_fname = os.path.join(output_path, "chemical_potential.dat")



if __name__ == '__main__':
    #chem = ChemicalPotential(0.2, eos_type='lattice_pce150')
    #chem = ChemicalPotential(0.2, eos_type='ideal_gas_baryon')
    #chem.get_chemical_potential_for_resonance('./')
    create_table(0.2,eos_type='ideal_gas_baryon')
