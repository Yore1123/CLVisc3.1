'''Lattice QCD EOS for 2+1flavor from Wuppertal-Budapest Group 2014'''
#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com

import numpy as np
from scipy.integrate import quad
import math
from scipy import interpolate
import os

hbarc = 0.19732

class GlueBall(object):
    '''EOS from WuppertalBudapest Group 2014.
    GlueBall EOS, with first order phase transition'''
    def __init__(self, f_eostable):
        glueball_dat = np.loadtxt(f_eostable)

        # T in units [GeV], others are dimensionless
        T = 1.0E-3*glueball_dat[:,4]
        e_o_T4 = glueball_dat[:,2]
        p_o_T4 = glueball_dat[:,3]
        s_o_T3 = glueball_dat[:,1]

        hbarc3 = hbarc**3
        self.energy_density = e_o_T4 * T**4 / hbarc3       # in units GeV/fm^3
        self.pressure = p_o_T4 * T**4 / hbarc3            # in units GeV/fm^3
        self.entropy_density = s_o_T3 * T**3 / hbarc3     # in units fm^{-3}
        self.T = T

    def create_table(self):
        fT_ed = interpolate.interp1d(self.energy_density, self.T)
        ed_new = np.linspace(0.0, 2000, 200000, endpoint=False)
        T_new = fT_ed(ed_new)
        fP_ed = interpolate.interp1d(self.energy_density, self.pressure)
        pre_new = fP_ed(ed_new)

        return ed_new, pre_new, T_new



glueball_cwd, glueball_cwf = os.path.split(__file__)

glueball_datafile = os.path.join(glueball_cwd, 'glueball_v2.dat')

eos = GlueBall(glueball_datafile)

ed, pr, T = eos.create_table()

ed_start = 0.0
ed_step = 0.01
num_ed = 200000



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    #pr = np.diff(pr[:])
    plt.plot(ed[:200], pr[:200], 'ro')
    #plt.xlim(0.5, 1.0)

    T_test = T[1001]
    ed_test = ed[1001]
    print(ed[1000])
    
    plt.show()


