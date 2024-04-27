#/usr/bin/env python

#Original Copyright (c) 2014-  Long-Gang Pang <lgpang@qq.com>
#Copyright (c) 2018- Xiang-Yu Wu <xiangyuwu@mails.ccnu.edu.cn>
#Copyright (c) 2024- Jun-Qi Tao <taojunqi@mails.ccnu.edu.cn>


from subprocess import call
import pandas as pd
import os
import numpy as np
import reader
from rotate import rotate

__cwd__, __cwf__ = os.path.split(__file__)

class Collision(object):
    def __init__(self, config):
        self.config = config
        centrality_file = os.path.join(__cwd__, config['centrality_file'])
        self.info = pd.read_csv(centrality_file)

    def get_smin_smax(self, cent='0_6'):
        '''get min/max initial total entropy for one
        centrality class, stored in auau200.csv or ...'''
        clow, chigh = cent.split('_')
        smin = self.entropy_bound(cent_bound = float(chigh))
        smax = self.entropy_bound(cent_bound = float(clow))
        return smin, smax

    def entropy_bound(self, cent_bound=5):
        '''get entropy value for one specific centrality bound'''
        self.info.set_index(['cent'])
        cents = self.info['cent']
        entropy = self.info['entropy']
        return np.interp(cent_bound, cents, entropy)

    def create_ini_1_3(self, cent, output_path, 
                       grid_max=15.0, grid_step=0.1, num_of_events=1, 
                       one_shot_ini=False, align_for_oneshot=False, 
                       reduced_thickness=0, fluctuation=1, nucleon_width=0.5, normalization=1):
        
        smin, smax = self.get_smin_smax(cent)

        call(['./trento', self.config['projectile'],
              self.config['target'],
              '%s'%num_of_events,
              '-o', output_path,
              '-x', '%s'%self.config['cross_section'],
              '--s-min', '%s'%smin,
              '--s-max', '%s'%smax,
              '--grid-max', '%s'%grid_max,
              '--grid-step', '%s'%grid_step,
              '-p', '%s'%reduced_thickness,
              '-k', '%s'%fluctuation,
              '-w', '%s'%nucleon_width,
              '-n', '%s'%normalization])
        
        if one_shot_ini:
            ngrid = int(2 * grid_max / grid_step)
            sxy = np.zeros((ngrid, ngrid), dtype=np.float32)
            events = os.listdir(output_path)
            print(events)
            num_of_events = 0
            for event in events:
                try:
                    fname = os.path.join(output_path, event)
                    dat = np.loadtxt(fname).reshape(ngrid, ngrid)
                    opt = reader.get_comments(fname)
                    sd_new = rotate(dat, opt['ixcm'], opt['iycm'], opt['phi_2'], ngrid, ngrid)
                    sxy += sd_new 
                    num_of_events += 1
                except:
                    print(fname, 'is not a trento event')
            np.savetxt(os.path.join(output_path, "one_shot_ini.dat"), sxy/num_of_events, header=cent)

    def create_ini_2_0(self, cent, output_path,
                       grid_max=15.0, grid_step=0.1, num_of_events=1,
                       one_shot_ini=False, align_for_oneshot=False,
                       reduced_thickness=0, fluctuation=1, nucleon_width=0.5, constit_width=0.5, constit_number=1, nucleon_min_dist=0, normalization=1):
        
        smin, smax = self.get_smin_smax(cent)

        call(['./trento', self.config['projectile'],
              self.config['target'],
              '%s'%num_of_events,
              '-o', output_path,
              '-x', '%s'%self.config['cross_section'],
              '--s-min', '%s'%smin,
              '--s-max', '%s'%smax,
              '--grid-max', '%s'%grid_max,
              '--grid-step', '%s'%grid_step,
              '-p', '%s'%reduced_thickness,
              '-k', '%s'%fluctuation,
              '-w', '%s'%nucleon_width,
              '-v', '%s'%constit_width,
              '-m', '%s'%constit_number,
              '-d', '%s'%nucleon_min_dist,
              '-n', '%s'%normalization])
        
        if one_shot_ini:
            ngrid = int(2 * grid_max / grid_step)
            sxy = np.zeros((ngrid, ngrid), dtype=np.float32)
            events = os.listdir(output_path)
            print(events)
            num_of_events = 0
            for event in events:
                try:
                    fname = os.path.join(output_path, event)
                    dat = np.loadtxt(fname).reshape(ngrid, ngrid)
                    opt = reader.get_comments(fname)
                    sd_new = rotate(dat, opt['ixcm'], opt['iycm'], opt['phi_2'], ngrid, ngrid)
                    sxy += sd_new 
                    num_of_events += 1
                except:
                    print(fname, 'is not a trento event')
            np.savetxt(os.path.join(output_path, "one_shot_ini.dat"), sxy/num_of_events, header=cent)


class AuAu200(Collision):
    def __init__(self):
        config = {'projectile':'Au',
                  'target':'Au',
                  'cross_section':4.23,
                  'centrality_file':'auau200_cent.csv'}
        super(AuAu200, self).__init__(config)

class PbPb2760(Collision):
    def __init__(self):
        config = {'projectile':'Pb',
                  'target':'Pb',
                  'cross_section':6.4,
                  'centrality_file':'pbpb2760_cent.csv'}
        super(PbPb2760, self).__init__(config)
       
class PbPb5020(Collision):
    def __init__(self):
        config = {'projectile':'Pb',
                  'target':'Pb',
                  'cross_section':7.0,
                  'centrality_file':'pbpb5020_cent.csv'}
        super(PbPb5020, self).__init__(config)

class XeXe5440(Collision):
    def __init__(self):
        config = {'projectile':'Xe',
                  'target':'Xe',
                  'cross_section':7.1,
                  'centrality_file':'xexe5440_cent.csv'}
        super(XeXe5440, self).__init__(config)

class RuRu200(Collision):
    def __init__(self):
        config = {'projectile':'Ru',
                  'target':'Ru',
                  'cross_section':4.23,
                  'centrality_file':'ruru200_cent.csv'}
        super(RuRu200, self).__init__(config)
class RuRu200(Collision):
    def __init__(self):
        config = {'projectile':'Ru',
                  'target':'Ru',
                  'cross_section':4.23,
                  'centrality_file':'ruru200_cent.csv'}
        super(RuRu200, self).__init__(config)
class Ru2Ru2200(Collision):
    def __init__(self):
        config = {'projectile':'Ru2',
                  'target':'Ru2',
                  'cross_section':4.23,
                  'centrality_file':'ru2ru2200_cent.csv'}
        super(Ru2Ru2200, self).__init__(config)
class Ru3Ru3200(Collision):
    def __init__(self):
        config = {'projectile':'Ru3',
                  'target':'Ru3',
                  'cross_section':4.23,
                  'centrality_file':'ru3ru3200_cent.csv'}
        super(Ru3Ru3200, self).__init__(config)
class Ru4Ru4200(Collision):
    def __init__(self):
        config = {'projectile':'Ru4',
                  'target':'Ru4',
                  'cross_section':4.23,
                  'centrality_file':'ru4ru4200_cent.csv'}
        super(Ru4Ru4200, self).__init__(config)

class ZrZr200(Collision):
    def __init__(self):
        config = {'projectile':'Zr',
                  'target':'Zr',
                  'cross_section':4.23,
                  'centrality_file':'zrzr200_cent.csv'}
        super(ZrZr200, self).__init__(config)

class AuAu160(Collision):
    def __init__(self):
        config = {'projectile':'Au',
                  'target':'Au',
                  'cross_section':4.08,
                  'centrality_file':'auau160_cent.csv'}
        super(AuAu160, self).__init__(config)

class AuAu95(Collision):
    def __init__(self):
        config = {'projectile':'Au',
                  'target':'Au',
                  'cross_section':3.80,
                  'centrality_file':'auau95_cent.csv'}
        super(AuAu95, self).__init__(config)


if __name__=='__main__':
    xexe = XeXe5440()
    xexe.create_ini('0_6', './dat', num_of_events=100, one_shot_ini=True)
