#/usr/bin/env python
#author: lgpang
#email: lgpang@qq.com
#createTime: Sat 25 Oct 2014 04:40:15 PM CST

import pyopencl as cl
from pyopencl import array
import os, sys
from time import time
import numpy as np
from scipy.special import hyp2f1
import unittest
import matplotlib.pyplot as plt

cwd, cwf = os.path.split(__file__)
sys.path.append(os.path.join(cwd, '..'))

from ideal import CLIdeal
from config import cfg



class TestGubser(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        #self.cfg.IEOS = 0
        self.cfg.eos_type = 'ideal_gas'
        self.ideal = CLIdeal(self.cfg)
        self.ctx = self.ideal.ctx
        self.queue = self.ideal.queue
        self.compile_options = self.ideal.compile_options
        from mcglauber import MCglauber
        ini = MCglauber(self.cfg, self.queue,self.compile_options,self.ideal.d_ev[1],self.ideal.d_nb[1])
        ini.eye_event() 

if __name__ == '__main__':
    unittest.main()
