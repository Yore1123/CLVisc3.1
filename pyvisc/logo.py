#/usr/bin/env python
import numpy as np





def print_logo():
    
    logo1 = '''
       ######  ##       ##     ## ####  ######   ######  
      ##    ## ##       ##     ##  ##  ##    ## ##    ## 
      ##       ##       ##     ##  ##  ##       ##       
      ##       ##       ##     ##  ##   ######  ##       
      ##       ##        ##   ##   ##        ## ##       
      ##    ## ##         ## ##    ##  ##    ## ##    ## 
       ######  ########    ###    ####  ######   ######  
    '''

    Declaration= '''
When using (3+1)-D CLVisc hydrodynamics model, please cite:
    
    (a) L.-G. Pang, H. Petersen, and X.-N. Wang, Phys.Rev.C 97 (2018) 6, 064918 , arXiv:1802.04449
    (b) X.-Y. Wu, G.-Y. Qin, L.-G. Pang, and X.-N. Wang,Phys.Rev.C 105 (2022) 3, 034909, arXiv:2107.04949
    (c) J.-Q. Tao, X. Fan, (In writing)
    '''

    print (logo1)
    print (Declaration)

if __name__=="__main__":
    print_logo()
