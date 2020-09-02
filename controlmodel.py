# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:32:17 2017

@author: lindsey
"""

'''
Extended model reparameterized to have a single parameter controlling the degradation of nuclear CRY
'''


import numpy  as np
import casadi as cs
import matplotlib.pyplot as plt


EqCount = 14
ParamCount = 46
               
y0in = np.array([2.94399774, 1.45982459, 1.37963843, 1.53271061, 0.64180771,
       2.51814106, 5.27221041, 5.14682548, 1.29418516, 0.89956422,
       0.5354595 , 0.36875376, 0.09342431, 0.05479389])
                 
period = 27.072

vtp    = cs.SX.sym("vtp")
ktp = cs.SX.sym("ktp")
kb = cs.SX.sym("kb")
vdp = cs.SX.sym("vdp")
kdp = cs.SX.sym("kdp")    

vtc1   = cs.SX.sym("vtc1")
vtc2   = cs.SX.sym("vtc2")
ktc = cs.SX.sym("ktc")
vdc1 = cs.SX.sym("vdc1")
vdc2 = cs.SX.sym("vdc2")    
kdc = cs.SX.sym("kdc")

vtror = cs.SX.sym("vtror")
ktror = cs.SX.sym("ktror")
vdror = cs.SX.sym("vdror")
kdror = cs.SX.sym("kdror")

vtrev = cs.SX.sym("vtrev")
ktrev = cs.SX.sym("ktrev")
vdrev = cs.SX.sym("vdrev")
kdrev = cs.SX.sym("kdrev")

klp = cs.SX.sym("klp")
vdP = cs.SX.sym("vdP")
kdP = cs.SX.sym("kdP")
vaCP = cs.SX.sym("vaCP")
vdCP = cs.SX.sym("vdCP")

klc = cs.SX.sym("klc") #do we need this or are we assuming removal here for simplicity
vdC1 = cs.SX.sym("vdC1")
kdC = cs.SX.sym("kdC")    
vdC2 = cs.SX.sym("vdC2")

klror = cs.SX.sym("klror")
vdROR = cs.SX.sym("vdROR")
kdROR = cs.SX.sym("kdROR") 

klrev = cs.SX.sym("klrev")
vdREV = cs.SX.sym("vdREV")
kdREV = cs.SX.sym("kdREV") 

vxROR = cs.SX.sym("vxROR")
vxREV = cs.SX.sym("vxREV")
kxREV = cs.SX.sym("kxREV")
kxROR = cs.SX.sym("kxROR")
vdb = cs.SX.sym("vdb")
kdb = cs.SX.sym("kdb")

klb = cs.SX.sym("klb")
vdB = cs.SX.sym("vdB")
kdB = cs.SX.sym("kdB")

vdC1N = cs.SX.sym("vdC1N")
kdCP = cs.SX.sym("kdCP")
vdC2N = cs.SX.sym("vdC2N")
mc = cs.SX.sym("mc")     

paramset = cs.vertcat([vtp, ktp, kb, vdp, kdp, #0-4
                       vtc1, vtc2, ktc, vdc1, vdc2, kdc, #5-10
                       vtror, ktror, vdror, kdror, #11-14
                       vtrev, ktrev, vdrev, kdrev, #15-18
                       klp, vdP, kdP, vaCP, vdCP, #19-23
                       vdC1, kdC, vdC2, #klc removed from front, #24-26
                       klror, vdROR, kdROR, #27-29
                       klrev, vdREV, kdREV, #30-32
                       vxROR, vxREV, kxREV, kxROR, vdb, kdb, #33-38
                       klb, vdB, kdB, #39-41
                       vdC1N, kdCP, mc]) #42-44

periodsenses = [ 1.21267171e+02, -6.46459190e+01, -1.83245333e+02, -6.62624839e+01, -3.78688277e+01,  
                7.11855423e+00, -1.54818625e+02,  3.32332161e+01, -2.46310305e+00,  1.80521026e+01, -2.92278674e+00, 
                -2.94376894e+00, 1.02294862e+00,  2.28938306e-01, -5.69834047e-02,  
                5.25018395e-02, -5.21233514e-01, -2.32208644e-03,  1.37817667e-02, 
                -2.04853396e+00, 3.38928057e-01, -5.19770932e+02, -6.19378292e+02, -6.02107260e+01,
                -1.74596697e-01, -1.36586419e+00,  4.35957618e+00, 
                -5.34836422e-01, 9.15423522e-02, -3.98859328e-02,  
                1.46158723e-01, -2.64549929e-02, 1.21034296e-02,  
                2.07635240e-01,  8.81979306e-02, -2.03725809e-01, -1.40704885e-01, -3.12079507e-01,  5.44207769e-01,  
                6.81729775e-01, -1.73452556e-01,  1.04764505e-01, 
                -8.48117819e+01, -1.20934658e+01, -1.66509432e-01]

       
param = [  .26726,   .33468,   .00117, .55011,   .00146,   
          .08212, .07313,   .28353,   .58013, .52993,   1.98812,   
          .08178, .23693,   .51117,   1.20013,
         1.15003,   .09945,   25.99980, 2.69984,   
         2.00001,   4.65000, .00031,   .00882,   .05999,
         1.59984,   2.08639,   1.51491,
         .35918,   1.29973,   1.95961,
         .25461,   1.29951,   1.96010,
         1.81951,   2.01031,   1.72406, 1.07518,   1.93448,   .55369,
         .37069,   1.86141,   2.71612,
         .07588,   .05499,   3.03242] #last parameter is mc      


def model():

    #==================================================================
    #setup of symbolics
    #==================================================================
    p   = cs.SX.sym("p")
    c1  = cs.SX.sym("c1")
    c2  = cs.SX.sym("c2")
    P   = cs.SX.sym("P")
    C1  = cs.SX.sym("C1")
    C2  = cs.SX.sym("C2")
    C1N = cs.SX.sym("C1N")
    C2N = cs.SX.sym("C2N")
    rev = cs.SX.sym("rev")
    ror = cs.SX.sym("ror")
    REV = cs.SX.sym("REV")
    ROR = cs.SX.sym("ROR")
    b = cs.SX.sym("b")
    B = cs.SX.sym("B")
    
    y = cs.vertcat([p, c1, c2, ror, rev, P, C1, C2, ROR, REV, b, B, C1N, C2N])
    
    # Time Variable
    t = cs.SX.sym("t")
    
    
    #===================================================================
    #Parameter definitions
    #===================================================================
    vtp    = cs.SX.sym("vtp")
    ktp = cs.SX.sym("ktp")
    kb = cs.SX.sym("kb")
    vdp = cs.SX.sym("vdp")
    kdp = cs.SX.sym("kdp")    
    
    vtc1   = cs.SX.sym("vtc1")
    vtc2   = cs.SX.sym("vtc2")
    ktc = cs.SX.sym("ktc")
    vdc1 = cs.SX.sym("vdc1")
    vdc2 = cs.SX.sym("vdc2")    
    kdc = cs.SX.sym("kdc")
    
    vtror = cs.SX.sym("vtror")
    ktror = cs.SX.sym("ktror")
    vdror = cs.SX.sym("vdror")
    kdror = cs.SX.sym("kdror")
    
    vtrev = cs.SX.sym("vtrev")
    ktrev = cs.SX.sym("ktrev")
    vdrev = cs.SX.sym("vdrev")
    kdrev = cs.SX.sym("kdrev")

    klp = cs.SX.sym("klp")
    vdP = cs.SX.sym("vdP")
    kdP = cs.SX.sym("kdP")
    vaCP = cs.SX.sym("vaCP")
    vdCP = cs.SX.sym("vdCP")

    klc = cs.SX.sym("klc") #do we need this or are we assuming removal here for simplicity
    vdC1 = cs.SX.sym("vdC1")
    kdC = cs.SX.sym("kdC")    
    vdC2 = cs.SX.sym("vdC2")
    
    klror = cs.SX.sym("klror")
    vdROR = cs.SX.sym("vdROR")
    kdROR = cs.SX.sym("kdROR") 
    
    klrev = cs.SX.sym("klrev")
    vdREV = cs.SX.sym("vdREV")
    kdREV = cs.SX.sym("kdREV") 
    
    vxROR = cs.SX.sym("vxROR")
    vxREV = cs.SX.sym("vxREV")
    kxREV = cs.SX.sym("kxREV")
    kxROR = cs.SX.sym("kxROR")
    vdb = cs.SX.sym("vdb")
    kdb = cs.SX.sym("kdb")

    klb = cs.SX.sym("klb")
    vdB = cs.SX.sym("vdB")
    kdB = cs.SX.sym("kdB")

    vdC1N = cs.SX.sym("vdC1N")
    kdCP = cs.SX.sym("kdCP")
    vdC2N = cs.SX.sym("vdC2N")
    mc = cs.SX.sym("mc")     
    
    paramset = cs.vertcat([vtp, ktp, kb, vdp, kdp, #0-4
                           vtc1, vtc2, ktc, vdc1, vdc2, kdc, #5-10
                           vtror, ktror, vdror, kdror, #11-14
                           vtrev, ktrev, vdrev, kdrev, #15-18
                           klp, vdP, kdP, vaCP, vdCP, #19-23
                           vdC1, kdC, vdC2, #klc removed from front, #24-26
                           klror, vdROR, kdROR, #27-29
                           klrev, vdREV, kdREV, #30-32
                           vxROR, vxREV, kxREV, kxROR, vdb, kdb, #33-38
                           klb, vdB, kdB, #39-41
                           vdC1N, kdCP, mc]) #42-44
    
    # Model Equations
    ode = [[]]*EqCount
    
    def txnE(vmax,km, kbc, dact1, dact2, Bc):
        return vmax/(km + (kbc/Bc) + dact1 + dact2)
        #return vmax/(km + (dact1 + dact2)**3)
    
    def txl(mrna,kt):
        return kt*mrna
    
    def MMdeg(species,vmax,km):
        return -vmax*(species)/(km+species)
        
    def cxMMdeg(species1,species2,vmax,km):
        return -vmax*(species1)/(km + species1 + species2)
        
    def cnrate(s1,s2,cmplx,ka,kd):
        # positive for reacting species, negative for complex
        return -ka*s1*s2 + kd*cmplx
        
    def txnb(vmax1, vmax2, k1, k2, species1, species2):
        return (vmax1*species1+vmax2)/(1+k1*species1+k2*species2)
    
    ode[0]  = txnE(vtp, ktp, kb, C1N, C2N, B) + MMdeg(p, vdp, kdp)
    #ode[0] = 0
    ode[1] = txnE(vtc1, ktc, kb, C1N, C2N, B) + MMdeg(c1, vdc1, kdc)
    #ode[1]=0
    ode[2] = txnE(vtc2, ktc, kb, C1N, C2N, B) + MMdeg(c2, vdc2, kdc)
    #ode[2] = 0
    ode[3] = txnE(vtror, ktror, kb, C1N, C2N, B) + MMdeg(ror, vdror, kdror)
    ode[4] = txnE(vtrev, ktrev, kb, C1N, C2N, B) + MMdeg(rev, vdrev, kdrev)
    #ode[3] = 0
    #ode[4]= 0
    ode[5]= txl(p, klp)+MMdeg(P, vdP, kdP)+cnrate(P, C1, C1N, vaCP, vdCP)+cnrate(P, C2, C2N, vaCP, vdCP)
    #ode[5]=0
    ode[6] = txl(c1, 1)+MMdeg(C1, vdC1, kdC)+cnrate(P, C1, C1N, vaCP, vdCP) #klc replaced with 1
    #ode[6] = 0
    ode[7] = txl(c2, 1)+MMdeg(C2, vdC2, kdC)+cnrate(P, C2, C2N, vaCP, vdCP) #klc replaced with 1
    #ode[7] =0
    ode[8] = txl(ror, klror)+MMdeg(ROR, vdROR, kdROR)
    ode[9] = txl(rev, klrev)+MMdeg(REV, vdREV, kdREV)
    #ode[8]=0
    #ode[9]=0
    ode[10] = txnb(vxROR, vxREV, kxREV, kxROR, REV, ROR)+MMdeg(b, vdb, kdb)
    ode[11] = txl(b, klb)+MMdeg(B, vdB, kdB)
    #ode[10]=0
    #ode[11]=0
    ode[12] = cxMMdeg(C1N, C2N, vdC1N, kdCP)-cnrate(C1, P, C1N, vaCP, vdCP)
    #ode[12] = 0
    ode[13] = cxMMdeg(C2N, C1N, mc*vdC1N, kdCP)-cnrate(C2, P, C2N, vaCP, vdCP) #changed to be in terms of vdC1N
    #ode[13]=0

    ode = cs.vertcat(ode)

    fn = cs.SXFunction(cs.daeIn(t=t, x=y, p=paramset),
                       cs.daeOut(ode=ode))
    
    fn.setOption("name","degmodel")
    
    return fn
    
    
if __name__ == "__main__":

    import LimitCycle as ctb    
    posmodel = ctb.Oscillator(model(), param)
    posmodel.calc_y0()
    
    print posmodel.T

    posmodel.intoptions['constraints']=None
    
    import matplotlib    
    
    #font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 28}

    #matplotlib.rc('font', **font)     
    
    dsol = posmodel.int_odes(3*posmodel.T, numsteps=7201)
    dts = posmodel.ts
    
    colormap = plt.cm.nipy_spectral
    plt.rcParams['font.family']='Arial'
    plt.figure()
    ax = plt.subplot(111)
    ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1, 14)])
    lineObjects = plt.plot(dts, dsol)   
    ax.set_xlabel('Time (cycles)')
    ax.set_xticks([0, posmodel.T, 2*posmodel.T, 3*posmodel.T])
    ax.set_xticklabels(['0', '1', '2', '3'])
    ax.set_ylabel('Expression Level')
    plt.legend(('p', 'c1', 'c2', 'ror', 'rev', 'P', 'C1', 'C2', 'ROR', 'REV', 'b', 'B', 'C1N', 'C2N'), loc='upper right')
    plt.show()
    