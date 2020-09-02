# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:32:17 2017

@author: lindsey
"""

from __future__ import division
import numpy  as np
import casadi as cs
import matplotlib.pyplot as plt


EqCount = 14
ParamCount = 45

y0in = np.array([2.94399774, 1.45982459, 1.37963843, 1.53271061, 0.64180771,
       2.51814106, 5.27221041, 5.14682548, 1.29418516, 0.89956422,
       0.5354595 , 0.36875376, 0.09342431, 0.05479389])
                 
period = 27.07168102648453
         
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
         .07588,   .05499,   .23010]


#parameter sets for different knockout conditions         
paramC1KO = [  .26726,   .33468,   .00117, .55011,   .00146,   
          0, .07313,   .28353,   .58013, .52993,   1.98812,   
          .08178, .23693,   .51117,   1.20013,
         1.15003,   .09945,   25.99980, 2.69984,   
         2.00001,   4.65000, .00031,   .00882,   .05999,
         1.59984,   2.08639,   1.51491,
         .35918,   1.29973,   1.95961,
         .25461,   1.29951,   1.96010,
         1.81951,   2.01031,   1.72406, 1.07518,   1.93448,   .55369,
         .37069,   1.86141,   2.71612,
         .07588,   .05499,   .23010]
         
paramC2KO = [  .26726,   .33468,   .00117, .55011,   .00146,   
          .08212, 0,   .28353,   .58013, .52993,   1.98812,   
          .08178, .23693,   .51117,   1.20013,
         1.15003,   .09945,   25.99980, 2.69984,   
         2.00001,   4.65000, .00031,   .00882,   .05999,
         1.59984,   2.08639,   1.51491,
         .35918,   1.29973,   1.95961,
         .25461,   1.29951,   1.96010,
         1.81951,   2.01031,   1.72406, 1.07518,   1.93448,   .55369,
         .37069,   1.86141,   2.71612,
         .07588,   .05499,   .23010]
         
paramCRYKO = [  .26726,   .33468,   .00117, .55011,   .00146,   
          0, 0,   .28353,   .58013, .52993,   1.98812,   
          .08178, .23693,   .51117,   1.20013,
         1.15003,   .09945,   25.99980, 2.69984,   
         2.00001,   4.65000, .00031,   .00882,   .05999,
         1.59984,   2.08639,   1.51491,
         .35918,   1.29973,   1.95961,
         .25461,   1.29951,   1.96010,
         1.81951,   2.01031,   1.72406, 1.07518,   1.93448,   .55369,
         .37069,   1.86141,   2.71612,
         .07588,   .05499,   .23010]
         
paramREVKO = [  .26726,   .33468,   .00117, .55011,   .00146,   
          .08212, .07313,   .28353,   .58013, .52993,   1.98812,   
          .08178, .23693,   .51117,   1.20013,
         0.0,   .09945,   25.99980, 2.69984,   
         2.00001,   4.65000, .00031,   .00882,   .05999,
         1.59984,   2.08639,   1.51491,
         .35918,   1.29973,   1.95961,
         .25461,   1.29951,   1.96010,
         1.81951,   2.01031,   1.72406, 1.07518,   1.93448,   .55369,
         .37069,   1.86141,   2.71612,
         .07588,   .05499,   .23010]
         
paramBMALKO = [  .26726,   .33468,   .00117, .55011,   .00146,   
          .08212, .07313,   .28353,   .58013, .52993,   1.98812,   
          .08178, .23693,   .51117,   1.20013,
         1.15003,   .09945,   25.99980, 2.69984,   
         2.00001,   4.65000, .00031,   .00882,   .05999,
         1.59984,   2.08639,   1.51491,
         .35918,   1.29973,   1.95961,
         .25461,   1.29951,   1.96010,
         0,   0,   1.72406, 1.07518,   1.93448,   .55369,
         .37069,   1.86141,   2.71612,
         .07588,   .05499,   .23010]
         
paramPERKO = [  0,   .33468,   .00117, .55011,   .00146,   
          .08212, .07313,   .28353,   .58013, .52993,   1.98812,   
          .08178, .23693,   .51117,   1.20013,
         1.15003,   .09945,   25.99980, 2.69984,   
         2.00001,   4.65000, .00031,   .00882,   .05999,
         1.59984,   2.08639,   1.51491,
         .35918,   1.29973,   1.95961,
         .25461,   1.29951,   1.96010,
         1.81951,   2.01031,   1.72406, 1.07518,   1.93448,   .55369,
         .37069,   1.86141,   2.71612,
         .07588,   .05499,   .23010]
         
paramRORKO = [  .26726,   .33468,   .00117, .55011,   .00146,   
          .08212, .07313,   .28353,   .58013, .52993,   1.98812,   
          0, .23693,   .51117,   1.20013,
         1.15003,   .09945,   25.99980, 2.69984,   
         2.00001,   4.65000, .00031,   .00882,   .05999,
         1.59984,   2.08639,   1.51491,
         .35918,   1.29973,   1.95961,
         .25461,   1.29951,   1.96010,
         1.81951,   2.01031,   1.72406, 1.07518,   1.93448,   .55369,
         .37069,   1.86141,   2.71612,
         .07588,   .05499,   .23010]
         
    

paramnames = ['vtp', 'ktp', 'kb', 'vdp', 'kdp',
              'vtc1', 'vtc2', 'ktc', 'vdc1', 'vdc2', 'kdc', 
              'vtror', 'ktror', 'vdror', 'kdror', 
              'vtrev', 'ktrev', 'vdrev', 'kdrev', 
              'klp', 'vdP', 'kdP', 'vaCP', 'vdCP', 
              'vdC1', 'kdC', 'vdC2', 
              'klror', 'vdROR', 'kdROR', 
              'klrev', 'vdREV', 'kdREV', 
              'vxROR', 'vxREV', 'kxREV', 'kxROR', 'vdb', 'kdb', 
              'klb', 'vdB', 'kdB', 
              'vdC1N', 'kdCP', 'vdC2N']

param_dict = {}
for i in range(45):
    param_dict[paramnames[i]] = i    


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
                           vdC1N, kdCP, vdC2N]) #42-44
    
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
    ode[1] = txnE(vtc1, ktc, kb, C1N, C2N, B) + MMdeg(c1, vdc1, kdc)
    ode[2] = txnE(vtc2, ktc, kb, C1N, C2N, B) + MMdeg(c2, vdc2, kdc)
    ode[3] = txnE(vtror, ktror, kb, C1N, C2N, B) + MMdeg(ror, vdror, kdror)
    ode[4] = txnE(vtrev, ktrev, kb, C1N, C2N, B) + MMdeg(rev, vdrev, kdrev)
    ode[5]= txl(p, klp)+MMdeg(P, vdP, kdP)+cnrate(P, C1, C1N, vaCP, vdCP)+cnrate(P, C2, C2N, vaCP, vdCP)
    ode[6] = txl(c1, 1)+MMdeg(C1, vdC1, kdC)+cnrate(P, C1, C1N, vaCP, vdCP) #klc replaced with 1
    ode[7] = txl(c2, 1)+MMdeg(C2, vdC2, kdC)+cnrate(P, C2, C2N, vaCP, vdCP) #klc replaced with 1
    ode[8] = txl(ror, klror)+MMdeg(ROR, vdROR, kdROR)
    ode[9] = txl(rev, klrev)+MMdeg(REV, vdREV, kdREV)
    ode[10] = txnb(vxROR, vxREV, kxREV, kxROR, REV, ROR)+MMdeg(b, vdb, kdb)
    ode[11] = txl(b, klb)+MMdeg(B, vdB, kdB)
    ode[12] = cxMMdeg(C1N, C2N, vdC1N, kdCP)-cnrate(C1, P, C1N, vaCP, vdCP)
    ode[13] = cxMMdeg(C2N, C1N, vdC2N, kdCP)-cnrate(C2, P, C2N, vaCP, vdCP)

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
    
    dsol = posmodel.int_odes(posmodel.T, numsteps = 100)
    dts = posmodel.ts

    colormap = plt.cm.nipy_spectral

    plt.figure()
    ax = plt.subplot(111)
    ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1, 14)])
    lineObjects = plt.plot(dts, dsol)   
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Expression Level')
    plt.legend(('p', 'c1', 'c2', 'ror', 'rev', 'P', 'C1', 'C2', 'ROR', 'REV', 'b', 'B', 'C1N', 'C2N'))
    plt.show()
    
    #get model parameter sensitivities
    posmodel.first_order_sensitivity()

    senses = posmodel.dTdp

    #ampsenses = posmodel.findARC_whole()   
    
    
    '''
    #this section contains code for analyzing the ARC and PRC from the LimitCycle module    
    
    posmodel.find_prc()

    #posmodel.findARC_whole()
    
    PRC_vdCP = [item[23] for item in posmodel.pPRC]
    PRC_vdC1N =[item[42] for item in posmodel.pPRC]
    PRC_vdC2N = [item[44] for item in posmodel.pPRC]

    PRC_KtxnREV = [item[35] for item in posmodel.pPRC]
    PRC_vdREV = [item[31] for item in posmodel.pPRC]

    PRC_vdC2 = [item[26] for item in posmodel.pPRC]
    
    phasetimes = [item/posmodel.T*2*np.pi for item in posmodel.prc_ts]
   
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$v_{d, C1N}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, PRC_vdC1N, linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial\Phi}{\partial t\partial\theta}$ (rad)', fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks([-np.pi, 0, np.pi], [r'$-\pi$', r'0', r'$\pi$'], fontsize = 35)
    plt.xlim(0, 2*np.pi)
    plt.tight_layout()
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$v_{d, C2N}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, PRC_vdC2N, linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial\Phi}{\partial t\partial\theta}$',  fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks([-np.pi, 0, np.pi], [r'$-\pi$', r'0', r'$\pi$'], fontsize = 35)
    plt.xlim(0, 2*np.pi)
    plt.tight_layout()
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$K_{txn, REV}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, PRC_KtxnREV, linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial\Phi}{\partial t\partial\theta}$',  fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks([-np.pi/100, 0, np.pi/100], [r'$\frac{-\pi}{100}$', r'0', r'$\frac{\pi}{100}$'], fontsize = 35)
    plt.xlim(0, 2*np.pi) 
    plt.tight_layout()
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$v_{d, REV}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, PRC_vdREV, linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial\Phi}{\partial t\partial\theta}$',  fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks([-np.pi/100, 0, np.pi/100], [r'$\frac{-\pi}{100}$', r'0', r'$\frac{\pi}{100}$'], fontsize = 35)
    plt.xlim(0, 2*np.pi) 
    plt.tight_layout()
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$v_{d, C2}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, PRC_vdC2, linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial\Phi}{\partial t\partial\theta}$',  fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks([-np.pi, 0, np.pi], [r'$-\pi$', r'0', r'$\pi$'], fontsize = 35)
    plt.xlim(0, 2*np.pi) 
    plt.tight_layout()
    
    #get ARCs
    ARC_vdC1Np = [item[0][42] for item in posmodel.pARC]
    ARC_vdC1Nb = [item[10][42] for item in posmodel.pARC] #b
    ARC_vdC1NB = [item[11][42] for item in posmodel.pARC] #B
    del ARC_vdC1Nb[24:26]
    del ARC_vdC1NB[24:26]
    
    phasetimes = [item/posmodel.T*2*np.pi for item in posmodel.arc_ts]
    del phasetimes[24:26]
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$v_{d, C1N}$', fontsize = 70, y = 1.05)
    #plt.plot(phasetimes, ARC_vdC1Np)
    plt.plot(phasetimes, ARC_vdC1Nb, 'r', linewidth = '3')
    plt.plot(phasetimes, ARC_vdC1NB, 'g', linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial A}{\partial t\partial\theta}$',  fontsize = 60)
    plt.legend(('Bmal1', 'BMAL1'), loc = 'upper left')
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.xlim(0, 2*np.pi)
    plt.tight_layout()

    ARC_vdC2Nb = [item[10][44] for item in posmodel.pARC] #b
    ARC_vdC2NB = [item[11][44] for item in posmodel.pARC] #B
    del ARC_vdC2Nb[24:26]
    del ARC_vdC2NB[24:26]
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$v_{d, C2N}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, ARC_vdC2Nb, 'r', linewidth = '3')
    plt.plot(phasetimes, ARC_vdC2NB, 'g', linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial A}{\partial t\partial\theta}$',  fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.xlim(0, 2*np.pi)
    plt.legend(('Bmal1', 'BMAL1'), loc = 'upper left')
    plt.tight_layout()
    
    ARC_KtxnREVb = [item[10][35] for item in posmodel.pARC] #b
    ARC_KtxnREVB = [item[11][35] for item in posmodel.pARC] #B 
    del ARC_KtxnREVb[24:26]
    del ARC_KtxnREVB[24:26]
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$K_{txn, REV}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, ARC_KtxnREVb, 'r', linewidth = '3')
    plt.plot(phasetimes, ARC_KtxnREVB, 'g', linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial A}{\partial t\partial\theta}$',  fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.xlim(0, 2*np.pi) 
    plt.legend(('Bmal1', 'BMAL1'), loc = 'upper left')
    plt.tight_layout()
    
    ARC_vdREVb = [item[10][31] for item in posmodel.pARC] #b
    ARC_vdREVB = [item[11][31] for item in posmodel.pARC] #B
    del ARC_vdREVb[24:26]
    del ARC_vdREVB[24:26]
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$v_{d, REV}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, ARC_vdREVb, 'r', linewidth = '3')
    plt.plot(phasetimes, ARC_vdREVB, 'g', linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial A}{\partial t\partial\theta}$',  fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks(fontsize = 35)
    plt.xlim(0, 2*np.pi) 
    plt.legend(('Bmal1', 'BMAL1'), loc = 'upper left')
    plt.tight_layout()
    
    ARC_vdC2b = [item[10][26] for item in posmodel.pARC] #b
    ARC_vdC2B = [item[11][26] for item in posmodel.pARC] #B 
    del ARC_vdC2b[24:26]
    del ARC_vdC2B[24:26]
    
    plt.figure(figsize = (20, 10))
    plt.title(r'$v_{d, C2}$', fontsize = 70, y = 1.05)
    plt.plot(phasetimes, ARC_vdC2b, 'r', linewidth = '3')
    plt.plot(phasetimes, ARC_vdC2B, 'g', linewidth = '3')
    plt.xlabel(r'Phase ($\Phi$)', fontsize = 40)
    plt.ylabel(r'$\frac{\partial A}{\partial t\partial\theta}$',  fontsize = 60)
    plt.xticks([0, np.pi, 2*np.pi], [r'0', r'$\pi$', r'$2\pi$'], fontsize = 35)
    plt.yticks([-.1, -.05, 0, .05, .1, .15], ['-.10', '-.05', '.00', '.05', '.10', '.15'], fontsize = 35)
    plt.xlim(0, 2*np.pi) 
    plt.legend(('Bmal1', 'BMAL1'), loc = 'upper left')
    plt.tight_layout()
    
    
    
    #PRC_vtp = [item[0] for item in posmodel.pPRC]
    #plt.figure()
    #plt.title('Extended vtp PRC')
    #plt.plot(posmodel.prc_ts, PRC_vtp)
    '''
    