# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:51:08 2017

@author: lindsey
"""

from __future__ import division
from scoop import futures

import random
import numpy as np

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

from scipy.interpolate import splrep, splev, UnivariateSpline

import LimitCycle as ctb 
import model as pm

import math

#Creator
#create maximizing fitness
creator.create("FitnessMax", base.Fitness, weights = (-1.0,)) #remember the comma so weights iterable
#create individual class with properties of list and fitness attribute type FitnessMax
creator.create("Individual", list, fitness = creator.FitnessMax)

#Toolbox
toolbox = base.Toolbox()

def sigmoid(x):
    if x>50:
        return 1
    if x<-50:
        return 0
    return 1/(1+np.exp(-x))

def make(icls):
    
    #settings for Peter model    
    #return  icls(np.random.normal(loc = [.1946 , .1306 , .1135 , .4255 , .2595 , .3263 , .6761 , .6079 , .0115 , 1.149 , 2.9699, .0338 , 1.523 , 1.6856, 2.01721, .1012 , 3.3179 , .05263, .04063 , .001755 , 3.00],
     #            scale = [.000001 , .000001 , .000001 , .000005 , .00005 , .00005 , .00005 , .00005 , .000001 , .00001 , .00003, .000001 , .00001 , .00001, .00001, .00001 , .00003 , .000005, .000005 , .000001 , .00003]))
    #original initialization
    '''
    return  icls(np.random.normal(loc = [  .195,   .425,   .01, .326,   .011,   
          .131, .114,   .259,   .676, .608,   1.149,   
          .1, 1,   .5,   5,
         .1,   1,   .5, 5,   
         3,   2.97, .034,   .041,   .002,
         1.523,   2.017,   1.69,
         .3,   1,   1,
         .5,   1,   1,
         1,   2,   1, 2,   2,   .5,
         1,   2,   .5,
         .101,   .053,   .335]
,

                 scale = [.005, .005, .007, .001, .0001,
                           .005, .005, .005, .0001, .0001, .001,
                          .005, .02, .02, .02,
                           .005, .02, .02, .02,
                           .001, .001, .0001, .0001, .00001,
                            .001, .001, .001,
                           .005, .05, .05,
                           .005, .005, .005,
                           .05, .05, .05, .05, .05, .04,
                           .05, .05, .04,
                           .0001, .0001, .0001]))
    '''
    #closer initialization point   
        
    return  icls(np.random.normal(loc = [  .18,   .425,   .01, .32,   .02,   
          .096, .086,   .259,   .63, .56,   1.2,   
          .1, .26,   .55,   1.15,
         1.4,   .15,   24, 3,   
         2.5,   2.97, .034,   .041,   .002,
         1.52,   2.02,   1.69,
         .19,   .7,   1.4,
         .08,   .2,   .95,
         5.2,   11,   8.1, 8.1,   6,   2.4,
         .4,   1.49,   .491,
         .101,   .053,   .335]
,

                 scale = [.005, .005, .007, .001, .0001,
                           .005, .005, .005, .0001, .0001, .001,
                          .005, .02, .02, .02,
                           .005, .02, .02, .02,
                           .001, .001, .0001, .0001, .00001,
                            .001, .001, .001,
                           .005, .05, .05,
                           .005, .005, .005,
                           .05, .05, .05, .05, .05, .04,
                           .05, .05, .04,
                           .0001, .0001, .0001]))
    
    
    #use the settings below to check that the parameter set doesn't change more with further iterations of the GA
    '''    
    return  icls(np.random.normal(loc = [  .267,   .335,   .00075, .55008,   .00136,   
          .082, .07329,   .28359,   .58, .53,   1.988,   
          .081, .23735,   .51142,   1.2,
         1.15000,   .09984,   25.99991, 2.69975,   
         2,   4.65, .0003,   .00893,   .06,
         1.6,   2.08659,   1.515,
         .35926,   1.3,   1.96,
         .25462,   1.3,   1.96,
         1.82,   2.01,   1.72413, 1.07425,   1.93441,   .55293,
         .37090,   1.86027,   2.71585,
         .07597,   .055,   .23]
,

                 scale = [.0005, .0005, .0002, .0001, .0001,
                           .0005, .0005, .0005, .0001, .0001, .0001,
                          .0005, .0002, .0002, .0002,
                           .00005, .0002, .0002, .0002,
                           .00001, .00001, .00001, .0001, .00001,
                            .0001, .0001, .0001,
                           .0005, .0005, .0005,
                           .0005, .0005, .0005,
                           .0005, .0005, .0005, .0005, .0005, .0004,
                           .0005, .0005, .0004,
                           .0001, .0001, .0001]))
    '''

toolbox.register("individual", make, icls = creator.Individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual) 

def evalModel(individual, toPrint = False):
    numsteps = 1000
    individual = [round(item, 5) for item in individual]
    if  not np.all(individual>0):
        return np.inf, 
    try:
        posL = ctb.Oscillator(pm.model(), individual)
        posL.calc_y0()
        if posL.T<10:
            return np.inf,         
        posL.intoptions['constraints']=None
        dsol = posL.int_odes(posL.T, numsteps=numsteps)
        dts = posL.ts
        
        p = dsol[:, 0]
        c1 = dsol[:, 1]
        c2 = dsol[:, 2]
        ror = dsol[:, 3]
        rev = dsol[:, 4]
        P = dsol[:, 5]
        C1 = dsol[:,6]
        C2 = dsol[:, 7]
        ROR = dsol[:, 8]
        REV = dsol[:, 9]
        b = dsol[:, 10]
        B = dsol[:, 11]
        C1N = dsol[:, 12]
        C2N = dsol[:, 13]
        
        
        T1 = numsteps
        
        #correct peak to trough ratios
        costs = np.zeros(43)    
        costs[0] = 10*sigmoid(10 - max(p)/min(p))
        costs[1] = np.abs(max(c1)/min(c1)-2.1)**2
        costs[2] = np.abs(max(c2)/min(c2)-2.2)**2
        costs[3] = 3*np.abs(max(ror)/min(ror)-4.1)**2
        costs[4] = 5*sigmoid(10 - max(rev)/min(rev))
        costs[5] = 5*sigmoid(10-max(b)/min(b))
        costs[6] = 5*sigmoid(20-max(P)/min(P))
        costs[7] = np.abs(max(C1)/min(C1)-3.7)**2
        costs[8] = np.abs(max(C2)/min(C2)-1.8)**2
        costs[9] = 5*sigmoid(max(ROR)/min(ROR)-5)
        costs[10] = 5*sigmoid(max(REV)/min(REV)-5)
        costs[11]= 3*np.abs(max(B)/min(B)-2.9)**2
        
        #correct species ratios
        costs[12]= 10*np.abs(max(P)/(max(P)+max(C1)+max(C2))-.11)**2
        costs[13]= 10*np.abs(max(C1)/(max(P)+max(C1)+max(C2))-.56)**2
        costs[14]= 10*np.abs(max(C2)/(max(P)+max(C1)+max(C2))-.33)**2
        costs[15]= 3*np.abs(max(B)/max(C1)-.19)**2
        costs[16]= 3*np.abs(max(ROR)/max(REV)-1.02)**2
        costs[17]= 3*np.abs(max(ror)/max(rev)-.78)**2
        costs[18]= 3*np.abs(max(b)/max(p)- .78)**2
        costs[19]= 3*np.abs(max(b)/max(ror)-1.01)**2
        costs[20]= 3*np.abs(max(b)/max(rev)-.78)**2
        costs[21]= 3*np.abs(max(p)/max(ror)-1.29)**2
        costs[22]= 3*np.abs(max(p)/max(rev)- .99)**2
        
        #correct phase differences
        pPeak = np.argmax(p)
        PPeak = np.argmax(P)
        if PPeak>pPeak:
            PPeak = PPeak-T1
        costs[23]=10*np.abs((pPeak-PPeak)/T1-.75)**2
        
        c1Peak = np.argmax(c1)
        C1Peak = np.argmax(C1)
        if C1Peak>c1Peak:
            C1Peak = C1Peak-T1
        costs[24]=5*np.abs((c1Peak-C1Peak)/T1-.75)**2
        
        c2Peak = np.argmax(c2)
        C2Peak = np.argmax(C2)
        if C2Peak>c2Peak:
            C2Peak = C2Peak-T1
        costs[25]=5*np.abs((c2Peak-C2Peak)/T1-.67)**2
        
        bPeak = np.argmax(b)
        BPeak = np.argmax(B)
        costs[26] = 5*(min(np.abs(bPeak-BPeak), np.abs(BPeak-(bPeak+T1)), np.abs((BPeak+T1)-bPeak))/T1)**2
        
        rorPeak = np.argmax(ror)
        RORPeak = np.argmax(ROR)
        if RORPeak>rorPeak:
            RORPeak = RORPeak - T1
        costs[27] = 5*np.abs((rorPeak-RORPeak)/T1-.67)**2
        
        revPeak = np.argmax(rev)
        REVPeak = np.argmax(REV)
        if REVPeak>revPeak:
            REVPeak = REVPeak-T1
        costs[28]= 5*np.abs((revPeak-REVPeak)/T1-.58)**2
        
        bPeak = np.argmax(b)
        pPeak = np.argmax(p)
        if pPeak>bPeak:
            pPeak = pPeak - T1
        costs[29]= 15*np.abs((bPeak-pPeak)/T1-.54)**2
        
        c1Peak = np.argmax(c1)
        bPeak = np.argmax(b)        
        if c1Peak>bPeak:
            c1Peak = c1Peak - T1
        costs[30] = 15*np.abs((bPeak-c1Peak)/T1-.25)**2
        
        if rorPeak>bPeak:
            rorPeak = rorPeak - T1
        costs[31] = 5*np.abs((bPeak-rorPeak)/T1-.29)**2
        
        if revPeak>bPeak:
            revPeak = revPeak - T1
        costs[32]= 5*np.abs((bPeak-revPeak)/T1-.67)**2
        
        revPeak = np.argmax(rev)
        PPeak = np.argmax(P)
        if PPeak>revPeak:
            PPeak = PPeak-T1
        costs[33] = 50*np.abs((revPeak-PPeak)/T1-.5)**2
        
        rorPeak = np.argmax(ror)
        if rorPeak>revPeak:
            rorPeak = rorPeak-T1
        costs[34]= 5*np.abs((revPeak-rorPeak)/T1-.79)**2
        
        C1Peak = np.argmax(C1)
        C1NPeak = np.argmax(C1N)
        costs[35] = 5*(min(np.abs(C1Peak-C1NPeak), np.abs(C1NPeak-(C1Peak+T1)), np.abs((C1NPeak+T1)-C1Peak))/T1)**2
        
        C2Peak = np.argmax(C2)
        C2NPeak = np.argmax(C2N)
        costs[36] = 5*(min([np.abs(C2Peak-C2NPeak), np.abs(C2NPeak-(C2Peak+T1)), np.abs((C2NPeak+T1)-C2Peak)])/T1)**2
        
        #effects of removal of Cry1 and Cry2
        
        Torig = posL.T        
        
        posL.first_order_sensitivity()
        senses = posL.dTdp
        costs[37] = 50*(senses[8]>0)
        costs[38] = 50*(senses[9]<0)
        
        cry1Knock = np.copy(individual)
        cry1Knock[5] = 0
        pcry1Knock = ctb.Oscillator(pm.model(), cry1Knock, posL.y0)
        pcry1Knock.approx_y0_T()
        TC1 = pcry1Knock.T  
        costs[39] = 100*(TC1/Torig > .95)+200*(TC1/Torig<.1)+300*math.isnan(TC1)        
        
        cry2Knock = np.copy(individual)
        cry2Knock[6] = 0
        pcry2Knock = ctb.Oscillator(pm.model(), cry2Knock, posL.y0)
        pcry2Knock.approx_y0_T()
        TC2 = pcry2Knock.T  
        costs[40] = 100*(TC2/Torig < 1.15)+300*math.isnan(TC2) 
        
        pPeak = np.argmax(p)
        c2Peak = np.argmax(c2)
        costs[41] = 5*(min([np.abs(pPeak-c2Peak), np.abs(c2Peak-(pPeak+T1)), np.abs((c2Peak+T1)-pPeak)])/T1)**2
        
        costs[42] = 20*(max(C1)<max(C2))
        
        if toPrint:
            return sum(costs), costs
        
        return sum(costs),
        
    except:
        return np.inf,
        
toolbox.register("evaluate", evalModel)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, 
                 mu = [0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0,
                           0, 0, 0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0,
                           0, 0, 0, 0, 0, 0,
                           0, 0, 0,
                           0, 0, 0],
  
                         
                sigma = [.05, .05, .0005, .04, .0005,
                           .005, .005, .05, .04, .04, .5,
                           .005, .04, .05, .5,
                           .05, .005, 2, .5,
                           .5, .5, .0005, .005, .005,
                           .5, .5, .5,
                           .05, .5, .5,
                           .05, .5, .5,
                           .5, .5, .5, .5, .5, .05,
                           .05, .5, .5,
                           .005, .005, .05],                          
                 indpb = [.05, .05, .05, .05, .05,
                           .05, .05, .05, .05, .05, .05,
                           .05, .05, .05, .05,
                           .05, .05, .05, .05,
                           .05, .05, .05, .05, .05,
                           .05, .05, .05,
                           .05, .05, .05,
                           .05, .05, .05,
                           .05, .05, .05, .05, .05, .05,
                           .05, .05, .05,
                           .05, .05, .05])
toolbox.register("select", tools.selTournament, tournsize = 3)

toolbox.register("map", futures.map)

NGEN = 50
CXPB = .1
MUTPB = .4

def main():
    pop = toolbox.population(n=10000)
    fitnesses = list(futures.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop,fitnesses):
        ind.fitness.values = fit
    
    evolution = [None]*NGEN    
    
    hof = tools.HallOfFame(1)
    hof.update(pop)
    for g in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = pop
        offspring = list(map(toolbox.clone, offspring))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits)/length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2/length - mean**2)**.5 
        hof.update(pop)
        try:
            evolution[g] = evalModel(hof[0], toPrint = True)[1]
        except:
            evolution[g] = -1

    return hof[0], evolution
    
if __name__ == '__main__':
    bestparamsposL, evolution = main()
    np.save("ParamSet.npy", bestparamsposL)
    np.save("Evolution.npy", evolution)
