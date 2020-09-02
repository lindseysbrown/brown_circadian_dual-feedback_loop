"""

solves the mpc problem using multi input control for the extended model
"""

# imports
from __future__ import division

import numpy as np
from scipy import integrate, optimize, stats, interpolate
import matplotlib.pyplot as plt
from matplotlib import gridspec
import casadi as cs
import brewer2mpl as colorbrewer
from concurrent import futures

#local imports
from controlmodel import model, param, y0in, paramset
import LimitCycle as lco
import PlotOptions as plo
import Utilities as ut

#set up model
pmodel = lco.Oscillator(model(), param, y0in)
pmodel.calc_y0()
pmodel.corestationary()
pmodel.limit_cycle()
pmodel.find_prc()

paramlist = ['vtp', 'ktp', 'kb', 'vdp', 'kdp', 'vtc1', 'vtc2', 'ktc', 'vdc1', 'vdc2', 'kdc',
             'vtror', 'ktror', 'vdror', 'kdror', 'vtrev', 'ktrev', 'vdrev', 'kdrev',
             'klp', 'vdP', 'kdP', 'vaCP', 'vdCP', 'vdC1', 'kdC', 'vdC2', 'klror', 'vdROR', 'kdROR',
             'klrev', 'vdREV', 'kdREV', 'vxROR', 'vxREV', 'kxREV', 'kxROR', 'vdb', 'kdb',
             'klb', 'vdB', 'kdB', 'vdC1N', 'kdCP', 'mc']


#comment out appropriate settings depending on desired inputs

#KL001 settings
#control_params = ['vdC1N']
#direction = [-1]

#Longdaysin settings
#control_params =['vdP']
#direction = [-1]

#Longdaysin and KL001 settings
#control_params = ['vdC1N', 'vdP']
#direction = [-1, -1]

#Longdaysin and FBXW7 settings
#control_params = ['vdP', 'vdREV']
#direction = [-1, 1]

#KL001 and FBXW7
control_params=['vdC1N', 'vdREV']
direction = [-1, 1]
 
power = .5
tol = .01

def getPRCs(controlparam):
    indices = [paramlist.index(controlin) for controlin in controlparam] #which parameters manipulating
    def prc_control(x):
        return np.array([pmodel.pPRC_interp.splines[i](x) for i in indices])
    maxu = [power*param[i] for i in indices]
    return prc_control, maxu, indices
    
prc_control, maxu, indices = getPRCs(control_params)

# start at negative prc region
times = np.linspace(0,pmodel.T,10001)
roots = pmodel.pPRC_interp.splines[indices[0]].root_offset()
neg_start_root = roots[0] # for when region becomes positive
pos_start_root = roots[1] # for when region becomes negative


start_time = 0


# define integration for one step

def step_integrate(phi0, u_vals, step):
    """ function that integrates one step forward. returns final phase,
    total shift value """
    def dphidt(phi, t):
        f = (2*np.pi)/pmodel.T
        for i in range(len(u_vals)):
            f = f + direction[i]*u_vals[i]*prc_control(start_time+(phi)*pmodel.T/(2*np.pi))[i]
        return f

    int_times = np.linspace(0,step,101) # in hours
    phis = integrate.odeint(dphidt, [phi0], int_times, hmax=0.01)
    return int_times, phis, phis[-1][0]-phi0-2*np.pi/pmodel.T*step


def getmatrix(us, pred_horiz):
    controlins = len(us)/pred_horiz
    if controlins != len(control_params):
        raise ValueError('The length of the control input guess is not correct')
    ureshape = []
    for i in range(len(control_params)):
        ureshape.append(np.array(us[i*pred_horiz:(i+1)*pred_horiz]))
    ureshape = np.array(ureshape).T
    return ureshape

def mpc_pred_horiz(init_phi, delta_phi_f, stepsize, pred_horiz):
    """
    performs the optimization over the predictive horizon to give the optimal
    set of steps u

    uses a dividing line of pi/2 as the adv/del
    """
    dividing_phi_f = np.pi/2 #should be calculated for each control input

    # get init phase
    phi0 = init_phi
    # get the ref phase
    ref_phase0 = init_phi + delta_phi_f
    # get times
    steps = np.arange(pred_horiz)
    tstarts = steps*stepsize
    tends   = (steps+1)*stepsize

    # get ref phases at end of each step
    ref_phis_pred = tends*2*np.pi/pmodel.T + ref_phase0

    # choose the direction
    if delta_phi_f > dividing_phi_f:
        # achieve the shift by a delay
        def optimize_inputs(us, phi0=phi0):
            endphis = []
            ureshape = getmatrix(us, pred_horiz)
            for u in ureshape:
                results = step_integrate(phi0, u, stepsize)
                endphis.append(results[1][-1])
                phi0 = results[1][-1][0]
            endphis = np.asarray(endphis).flatten()
            errs= np.abs(endphis-(ref_phis_pred-2*np.pi))
            return 10*errs.sum()+us.sum()
    else:
        # achieve the shift by an advance
        def optimize_inputs(us, phi0=phi0):
            endphis = []
            ureshape = getmatrix(us, pred_horiz)
            for u in ureshape:
                results = step_integrate(phi0, u, stepsize)
                endphis.append(results[1][-1])
                phi0 = results[1][-1][0]
            endphis = np.asarray(endphis).flatten()
            errs= np.abs(endphis-ref_phis_pred)
            return 10*errs.sum()+us.sum()
            
    guess = np.array([])
    bound = np.array([]).reshape(0, 2)
    for i in range(len(control_params)):
        guess = np.concatenate((guess, [.5*maxu[i]]*pred_horiz))
        bound = np.concatenate((bound, [[0, maxu[i]]]*pred_horiz))        

    mins = optimize.minimize(optimize_inputs, guess,
                             bounds = bound)
    uresult = getmatrix(mins.x, pred_horiz)
    return uresult




def mpc_problem(init_phi, ts, ref_phis, pred_horizon):
    """
    uses MPC to track the reference phis, using a stepsize and a predictive horizon.
    ts should be separated by stepsize.
    """
    # set up the system it gets applied to
    y0mpc = pmodel.lc(init_phi*pmodel.T/(2*np.pi)+start_time)

    # get step size, current phase, etc
    stepsize = ts[1]-ts[0]
    u_input = []
    sys_state = y0mpc

    sys_phis = []
    for idx, inst_time in enumerate(ts):
        #get the ref phase at the time, compare to system phase at the time
        ref_phi = ref_phis[idx]
        
        sys_phi = (pmodel.phase_of_point(sys_state, tol = tol)-
                        pmodel._t_to_phi(start_time))%(2*np.pi)
        sys_phis.append(sys_phi)

        # phase error
        phi_diff = np.angle(np.exp(1j*(sys_phi-ref_phi))) 
        delta_phi_f = -phi_diff 

        if np.abs(phi_diff) > 0.1: #this value may be changed as desired
            # calculate the optimal inputs
            us_opt = mpc_pred_horiz(sys_phi, delta_phi_f, stepsize, pred_horizon)
            u_apply = us_opt[0]

        else:
            u_apply = [0]*len(control_params)


        print delta_phi_f, u_apply
        # move forward a step
        u_input.append(u_apply)
        mpc_param = updateparam(u_apply)        
        mpc_sys = lco.Oscillator(model(), mpc_param, sys_state)
        sys_progress = mpc_sys.int_odes(stepsize)
        sys_state = sys_progress[-1]

    return ts, sys_phis, u_input

def updateparam(u):
    mpc_param = np.copy(param)
    for (ind, val, direct) in zip(indices, u, direction):
        mpc_param[ind] = mpc_param[ind]+direct*val
    return mpc_param



def solutions_from_us(us, stepsize, init_phi):
    """ takes a set of us, a stepsize, and a phi0; returns the trajectories of the solutions """

    pred_horiz = len(us)
    phases = []
    ref_phases = []
    states = []
    times = []
    running_time = 0
    phi0 = init_phi

    y0mpc = pmodel.lc(phi0*pmodel.T/(2*np.pi)+start_time)
    sys_state = y0mpc

    for u_apply in us:
        mpc_param = updateparam(u_apply)        
        mpc_sys = lco.Oscillator(model(), mpc_param, sys_state)
        sys_progress = mpc_sys.int_odes(stepsize, numsteps=10)

        # append new times and phases
        times = times+list(mpc_sys.ts[:-1]+running_time)
        phases = phases + [pmodel.phase_of_point(state, tol = tol) for state in sys_progress[:-1]]

        #update for next step
        sys_state = sys_progress[-1]
        running_time = running_time+mpc_sys.ts[-1]

    u0_phases = init_phi+np.asarray(times)*2*np.pi/pmodel.T

    return {'u0_phases': np.asarray(u0_phases),
            'times': np.asarray(times),
            'phases': np.asarray(phases),
            'us': np.asarray(us)
            }

# phase changes
t1 = 6
t2 = 42
ref_phase_jump1 = 5/24*2*np.pi
ref_phase_jump2 = -11/24*2*np.pi

# solve the mpc problem
initial_phase = 0
step_2h = 2*pmodel.T/24
ts = np.arange(0,200,step_2h)
time_jump1 = ts[t1]
time_jump2 = ts[t2]

reference_phases = (2*np.pi/pmodel.T)*ts +\
             np.array([0]*t1+[ref_phase_jump1]*(len(ts)-t1)) +\
             np.array([0]*t2+[ref_phase_jump2]*(len(ts)-t2))
ts, sys_phis, us_mpc = mpc_problem(0, ts, reference_phases, 3)

# collect the result from us
mpc_solution = solutions_from_us(us_mpc, step_2h, initial_phase)

# get more tightly sampled phases
ts_tight = np.arange(0,200,0.1)
ref_phases_tight = (2*np.pi/pmodel.T)*ts_tight
for idx, time in enumerate(ts_tight):
    if time > time_jump1:
        ref_phases_tight[idx] = ref_phases_tight[idx]+ref_phase_jump1
    if time > time_jump2:
        ref_phases_tight[idx] = ref_phases_tight[idx]+ref_phase_jump2

# plot to show it works
plt.plot(ts,reference_phases, label = 'step phis')
plt.plot(ts_tight, ref_phases_tight, label = 'tight phis')
plt.plot(mpc_solution['times'],
    np.unwrap(mpc_solution['phases'])-pmodel._t_to_phi(start_time),
    label='mpc phis')
plt.legend()


results = mpc_solution
normtimes = results['times']*24/pmodel.T #divide by model period
normtstight = ts_tight*24/pmodel.T

np.savez('KL001FBXoutput.npz', phases = results['phases'], times=results['times'], us = results['us'])


plo.PlotOptions(ticks='in')
plt.figure(figsize=(10,8.5))
gs = gridspec.GridSpec(2,1, height_ratios = (2.5,1))
ax = plt.subplot(gs[0,0])

ax.plot(normtimes, np.sin(results['phases']-pmodel._t_to_phi(start_time)),
    'i', linewidth = 3, label = 'MPC Solution')
ax.plot(normtstight, np.sin(ref_phases_tight), 'k:', label='Reference Trajectory', linewidth = 3)
ax.set_xlim([0,180])
ax.set_ylabel('sin($\phi$)', fontsize = 20)
ax.set_xticklabels('')
ax.set_ylim([-1.05,1.35])
ax.legend()
ax.set_xticks([0,24,48,72,96,120,144,168])
ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
ax.set_title(r'Control with KL001 and FBXW7-$\alpha$', fontsize = 30)

bx = plt.subplot(gs[1,0])
# get spline representation of phis
for i in range(len(control_params)):
    bx.plot(normtimes, prc_control(pmodel._phi_to_t(results['phases'])).T, 'k', label='PRC', linewidth = 3)

bx2 = bx.twinx()
for i in range(len(control_params)):
    us = np.hstack([[0],results['us'][:, i]])
    utimes = 2*np.arange(len(us))
    bx2.step(utimes, us, linewidth = 3)

#bx2.set_ylim([0.0,0.1])
bx2.set_ylabel('u(t)', fontsize = 20)
bx.set_ylabel('PRC', fontsize = 20)
bx.set_ylim([-4, 4])
bx.set_xlim([0,180])
bx.set_xticks([0,24,48,72,96,120,144,168])
bx.set_xlabel('Time (h)', fontsize = 20)
bx.tick_params(axis = 'both', which = 'major', labelsize = 15)
plt.tick_params(axis = 'both', which = 'major', labelsize = 15)
plt.legend()
plt.tight_layout(**plo.layout_pad)


