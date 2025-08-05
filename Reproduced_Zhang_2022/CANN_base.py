#!/usr/bin/env python3

"""
Program:  cann_base.py
Author:   Chi Chung Alan Fung

This program is an implementation of the model used in the papers by Fung, 
Wong and Wu in 2008 and 2010. This program is written for those who are 
interested in the work. Also, this program code could be a reference program 
code for researchers who are going to work on the model.

The equation used here may be slightly different from the equation reported in 
the paper by Fung, Wong and Wu (2008), due to the rescaling. The equation used 
here is given by

\tau \frac{du(x,t)}{dt} = 
-u(x,t) + \int dx^\prime J(x,x^\prime) r(x^\prime, t) 
+ A \exp\left[-\frac{\left|x-z_0\right|^2}{4a^2}\right]

J(x,x^\prime) = 
\left(\right)\exp\left[-\frac{\left|x-x^\prime\right|^2}{2a^2}\right]

r(x,t) =
\frac{u(x,t)^2}{1+(k/(8*\sqrt{2\pi}a))*\int dx^\prime u(x^\prime, t)^2}
"""

# Modules
import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt
import argparse

# Class for the model
class cann_model:
    # define the range of perferred stimuli
    z_min = - np.pi;              
    z_range = 2.0 * np.pi;
    # define the time scale
    tau = 1.0
        
    # function for periodic boundary condition
    def dist(self, c):
        tmp = np.remainder(c, self.z_range)
        
        # routine for numbers
        if isinstance(tmp, (int, float)):
            if tmp > (0.5 * self.z_range):
                return (tmp - self.z_range);
            return tmp;
        
        # routine for numpy arraies
        for tmp_1 in np.nditer(tmp, op_flags=['readwrite']):
            if tmp_1 > (0.5 * self.z_range):
                tmp_1[...] = tmp_1 - self.z_range;
        
        return tmp;
    
    # constructor (?)
    def __init__(self, argument):
        self.k = argument.k;    # rescaled inhibition
        self.a = argument.a;    # range of excitatory connection
        self.N = argument.N;    # number of units / neurons
        self.dx = self.z_range / self.N     # separation between neurons
        
        # define perferred stimuli for each neuron
        self.x = (np.arange(0,self.N,1)+0.5) * self.dx + self.z_min;
        
        # calculate the excitatory couple for each pair of neurons
        self.Jxx = np.zeros((self.N, self.N));
        for i in range(self.Jxx.shape[0]):
            for j in range(self.Jxx.shape[1]):
                self.Jxx[i][j] = \
                np.exp(-0.5 * np.square(self.dist(self.x[i] \
                                                  - self.x[j]) / self.a)) \
                / (np.sqrt(2*np.pi) * self.a);
                
        self.u = np.zeros((self.N));    # initialize neuronal inputs
        self.r = np.zeros((self.N));    # initialize neuronal activities
        self.input = np.zeros((self.N));    # initialial the external input
    
    # function for setting external iput for each neuron
    def set_input(self, A, z0):
        self.input = \
        A * np.exp(-0.25 * np.square(self.dist(self.x - z0) / self.a));
    
    # function for calculation of neuronal activity of each neuron
    def cal_r_or_u(self, u):
        u0 = 0.5 * (u + np.abs(u));
        r = np.square(u0);
        B = 1.0 + 0.125 * self.k * np.sum(r) * self.dx \
        / (np.sqrt(2*np.pi) * self.a);
        r = r / B;
        return r;
    
    # 
    def cm_of_u(self):
        max_i = self.u.argmax()
        cm = np.dot(self.dist(self.x - self.x[max_i]), self.u) / self.u.sum()
        cm = cm + self.x[max_i]
        return cm;
    
    # function for calculation of derivatives
    def get_dudt(self, t, u):
        dudt = \
        -u + np.dot(self.Jxx, self.cal_r_or_u(u)) * self.dx + self.input;
        dudt = dudt / self.tau;
        return dudt
        
                
"""
Begining of the program
"""
# acquiring parameters
parser = argparse.ArgumentParser(description="")

parser.add_argument("-k", metavar="float", type=float, \
                    help="rescaled Inhibition", default=0.5)

parser.add_argument("-a", metavar="float", type=float, \
                    help="width of excitatory couplings", default=0.5)

parser.add_argument("-N", metavar="int", type=int, \
                    help="number of excitatory units", default=128)

parser.add_argument("-A", metavar="float", type=float, \
                    help="magnitude of the external input", default=0.5)

parser.add_argument("-z0", metavar="float", type=float, \
                    help="sudden change of the external input", \
                    default=0.5*np.pi)

arg = parser.parse_args()

# construct a CANN object
cann = cann_model(arg)

# setting up an initial condition of neuronal inputs 
# so that tracking can be reasonabl for small A and k < 1
if arg.k < 1.0:
    cann.set_input(np.sqrt(32.0)/arg.k, 0)
else:
    cann.set_input(np.sqrt(32.0), 0)
cann.u = cann.input

# setting up an external input according to the inputted parameter
cann.set_input(arg.A, 0)

# run the simulation for 100 tau to initialize the network state
# before the shift of the external input
out = spint.solve_ivp(cann.get_dudt, (0, 100), cann.u, method="RK45");

# update the network state in the CANN object
cann.u = out.y[:,-1]

# change the stimulus location from 0 to z0
cann.set_input(arg.A, arg.z0)

# take a initial snapshot
snapshots = np.array([cann.u])

# run the simulation and take snapshots every 10 taus
for t in range(0,20000,10):
    # decide the period of this step
    t0 = t
    t1 = t + 10
    # run the simulation and update the state in the CANN object
    out = spint.solve_ivp(cann.get_dudt, (t0, t1), cann.u, method="RK45");
    cann.u = out.y[:,-1]
    # store the snapshot
    snapshots = np.append(snapshots, [cann.u.transpose()], axis=0)
    # if the center of mass of the neuronal input is close to 
    # the destination, simulation terminates.
    if np.abs(cann.cm_of_u() - arg.z0) < (0.05):
        break;
    
# make a graphic output of the result
out_fig = plt.figure()

# define title and axes' title
plt.xlabel(r'$x$')
plt.ylabel(r'$\tilde{u}(x,t)$')
title_out = "Tracking from 0 to %0.3f.\n"\
+"Snapshots were taken every 10 "\
+r'$\tau$'
plt.title(title_out % arg.z0)

# determining right range for the plot
y_max_tmp = np.ceil(snapshots.max()*1.2)
plt.xlim(xmax=np.pi, xmin=-np.pi)
plt.ylim(ymax=y_max_tmp, ymin=y_max_tmp*-0.05)

# draw an arrow to indicate the direction
plt.arrow(0,cann.u.max()*1.1,(arg.z0-0.2),0, head_width=0.4, \
          head_length=0.2, color='black')

# plot all the stored snapshots
for i in range(snapshots.shape[0]):
    plt.plot(cann.x, snapshots[i,:])

# show the plot
out_fig.show()
plt.show()

# exit the program
exit()