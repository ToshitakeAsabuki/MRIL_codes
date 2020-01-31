"""
Copyright 2019 Toshitake Asabuki

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
        
http://www.apache.org/licenses/LICENSE-2.0
            
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# coding: UTF-8
from __future__ import division
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy.matlib
import os
import shutil
import sys
import matplotlib.cm as cm

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['pdf.fonttype'] = 42
params = {'backend': 'ps',
    'axes.labelsize': 11,
    'text.fontsize': 11,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.usetex': False,
    'figure.figsize': [10 / 2.54, 6 / 2.54]}

"""
    In this demo code, the true signals are periodic, but you can train with much complex signals.
    You can introduce inhibitory STDP for lateral inhibition to train with unknown number of sources.
    """
    


##############
"""
    activation function
"""
beta = 5
def g(x):

    alpha = 1
  
    theta= 0.5

    ans = 1/(1+alpha*np.exp(beta*(-x+theta)))

    return ans
##############

"""
    parameters
"""
    
N = 2 # number of outputs

dt = 1
nsecs = 5000000 # length of inputs during learning
simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)

eps = 10**(-5)*0.5 #learning rate

tau =15 # membrane time constant
tau_syn = 5 # synaptic time constant
n_in = 500 # number of input neurons
PSP = np.zeros(n_in) #post synaptic potentials
I_syn = np.zeros(n_in) # synaptic current
g_L = 1/tau #leak conductance of soma
g_d = 0.7 #strength of dendrite to somatic interaction
w  = np.random.randn(n_in,N)/np.sqrt(n_in) #synaptic weights projecting to dendrite. This will be trained.
t0 = 9000 #time intervals for calcurating the mean and variance of activity.
gain = 10 #maximum firing rate of input neurons
noise=0.5 #strength of output noise in the learning rule. In the manuscript, we changed this in [0,0.9].
#############################
"""
mixing matrix
"""
q_cross = 0.5 # cross-talk noise
mixing_mat = np.zeros((n_in,N)) #  Here, the number of mixture is equal to the number of output neurons.
Q = np.zeros((N,N))
denom=np.sqrt(1+q_cross**2)
Q[0,:] = [1/denom,q_cross/denom]
Q[1,:] = [q_cross/denom,1/denom]

for i in range(n_in):
    mixing_mat[i,:] = Q[np.random.randint(N),:]
############################
"""
    fixed lateral inhibition.
"""
w_inh_max =0.4
w_inh =np.ones((N,N))*w_inh_max

w_inh[w_inh<0] = 0
w_inh[w_inh>w_inh_max] = w_inh_max

mask = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i!=j:
            mask[i,j]=1
w_inh *= mask
###############################

V_som_list=np.zeros((N,t0))

V_dend = np.zeros(N)
V_som = np.zeros(N)
f = np.zeros(N) #output firing rate. Here, this takes the value in [0,1].
########################
"""
    calculating min and max of true signals.
    """
freq = 0.4
time = np.arange(10000)
source = np.zeros((2,len(time)))

source[0,:] = (-np.sin(2*np.pi*time/1000*1.2*freq+100)-np.sin(2*np.pi*time/500*1.2*freq+100)*0.3)*2
source[1,:] = np.sin(2*np.pi*time/1000*freq+2000)+np.sin(2*np.pi*time/500*freq+2000)*2

rate_in = np.dot(mixing_mat,source)
rate_min = np.amin(rate_in,axis=1)
rate_max = np.amax(rate_in,axis=1)

print("")
print("***********")
print("Learning... ")
print("***********")
source=np.zeros(2) #true signals
for i in range(simtime_len):
    if int(i / simtime_len * 100) % 5 == 0.0:
        if int(i / simtime_len * 100) > int((i - 1) / simtime_len * 100):
            print(" " + str(int(i / simtime_len * 100)) + "% ")
    source[0] = (-np.sin(2*np.pi*i/1000*1.2*freq+100)-np.sin(2*np.pi*i/500*1.2*freq+100)*0.3)*2
    source[1] = np.sin(2.*np.pi*i/1000*freq+2000)+np.sin(2.*np.pi*i/500*freq+2000)*2

    rate_in = (np.dot(mixing_mat,source)-rate_min)/(rate_max-rate_min)*gain #normalized mixtures

    prate = dt*rate_in*(10**-3) #prob of spikes of input neurons

    id = np.random.rand(n_in)<prate #input neurons that spikes.
    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[id]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit=PSP*25 # PSPs are normalized.
    V_dend = np.dot(w.T,PSP_unit) #voltage of dendrite

    V_som_list = np.roll(V_som_list, -1,axis=1)

    V_som = (1.0-dt*g_L)*V_som +g_d*(V_dend-V_som)+np.dot(-w_inh,f) #voltage of  soma
    V_som_list[:,-1] = V_som

    if i>t0:
        f = np.clip(g((V_som-np.mean(V_som_list,axis=1)) / np.std(V_som_list,axis=1))+np.random.randn(N)*noise,0,1) #noisy firing rates are used during training.

        w += eps  *np.outer((f-g(V_dend*g_d/(g_d+g_L))) , PSP_unit).T*beta*(1-g(V_dend*g_d/(g_d+g_L))) #the main learning rule
        w-=eps*w*0.5

print("")
print("***********")
print("Testing... ")
print("***********")
test_len = 10000 #length of inputs for test.
loop = 20 #outpus are averaged over 20 trials.

f_list = np.zeros((N,test_len*loop))

out1_lists = np.zeros((loop,test_len))
out2_lists = np.zeros((loop,test_len))

for j in range(loop):
    PSP = np.zeros(n_in)
    I_syn = np.zeros(n_in)


    V_dend = np.zeros(N)

    V_som = np.zeros(N)
    id = np.zeros((test_len,n_in),dtype=bool)
    for i in range(test_len):
       
        source[0] = (-np.sin(2*np.pi*i/1000*1.2*freq+100)-np.sin(2*np.pi*i/500*1.2*freq+100)*0.3)*2
        source[1] = np.sin(2.*np.pi*i/1000*freq+2000)+np.sin(2.*np.pi*i/500*freq+2000)*2

        rate_in = rate_in = (np.dot(mixing_mat,source)-rate_min)/(rate_max-rate_min)*gain
   
        prate = dt*rate_in*(10**-3)
        
        id[i,:] = np.random.rand(n_in)<prate
        I_syn = (1.0 - dt / tau_syn) * I_syn
        I_syn[id[i,:]]+=1/tau/tau_syn
        PSP = (1.0 - dt / tau) * PSP + I_syn
        PSP_unit=PSP*25
        V_dend = np.dot(w.T,PSP_unit)
        
        V_som = (1.0-dt*g_L)*V_som +g_d*(V_dend-V_som)+np.dot(-w_inh,f)
 
        f = g(V_som)

        out1_lists[j,i]=f[0]
        out2_lists[j,i]=f[1]
"""
    plotting the mized, true, and output signals.
    """

true_sources=np.zeros((2,test_len))
true_sources[0,:]=(-np.sin(2*np.pi*np.arange(test_len)/1000*1.2*freq+100)-np.sin(2*np.pi*np.arange(test_len)/500*1.2*freq+100)*0.3)*2
true_sources[1,:]=np.sin(2.*np.pi*np.arange(test_len)/1000*freq+2000)+np.sin(2.*np.pi*np.arange(test_len)/500*freq+2000)*2

mixed_sources=np.dot(Q,true_sources)

fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax2 = fig.add_subplot(211)
pl.plot(true_sources[0,:],lw=1.5,c='orangered')

ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.xaxis.set_major_locator(pl.NullLocator())

ax3 = fig.add_subplot(212)
pl.plot(true_sources[1,:],lw=1.5,c='limegreen')

ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')

plt.xlabel("Time [ms]", fontsize=11)
ax.set_ylabel("Activity", fontsize=11)

fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95,hspace=0.3)

plt.savefig('true.pdf', fmt='pdf', dpi=350)

fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax2 = fig.add_subplot(211)
pl.plot(mixed_sources[0,:],lw=1.5,c='orangered')

ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.xaxis.set_major_locator(pl.NullLocator())

ax3 = fig.add_subplot(212)
pl.plot(mixed_sources[1,:],lw=1.5,c='limegreen')

ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')

plt.xlabel("Time [ms]", fontsize=11)
ax.set_ylabel("Activity", fontsize=11)

fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95,hspace=0.3)

plt.savefig('mixed.pdf', fmt='pdf', dpi=350)




fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax2 = fig.add_subplot(211)
pl.plot(np.mean(out1_lists,axis=0),lw=1.5,c='orangered')

ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.xaxis.set_major_locator(pl.NullLocator())

ax3 = fig.add_subplot(212)
pl.plot(np.mean(out2_lists,axis=0),lw=1.5,c='limegreen')

ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')

plt.xlabel("Time [ms]", fontsize=11)
ax.set_ylabel("Activity", fontsize=11)

fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95,hspace=0.3)

plt.savefig('outputs.pdf', fmt='pdf', dpi=350)
