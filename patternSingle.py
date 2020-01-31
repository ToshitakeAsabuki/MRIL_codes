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
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pylab as pl
import matplotlib as mpl
import shutil
import copy


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

beta = 5
def g(x):
    
    alpha = 1
    
    theta=1.7
    
    ans = 1/(1+alpha*np.exp(beta*(-x+theta)))
    return ans



width = 50

mean_rate = 5
r_sig = mean_rate
r_noise = mean_rate - r_sig
gain=10
n_in=2000
dt=1

pat1 = np.zeros((n_in,width),dtype=bool)
pat2 = np.zeros((n_in,width),dtype=bool)
pat3 = np.zeros((n_in,width),dtype=bool)

for i in range(n_in):
    for j in range(width):
        if np.random.rand()<r_sig*dt*10**(-3):
            pat1[i,j]=1
        if np.random.rand()<r_sig*dt*10**(-3):
            pat2[i,j]=1
        if np.random.rand()<r_sig*dt*10**(-3):
            pat3[i,j]=1

N = 1

nsecs = width *10000
simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)

tau =15
tau_syn = 5
n_syn = n_in

PSP = np.zeros(n_in)
I_syn = np.zeros(n_in)
g_L=1/tau
g_d=0.7
w  = np.random.randn(n_syn,N)/np.sqrt(n_syn)

eps = 10**(-6)

p_connect = 1

spike_time = np.zeros(N)

window = 300
V_som_list=np.zeros((N,window*width))

V_dend = np.zeros(N)

V_som = np.zeros(N)

connection_list = np.zeros((N,n_in),dtype=bool)
for i in range(N):
    connection_list[i,np.random.choice(np.arange(n_in), n_syn, replace = False)]=1

f = np.zeros(N)

random_start=0


type='random'

print("")
print("***********")
print("Learning... ")
print("***********")
for i in range(simtime_len):
    if i == random_start:
        type='random'

        random_width=np.random.randint(1*width,3*width)
        pat_start=i+random_width

    if i==pat_start:

        random_start=i+width
        dice=np.random.rand()

        if dice<1/3:

            type='pat1'
        
        elif dice<2/3:

            type='pat2'

        else:

            type='pat3'
    if type=='random':
        spike_mat = np.zeros(n_in,dtype=bool)
        spike_mat[np.random.rand(n_in)<mean_rate*dt*10**(-3)]=1
    elif type=='pat1':
        spike_mat =pat1[:,i-pat_start]
    elif type=='pat2':
        spike_mat =pat2[:,i-pat_start]

    elif type=='pat3':
        spike_mat =pat3[:,i-pat_start]
        
    if int(i / simtime_len * 100) % 5 == 0.0:
        if int(i / simtime_len * 100) > int((i - 1) / simtime_len * 100):
            print(" " + str(int(i / simtime_len * 100)) + "% ")
    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[spike_mat]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit = PSP*25
    V_dend = np.dot(w.T,PSP_unit)
    V_som_list = np.roll(V_som_list, -1,axis=1)

    V_som = (1.0-dt*g_L)*V_som +g_d*(V_dend-V_som)
    V_som_list[:,-1] = V_som

    if i>width*window:

        f = g((V_som-np.mean(V_som_list,axis=1)) / np.std(V_som_list,axis=1))#*gain

        w += eps  *np.outer((f-g(V_dend*g_d/(g_d+g_L))) , PSP_unit).T*beta*(1-g(V_dend*g_d/(g_d+g_L)))
        w -= eps*w*5


print("")
print("***********")
print("Testing... ")
print("***********")

test_len=60*width*5
plot_len=700
synaptic_input_matrix=np.zeros((n_in*N,test_len))
spike_mat=np.zeros((n_in,test_len),dtype=bool)
PSP = np.zeros(n_in)
I_syn = np.zeros(n_in)
synaptic_input_matrix=np.zeros((n_in*N,test_len))

V_dend_list =np.zeros((N,test_len))

V_dend = np.zeros(N)

V_som = np.zeros(N)
f_list = np.zeros((N,test_len))
pat1_start=[]
pat2_start=[]
pat3_start=[]
random_start=0

random_mat=np.zeros((n_in,test_len))
pat1_mat=np.zeros((n_in,test_len))
pat2_mat=np.zeros((n_in,test_len))
pat3_mat=np.zeros((n_in,test_len))
for i in range(test_len):
    if i == random_start:

        random_width=np.random.randint(1*width,3*width)
        pat_start=i+random_width
        spike_mat[:,i:min(i+random_width,test_len)] = np.zeros((n_in,min(test_len-i,random_width)),dtype=bool)
        for j in range(n_in):
            for k in range(min(random_width,test_len-i)):
                if np.random.rand()<mean_rate*dt*10**(-3):
                    spike_mat[j,k+i]=1

        random_mat[:,i:min(i+random_width,test_len)]=spike_mat[:,i:min(i+random_width,test_len)]
    if i==pat_start:

        random_start=i+width
        p_pat1=1/3
        p_pat2=2/3

        if pat1_start==[]:
            p_pat1=1
        if pat2_start==[]:
            p_pat1=0
            p_pat2=1
        if pat3_start==[]:
            p_pat1=0
            p_pat2=0
        dice=np.random.rand()
        if dice<p_pat1:
            pat1_start.append(i)
            spike_mat[:,i:min(i+width,test_len)]=pat1[:,0:min(test_len-i,width)]
            
        elif dice<p_pat2:
            pat2_start.append(i)
            spike_mat[:,i:min(i+width,test_len)]=pat2[:,0:min(test_len-i,width)]

        else:
            pat3_start.append(i)
            
            spike_mat[:,i:min(i+width,test_len)]=pat3[:,0:min(width,test_len-i)]

for i in range(test_len):
    if i in pat1_start:
        pat1_mat[:,i:min(i+width,test_len)]=spike_mat[:,i:min(i+width,test_len)]
    if i in pat2_start:
        pat2_mat[:,i:min(i+width,test_len)]=spike_mat[:,i:min(i+width,test_len)]
    if i in pat3_start:
        pat3_mat[:,i:min(i+width,test_len)]=spike_mat[:,i:min(i+width,test_len)]
for i in range(test_len):
    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[spike_mat[:,i]]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit=PSP*25
    for l in range(N):
        synaptic_input_matrix[l*n_in:(l+1)*n_in,i]=PSP_unit*w[:,l]
    V_dend = np.dot(w.T,PSP_unit)

    V_som = (1.0-dt*g_L)*V_som +g_d*(V_dend-V_som)
    for k in range(N):
        f[k] = g(V_som[k])

    f_list[:,i]=f


nspk_random,tspk_random = pl.nonzero(random_mat==1)
nspk1,tspk1 = pl.nonzero(pat1_mat==1)
nspk2,tspk2 = pl.nonzero(pat2_mat==1)
nspk3,tspk3 = pl.nonzero(pat3_mat==1)



fig = plt.figure(figsize=(7, 2))
ax = plt.subplot(1, 1, 1)
for i in range(N):
    pl.plot(f_list[i,:],lw=1.5,c='k')


for i in (pat1_start):
    
    pl.axvspan(i, i + width, facecolor='orangered', alpha=0.3,linewidth=0)
for i in (pat2_start):
    
    pl.axvspan(i, i + width, facecolor='dodgerblue', alpha=0.3,linewidth=0)
for i in (pat3_start):
    
    pl.axvspan(i, i + width, facecolor='limegreen', alpha=0.3,linewidth=0)

pl.xlim([0,1400])
pl.ylim([-0.1,1.1])
plt.xlabel("Time [ms]", fontsize=11)
plt.ylabel("Activity", fontsize=11)

fig.subplots_adjust(bottom=0.25, left=0.1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

plt.savefig('activity.pdf', fmt='pdf', dpi=350)



target1=np.zeros(test_len)
target2=np.zeros(test_len)
target3=np.zeros(test_len)
for i in (pat1_start):
    target1[i:min(test_len,i+width)]=1
for i in (pat2_start):
    target2[i:min(test_len,i+width)]=1
for i in (pat3_start):
    target3[i:min(test_len,i+width)]=1

chunk_corr = np.zeros(3)

for i in range(N):
    chunk_corr[0]=np.corrcoef(f_list[i,:],target1)[0,1]
    chunk_corr[1]=np.corrcoef(f_list[i,:],target2)[0,1]
    chunk_corr[2]=np.corrcoef(f_list[i,:],target3)[0,1]

fig = plt.figure(figsize=(7, 2))
ax = fig.add_subplot(111)
for i in pat1_start:
    plt.vlines([i], 0, 2000+150, "orangered", linestyles='dashed',lw=1)
    if i+width<plot_len:
        plt.vlines([i+width], 0, 2000+150, "orangered", linestyles='dashed',lw=1)
    plt.hlines([2000+150], i, min(i+width,plot_len), "orangered", linestyles='solid',lw=5)
for i in pat2_start:
    plt.vlines([i], 0, 2000+150, "dodgerblue", linestyles='dashed',lw=1)
    if i+width<plot_len:
        plt.vlines([i+width], 0, 2000+150, "dodgerblue", linestyles='dashed',lw=1)
    plt.hlines([2000+150], i, min(i+width,plot_len), "dodgerblue", linestyles='solid',lw=5)
for i in pat3_start:
    plt.vlines([i], 0, 2000+150, "limegreen", linestyles='dashed',lw=1)
    if i+width<plot_len:
        plt.vlines([i+width], 0, 2000+150, "limegreen", linestyles='dashed',lw=1)
    plt.hlines([2000+150], i, min(i+width,plot_len), "limegreen", linestyles='solid',lw=5)
pl.plot(tspk_random,nspk_random,c='k',marker='.',lw=0,markersize=1.5)
pl.plot(tspk1,nspk1,c='orangered',marker='.',lw=0,markersize=1.5)
pl.plot(tspk2,nspk2,c='dodgerblue',marker='.',lw=0,markersize=1.5)
pl.plot(tspk3,nspk3,c='limegreen',marker='.',lw=0,markersize=1.5)
plt.ylabel("Neuron id", fontsize=11)

fig.subplots_adjust(bottom=0.25, left=0.1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
pl.xlim([0,plot_len])
pl.ylim([0,2000+150])
plt.savefig('raster.pdf', fmt='pdf', dpi=350)




