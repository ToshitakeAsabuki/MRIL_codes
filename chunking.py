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
import sklearn.decomposition
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

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
    
    theta= 0.5

    ans = 1/(1+alpha*np.exp(beta*(-x+theta)))
    return ans

N = 10
width = 30
dt = 1

nsecs = width *10000
simtime = np.arange(0, nsecs, dt)
simtime_len = len(simtime)

tau =15
tau_syn = 5
n_syn = 1000
n_in =  n_syn
PSP = np.zeros(n_in)
I_syn = np.zeros(n_in)
g_L=1/tau
g_d=0.7
w  = np.random.randn(n_syn,N)/np.sqrt(n_syn)*1

poisson_signal =10
poisson_noise = 0.
eps = 10**(-4)#/poisson_signal

cA_p = -1*0.105#*0.1
cA_d = 0.525*0.1#*0.1
tau_p=20
tau_d=40
A_pre_p=np.zeros(N)
A_pre_d=np.zeros(N)
A_post_p=np.zeros(N)
A_post_d=np.zeros(N)

p_connect = 1
max_rate = poisson_signal
w_inh_max = 0.1#5/np.sqrt(N)#/max_rate

spike_time = np.zeros(N)

w_inh =np.ones((N,N))*w_inh_max

mask = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i!=j:
            if np.random.rand()<p_connect:
                mask[i,j]=1
w_inh*=mask
w_inh[w_inh<0] = 0
w_inh[w_inh>w_inh_max] = w_inh_max



window = 300
V_som_list=np.zeros((N,window*width))

V_dend = np.zeros(N)

V_som = np.zeros(N)

connection_list = np.zeros((N,n_in),dtype=bool)
for i in range(N):
    connection_list[i,np.random.choice(np.arange(n_in), n_syn, replace = False)]=1

f = np.zeros(N)


chunk_list = [['a', 'b', 'c', 'd'],['e', 'f', 'g', 'h'],['i', 'j', 'k', 'l']]
n_chunk = len(chunk_list)


sym_list = ['a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l']

chunk = chunk_list[np.random.randint(n_chunk)]

sample_num=10
sample_len = width*len(chunk_list[0])*sample_num
test_len=sample_len

m = 0

input_pref = np.zeros(n_in)

for i in range(n_in):
    input_pref[i] = np.random.randint(len(sym_list))

PSP_mat = np.zeros((N,n_syn))
symbol_pat=np.zeros((n_in,len(sym_list)),dtype=bool)
for i in range(n_in):
    symbol_pat[i,np.random.choice(np.arange(len(sym_list)), 1, replace = False)]=1
print("")
print("***********")
print("Learning... ")
print("***********")

for i in range(simtime_len):
    if (i % (width/dt) == 0 and i > 0):

        if m == len(chunk) - 1:
            chunk = chunk_list[np.random.randint(n_chunk)]
        
            m = 0
        
        else:
            
            m += 1

    rate_in = np.ones(n_in)*poisson_noise

    input_id =symbol_pat[:,sym_list.index(chunk[m])]
    
    rate_in[input_id] = poisson_signal
    prate = dt*rate_in*(10**-3)
    
    if int(i / simtime_len * 100) % 5 == 0.0:
        if int(i / simtime_len * 100) > int((i - 1) / simtime_len * 100):
            print(" " + str(int(i / simtime_len * 100)) + "% ")

    id = np.random.rand(n_in)<prate
    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[id]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit=PSP*25
    
    for k in range(N):
        PSP_mat[k,:] = PSP_unit[connection_list[k]]
        
    V_dend = np.diag(np.dot(PSP_mat,w))

    V_som_list = np.roll(V_som_list, -1,axis=1)
    for k in range(N):
        V_som[k] = (1.0-dt/tau)*V_som[k] +g_d*(V_dend[k]-V_som[k])+np.dot(-w_inh[k,:],f)

    V_som_list[:,-1] = V_som
    
    if i>width*window:
        
        for k in range(N):
            f[k]=g((V_som[k]-np.mean(V_som_list[k,:])) / np.std(V_som_list[k,:]))#*max_rate

   
            w[:,k] += eps  *(f[k]-g(V_dend[k]*g_d/(g_d+g_L))*1) * PSP_unit[connection_list[k]]*beta*(1-g(V_dend[k]*g_d/(g_d+g_L)))

        w-=eps*w*0.05
    A_pre_p = (1.0 - dt / tau_p) * A_pre_p
    A_pre_d = (1.0 - dt / tau_d) * A_pre_d
    A_post_p = (1.0 - dt / tau_p) * A_post_p
    A_post_d = (1.0 - dt / tau_d) * A_post_d

    for k in range(N):
        if np.random.rand()<dt*f[k]*max_rate*(10**-3):
     
                A_pre_p[k]+=cA_p
                A_pre_d[k]+=cA_d
                A_post_p[k]+=cA_p
                A_post_d[k]+=cA_d
                
                w_inh[k,:]+=(A_pre_p+A_pre_d)*w_inh_max
                w_inh[:,k]+=(A_post_p+A_post_d)*w_inh_max
                spike_time[k]=i

    w_inh*=mask
    w_inh[w_inh<0] = 0
    w_inh[w_inh>w_inh_max] = w_inh_max


print("")
print("***********")
print("Testing... ")
print("***********")

PSP = np.zeros(n_in)
I_syn = np.zeros(n_in)
synaptic_input_matrix=np.zeros((n_in*N,test_len))

V_dend_list =np.zeros((N,test_len))

V_dend = np.zeros(N)

V_som = np.zeros(N)
chunk_id = np.random.randint(n_chunk)
chunk = chunk_list[chunk_id]

m = 0

f_list = np.zeros((N,test_len))

chunk_start = [[] for k in range(n_chunk)]

chunk_start[chunk_id].append(0)

id = np.zeros((test_len,n_in),dtype=bool)

for i in range(test_len):
    if (i % (width/dt) == 0 and i > 0):
        
        if m == len(chunk) - 1:
            chunk_id = np.random.randint(n_chunk)
            chunk = chunk_list[chunk_id]
            chunk_start[chunk_id].append(i)
            m = 0
        
        else:
            
            m += 1

    rate_in = np.ones(n_in)*poisson_noise

    input_id =symbol_pat[:,sym_list.index(chunk[m])]
    rate_in[input_id] = poisson_signal
    prate = dt*rate_in*(10**-3)
    

    id[i,:] = np.random.rand(n_in)<prate

    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[id[i,:]]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit=PSP*25
    for l in range(N):
        synaptic_input_matrix[l*n_in:(l+1)*n_in,i]=PSP*w[:,l]
    for k in range(N):
        PSP_mat[k,:] = PSP_unit[connection_list[k]]
        
    V_dend = np.diag(np.dot(PSP_mat,w))
    
    for k in range(N):
        V_som[k] = (1.0-dt/tau)*V_som[k] +g_d*(V_dend[k]-1*V_som[k])+np.dot(-w_inh[k,:],f)
    V_dend_list[:,i] = V_dend
    for k in range(N):
        f[k] = g(V_som[k])#*max_rate

    f_list[:,i]=f#/max_rate

chunk1_start=np.array(chunk_start[0])
chunk2_start=np.array(chunk_start[1])
chunk3_start=np.array(chunk_start[2])
tspk,nspk = pl.nonzero(id==1)



###################
##
##  Plotting
##
###################



max1 = np.zeros(N)
min1 = np.zeros(N)
for i in range(N):
    max1[i] = np.max(f_list[i,0:sample_len])
    min1[i] = np.min(f_list[i,0:sample_len])
avg_norm1 = np.zeros((N,sample_len))

for i in range(N):
    avg_norm1[i,:] = (f_list[i,0:sample_len]-min1[i])/(max1[i]-min1[i])

t = np.zeros(N)
for j in range(N):
    arg = np.angle(np.dot(avg_norm1[j,:],np.exp(np.arange(sample_len)/(sample_len)*2*np.pi*1j))/sum(avg_norm1[j,:]))
    if arg<0:
        arg += 2*np.pi
    t[j] = sample_len/(2*np.pi)*arg

index = np.zeros(N)

index = np.argsort(t)
avg_sorted = np.zeros((N,sample_len))
for i in range(N):
    avg_sorted[i,:] = avg_norm1[int(index[i]),:]

fig, ax = plt.subplots(figsize=(4,3))

cax=plt.imshow(avg_sorted, interpolation='nearest', aspect="auto",cmap='jet')

cbar = fig.colorbar(cax, ticks=[0, 1], orientation='vertical')
cbar.ax.set_yticklabels(['min', 'max'],fontsize=10)

plt.xlabel("Time [ms]",fontsize=10)
plt.ylabel("Neurons (sorted)",fontsize=10)
plt.yticks([0,N-1],["1","%d"%N],fontsize=10)
ax.tick_params(length=1.3, width=0.05, labelsize=10)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
plt.ylim([-0.5,N-0.5])
pl.xlim([0,sample_len])

fig.subplots_adjust(left=0.15,bottom=0.25,right=1)
for l in range(sample_num):
    ax.axvline(x=width*len(chunk)*(l+1), ymin=0, ymax=N, color='gray', linewidth=1)

plt.savefig('activity_map.pdf', fmt='pdf',dpi=350)


weight_sorted_row =np.zeros((N,N))
weight_sorted_column =np.zeros((N,N))
for i in range(N):
    weight_sorted_row[i,:] = w_inh[int(index[i]),:]
for i in range(N):
    weight_sorted_column[:,i] =weight_sorted_row[:,int(index[i])]
fig, ax = plt.subplots(figsize=(4,3))

cax=plt.imshow(weight_sorted_column, interpolation='nearest', aspect="auto",cmap='jet')

cbar = fig.colorbar(cax, orientation='vertical')

plt.xlabel("Presynaptic neuron",fontsize=10)
plt.ylabel("Postsynaptic neuron",fontsize=10)

ax.tick_params(length=1.3, width=0.05, labelsize=11)
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
plt.xticks([0,1,2,3,4,5,6,7,8,9],['1','2','3','4','5','6','7','8','9','10'],fontsize=11)
plt.yticks([0,1,2,3,4,5,6,7,8,9],['1','2','3','4','5','6','7','8','9','10'],fontsize=11)
fig.subplots_adjust(left=0.15,bottom=0.25,right=0.8)

plt.savefig('Winh_map.pdf', fmt='pdf',dpi=350)



fig = plt.figure(figsize=(8, 3))
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
ax2 = fig.add_subplot(311)
pl.plot(f_list[int(index[0]),:],lw=1.5,c='k')
for i in (chunk1_start):
    
    pl.axvspan(i, i + width*4 , facecolor='orangered', alpha=0.3,linewidth=0)
for i in (chunk2_start):
    
    pl.axvspan(i, i + width*4 , facecolor='limegreen', alpha=0.3,linewidth=0)
for i in (chunk3_start):
    
    pl.axvspan(i, i + width*4 , facecolor='dodgerblue', alpha=0.3,linewidth=0)
pl.xlim([0,test_len])
pl.ylim([-0.01,1.01])
plt.yticks( np.arange(0,1.01 , 0.5) )
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.xaxis.set_major_locator(pl.NullLocator())
ax2 = fig.add_subplot(312)
pl.plot(f_list[int(index[4]),:],lw=1.5,c='k')
for i in (chunk1_start):
    
    pl.axvspan(i, i + width*4 , facecolor='orangered', alpha=0.3,linewidth=0)
for i in (chunk2_start):
    
    pl.axvspan(i, i + width*4 , facecolor='limegreen', alpha=0.3,linewidth=0)
for i in (chunk3_start):
    
    pl.axvspan(i, i + width*4 , facecolor='dodgerblue', alpha=0.3,linewidth=0)
pl.xlim([0,test_len])
pl.ylim([-0.01,1.01])
plt.yticks( np.arange(0,1.01 , 0.5) )
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.xaxis.set_major_locator(pl.NullLocator())
ax3 = fig.add_subplot(313)
pl.plot(f_list[int(index[9]),:],lw=1.5,c='k')
for i in (chunk1_start):
    
    pl.axvspan(i, i + width*4 , facecolor='orangered', alpha=0.3,linewidth=0)
for i in (chunk2_start):
    
    pl.axvspan(i, i + width*4 , facecolor='limegreen', alpha=0.3,linewidth=0)
for i in (chunk3_start):
    
    pl.axvspan(i, i + width*4 , facecolor='dodgerblue', alpha=0.3,linewidth=0)
pl.xlim([0,test_len])
pl.ylim([-0.01,1.01])
plt.yticks( np.arange(0,1.01 , 0.5) )
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')

plt.xlabel("Time [ms]", fontsize=11)
ax.set_ylabel("Activity", fontsize=11)

fig.subplots_adjust(bottom=0.15, left=0.1, right=0.95,hspace=0.3)

plt.savefig('activities.pdf', fmt='pdf', dpi=350)


