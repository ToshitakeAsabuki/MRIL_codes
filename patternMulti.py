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


width = 50


mean_rate = 5
r_sig = mean_rate
r_noise = mean_rate - r_sig
gain=30
n_in=500
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

N = 10

nsecs = width *15000
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

eps = 10**(-6)#*0.5

p_connect = 1

w_inh_max =0.1#5/np.sqrt(N)

w_inh =np.ones((N,N))*w_inh_max
spike_time = np.zeros(N)

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

cA_p = -1*0.105*0.1#*0.3
cA_d = 0.525*0.1*0.1#*0.3
tau_p=20
tau_d=40
A_pre_p=np.zeros(N)
A_pre_d=np.zeros(N)
A_post_p=np.zeros(N)
A_post_d=np.zeros(N)
spike_mat=np.zeros((n_in,simtime_len),dtype=bool)

random_start=0



print("")
print("***********")
print("Learning... ")
print("***********")
for i in range(simtime_len):
    
    if int(i / simtime_len * 100) % 5 == 0.0:
        if int(i / simtime_len * 100) > int((i - 1) / simtime_len * 100):
            print(" " + str(int(i / simtime_len * 100)) + "% ")
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
    I_syn = (1.0 - dt / tau_syn) * I_syn
    I_syn[spike_mat]+=1/tau/tau_syn
    PSP = (1.0 - dt / tau) * PSP + I_syn
    PSP_unit=PSP*25
    V_dend = np.dot(w.T,PSP_unit)
    V_som_list = np.roll(V_som_list, -1,axis=1)

    V_som = (1.0-dt*g_L)*V_som +g_d*(V_dend-V_som)+np.dot(-w_inh,f)
    V_som_list[:,-1] = V_som

    if i>width*window:

        f = g((V_som-np.mean(V_som_list,axis=1)) / np.std(V_som_list,axis=1))
       
        w += eps  *np.outer((f-g(V_dend*g_d/(g_d+g_L))) , PSP_unit).T*beta*(1-g(V_dend*g_d/(g_d+g_L)))
        w-=eps*w*5

    A_pre_p = (1.0 - dt / tau_p) * A_pre_p
    A_pre_d = (1.0 - dt / tau_d) * A_pre_d
    A_post_p = (1.0 - dt / tau_p) * A_post_p
    A_post_d = (1.0 - dt / tau_d) * A_post_d

    for k in range(N):
        if np.random.rand()<dt*f[k]*gain*(10**-3):

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

test_len=60*width
plot_len=1500
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
        #m=0
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

    V_som = (1.0-dt*g_L)*V_som +g_d*(V_dend-V_som)+np.dot(-w_inh,f)
    for k in range(N):
        f[k] = g(V_som[k])

    f_list[:,i]=f


nspk_random,tspk_random = pl.nonzero(random_mat[0:200,:]==1)
nspk1,tspk1 = pl.nonzero(pat1_mat[0:200,:]==1)
nspk2,tspk2 = pl.nonzero(pat2_mat[0:200,:]==1)
nspk3,tspk3 = pl.nonzero(pat3_mat[0:200,:]==1)


fig = plt.figure(figsize=(7, 2))
ax = fig.add_subplot(111)
for i in pat1_start:
    plt.vlines([i], 0, n_in, "dodgerblue", linestyles='dashed',lw=1)
    if i+width<plot_len:
        plt.vlines([i+width], 0, n_in, "dodgerblue", linestyles='dashed',lw=1)
    plt.hlines([200+15], i, min(i+width,plot_len), "dodgerblue", linestyles='solid',lw=3)
for i in pat2_start:
    plt.vlines([i], 0, n_in, "orangered", linestyles='dashed',lw=1)
    if i+width<plot_len:
        plt.vlines([i+width], 0, n_in, "orangered", linestyles='dashed',lw=1)
    plt.hlines([200+15], i, min(i+width,plot_len), "orangered", linestyles='solid',lw=3)
for i in pat3_start:
    plt.vlines([i], 0, n_in, "limegreen", linestyles='dashed',lw=1)
    if i+width<plot_len:
        plt.vlines([i+width], 0, n_in, "limegreen", linestyles='dashed',lw=1)
    plt.hlines([200+15], i, min(i+width,plot_len), "limegreen", linestyles='solid',lw=3)
pl.plot(tspk_random,nspk_random,'k.',markersize=2)
pl.plot(tspk1,nspk1,'b.',markersize=2)
pl.plot(tspk2,nspk2,'r.',markersize=2)
pl.plot(tspk3,nspk3,'g.',markersize=2)
plt.ylabel("Neuron id", fontsize=11)

fig.subplots_adjust(bottom=0.25, left=0.1)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
pl.xlim([0,plot_len])
pl.ylim([0,200+20])
plt.savefig('raster.pdf', fmt='pdf', dpi=350)


sample_len = test_len

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
pl.xlim([0,plot_len])

fig.subplots_adjust(left=0.15,bottom=0.25,right=1)

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

target1=np.zeros(test_len)
target2=np.zeros(test_len)
target3=np.zeros(test_len)
for i in (pat1_start):
    target1[i:min(test_len,i+width)]=1
for i in (pat2_start):
    target2[i:min(test_len,i+width)]=1
for i in (pat3_start):
    target3[i:min(test_len,i+width)]=1

chunk_corr = np.zeros((N,3))

for i in range(N):
    chunk_corr[i,0]=np.corrcoef(f_list[i,:],target1)[0,1]
    chunk_corr[i,1]=np.corrcoef(f_list[i,:],target2)[0,1]
    chunk_corr[i,2]=np.corrcoef(f_list[i,:],target3)[0,1]
performance = 0
for i in range(N):
    performance+=max(chunk_corr[i,:])
performance/=N
print(performance)

fig = plt.figure(figsize=(7, 3))
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

ax1 = fig.add_subplot(311)
pl.plot(f_list[np.argmax(chunk_corr[:,0]),:],lw=1.5,c='k')
for i in (pat1_start):
    pl.axvspan(i, i + width, facecolor='dodgerblue', alpha=0.3,linewidth=0)

pl.xlim([0,plot_len])
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.spines['right'].set_color('none')
ax1.spines['top'].set_color('none')
#plt.yticks( np.arange(0, 1.1, 0.5) )
ax1.xaxis.set_major_locator(pl.NullLocator())

ax2 = fig.add_subplot(312)
pl.plot(f_list[np.argmax(chunk_corr[:,1]),:],lw=1.5,c='k')

for i in (pat2_start):
    pl.axvspan(i, i + width, facecolor='orangered', alpha=0.3,linewidth=0)

pl.xlim([0,plot_len])
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
#plt.yticks( np.arange(0, 1.1, 0.5) )
ax2.xaxis.set_major_locator(pl.NullLocator())

ax3 = fig.add_subplot(313)
pl.plot(f_list[np.argmax(chunk_corr[:,2]),:],lw=1.5,c='k')

for i in (pat3_start):
    pl.axvspan(i, i + width, facecolor='limegreen', alpha=0.3,linewidth=0)
pl.xlim([0,plot_len])
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')
#plt.yticks( np.arange(0, 1.1, 0.5) )
plt.xlabel("Time [ms]", fontsize=11)

fig.subplots_adjust(bottom=0.15, left=0.1)
plt.savefig('activities.pdf', fmt='pdf', dpi=350)

