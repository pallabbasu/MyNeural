#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec






def plot_losses(losses, w, save_name=None):
    
    up = -1 #3500
    
    
    m = np.ones(w)/w    
    avg_loss = np.convolve(losses,m,'valid')

    
    plt.figure(22, figsize=(13,10))
    plt.clf()
       
    plt.plot(avg_loss[:up])
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.grid()

 
    if save_name:
        plt.savefig(save_name)



def plot_losses2(losses0,losses1, w, save_name=None):   
    
    m = np.ones(w)/w    
    avg_loss = np.convolve(losses0,m,'valid')
    avg_loss1 = np.convolve(losses1,m,'valid')
    
    plt.figure(23, figsize=(13,10))
    plt.clf()
       
    up = -1 #len(avg_loss1)
    
    plt.subplot(211)
    plt.plot(avg_loss[:up], label='L1')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.grid()
   
    plt.subplot(212)
    plt.plot(avg_loss1[:up], label='L2')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid()

#    plt.legend()

 
    if save_name:
        plt.savefig(save_name)


def plot_losses21(losses0,losses1, w, save_name=None):   
    
    m = np.ones(w)/w    
    avg_loss = np.convolve(losses0,m,'valid')
    avg_loss1 = np.convolve(losses1,m,'valid')
    
    plt.figure(23, figsize=(13,10))
    plt.clf()
       
    up = -1 #len(avg_loss1)
    
    plt.plot(avg_loss[:up], label='L1')
    plt.plot(avg_loss1[:up], label='L2')

    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.grid()

#    plt.legend()

 
    if save_name:
        plt.savefig(save_name)






def plot_losses22(losses0,losses1, losses2, w, save_name=None):   
    
    m = np.ones(w)/w    
    avg_loss = np.convolve(losses0,m,'valid')
    avg_loss1 = np.convolve(losses1,m,'valid')
    avg_loss2 = np.convolve(losses2,m,'valid')
    
    plt.figure(23, figsize=(13,10))
    plt.clf()
       
    up = -1 #len(avg_loss1)
    
    plt.subplot(211)
    plt.plot(avg_loss[:up], label='L1')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.grid()
   
    plt.subplot(212)
    plt.plot(avg_loss1[:up], label='Acc')
    plt.plot(avg_loss2[:up], label='Max_p')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid()

    plt.legend()

 
    if save_name:
        plt.savefig(save_name)


def plot_losses23(losses0,losses1, losses2, losses3, w, save_name=None):   
    
    m = np.ones(w)/w    
    avg_loss = np.convolve(losses0,m,'valid')
    avg_loss1 = np.convolve(losses1,m,'valid')
    avg_loss2 = np.convolve(losses2,m,'valid')
    avg_loss3 = np.convolve(losses3,m,'valid')
    
    plt.figure(23, figsize=(13,10))
    plt.clf()
       
    up = -1 #len(avg_loss1)
    
    plt.subplot(211)
    plt.plot(avg_loss[:up], label='L1')
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    plt.grid()
   
    plt.subplot(212)
    plt.plot(avg_loss1[:up], label='Acc')
    plt.plot(avg_loss2[:up], label='Max_p')
    plt.plot(avg_loss3[:up], label='Min_p')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid()

    plt.legend()

 
    if save_name:
        plt.savefig(save_name)

    

# seed = 0
# sep = 8

# for seed in range(100,120):
#     plot_raw_denoiser(model_rd2, data_generator_raw, seed, model_name, num_spikes, sep, train=True)


def plot_raw_denoiser(model_rd, data_generator_raw, seed, model_name, num_spikes, sep, train=False):


    raw_color = 'grey'
    spikes_color = 'firebrick'
    
    out  = data_generator_raw.generate(batch_size=4, seed =seed, num_spikes=num_spikes, train=train, separation=sep)            
    raw_data = out['raw_data'] 
    clean_data = out['clean_data']
    templates = out['dil_templates']
    colls = out['dil_coll']
    locs = out['locations']
        
    denoised_data, data_sigma = model_rd.predict(raw_data)

    rr = raw_data.cpu().numpy()
    cc = clean_data.cpu().numpy()
    tt = templates.cpu().numpy()
    ll = colls.cpu().numpy()
    dd = denoised_data.cpu().numpy()    

    if num_spikes ==1:
        ideal_raw_data = out['ideal_raw_data'] 
        irr = ideal_raw_data.cpu().numpy()
    
    t = 121
    
    rows = 1
    ii = [1,0,2,3]
    plt.figure(9)
    
    plt.clf()
    for i in range(rows):
        
        plt.subplot(rows,1,i+1)
        
        j = ii[i]
        
        
        if num_spikes ==2:
            x_temp=range(locs[j,0],locs[j,0]+t)            
            plt.plot(x_temp,tt[j,0,:],spikes_color, linestyle = ':', label='Spike 1,2')        
            x_temp=range(locs[j,1],locs[j,1]+t)
            plt.plot(x_temp,ll[j,0,:],spikes_color, linestyle= ':')
            
            
            plt.plot(cc[j,0,:],'b', label='Spike 1 + 2')    
    
        
        
        if num_spikes ==1:
            plt.plot(irr[j,0,:],'b', label='Ideal noisy signal')       

             
        plt.plot(rr[j,0,:],raw_color, label='Observed signal')           
        
        plt.plot(dd[j,0,:],'k', label='Denoised signal')
    
    
        if i == 0:
            plt.legend(facecolor='white', framealpha=1, loc='lower right')
#            plt.legend(loc='right')


        if i==rows-1:
            plt.xlabel('Time (ms)')
        axes = plt.gca()        
        axes.set_xticklabels([str(t) for t in range(-2,15,2)])
        
    save_name = './figures/raw_denoiser/' + model_name + '_'  + str(seed) + '.pdf'     
    plt.savefig(save_name, bbox_inches="tight")            



# sep = 8
# for seed in range(20):
#     plot_timing(data_generator_raw, model_rd2, model_timer, seed, sep)



def plot_timing(data_generator_raw, model_rd2, model_timer, seed, separation):


    
    
    out  = data_generator_raw.generate(batch_size=15, seed =seed, num_spikes=2, separation = separation,train=True)            
    raw_data = out['raw_data'] 
    templates = out['dil_templates']
    colls = out['dil_coll']
    locations = out['locations']

    
    denoised_data, data_sigma = model_rd2.predict(raw_data)
    template_locations = locations[:,0]
    _, mm,lp = model_timer(denoised_data, template_locations)


    t=121
    tm = locations[:,0]
    tc = locations[:,1]
    
    cd = denoised_data.cpu().numpy()
    rr = raw_data.cpu().numpy()
    tt = templates.cpu().numpy()
    cc = colls.cpu().numpy()
    
    
    mconv = np.zeros(15,dtype=np.int64)
    convs= np.zeros([15,30],dtype=np.int64)
    
    for i in range(15):
        mconv[i] = np.correlate(rr[i,0,:],tt[i,0,:],'valid').argmax()
        convs[i,:] = np.correlate(rr[i,0,:],tt[i,0,:],'valid')    
   
    
    ii = [i for i in range(15) if (tm[i]==mm[i] and tm[i]!=mconv[i]) ]

    for i in range(len(ii)):
        
        plt.figure(10)
        plt.clf()
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
        
        
        j = ii[i]
        
        ax0 = plt.subplot(gs[0,0])    
        ax0.plot(rr[j,0,:],linestyle='-', color= 'silver', label='Observed signal')
        ax0.plot(cd[j,0,:],'k', label='Denoised signal')

        
        
        # x_temp=range(mconv[j],mconv[j] + t)
        # ax0.plot(x_temp,tt[j,0,:],'forestgreen',  label='CMP Timing')
        
        loc_temp = template_locations[j]
        x_temp=range(loc_temp,loc_temp+t)
        ax0.plot(x_temp,tt[j,0,:],'firebrick', label='Spike 1')
        
        
        loc_coll = locations[j,1]
        x_temp=range(loc_coll,loc_coll+t)
        ax0.plot(x_temp,cc[j,0,:],':', color = 'firebrick', label='Spike 2')
        
        ax0.set_xticklabels([str(t) for t in range(-2,15,2)])        

        plt.legend(facecolor='white', framealpha=1, loc='lower right')
        plt.xlabel('Time (ms)')
        
        #############
        
        
        
        ax1 = plt.subplot(gs[0,1])    
        
        
        y1 = convs[j,:]
        y1= (y1-y1.min())/y1.ptp()
        
        y2 = lp[j,:]
        y2= (y2-y2.min())/y2.ptp()
        
        ax1.plot(y2,'firebrick', label='NMP')
        ax1.plot(y1,'forestgreen', label='CMP')        
        ax1.axvline(x=tm[j], ymin=0, ymax=1,linestyle='-', linewidth=1, color='gray')
        ax1.axvline(x=tc[j], ymin=0, ymax=1,linestyle=':', linewidth=1, color='gray')
    
        ax1.set_xticks([0,5,10,15,20,25,29])
    
        plt.xlabel('Time shift')
        plt.legend(loc='lower right',facecolor='white', framealpha=1)
        
        
    
        save_name = './figures/timing/timing_seed_' + str(seed) + '_i_' + str(i) + '_sep_' + str(separation) +'.pdf'
        plt.savefig(save_name, bbox_inches="tight")            
    
    
    
