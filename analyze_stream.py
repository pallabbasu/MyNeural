#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import torch
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from noise_generator import Noise_Generator
from config import params





from models.matcher_stream_np_noisy2 import matcher
model2 = matcher(params).to(params['device'])
model_checkpoint = './saved_models/matcher_stream_np_noisy2_21.pt'
checkpoint = torch.load(model_checkpoint)    
model2.load_state_dict(checkpoint['model_state_dict'])
model2 = model2.to(params['device'])




data_dir = './data/'
fname_templates = data_dir + 'templates_np.npy'
all_templates = np.load(fname_templates)
num_templates = all_templates.shape[0]
tl = all_templates.shape[1]

noise_generator = Noise_Generator()





# load and prepare stream data

data_set = 'data_raw_30000hz_300sec_5hz_to_25hz'


fname_times = data_dir + data_set  + '_times.npy'
times= np.load(fname_times, allow_pickle = True)



# make dictionart td[spike_time] = num_of_template

td = {}
for i in range(num_templates):
    ti = times[i]
    for j in ti:
        td[j] = i

td_keys = set(td.keys())
all_spike_times = np.sort(list(td.keys()))


num_events = len(all_spike_times)







def generate_stream(T0,T1, seed=None, noise=True):

    if seed is not None:
        np.random.seed(seed)


    truth = np.where((all_spike_times >= T0) & (all_spike_times <= T1) )[0]    
    lt = len(truth)    
    
    true_ar = np.zeros([lt,2],dtype=np.int32)            #[time locations, template ids]
    true_ar[:,0] = all_spike_times[truth]
    
    stream_segment = np.zeros(T1-T0+tl)
    
    pt = np.random.choice(num_templates, size=num_templates,replace=False)     
    
    templates = all_templates[pt,:]
    dilations = 1#.95 +.1*np.random.rand(num_templates,1) 
    templates *= dilations
    
    for i in range(lt):
        
        ti = all_spike_times[truth[i]]
        k =  pt[td[ti]]
        true_ar[i,1] = k
        stream_segment[ti-T0:ti-T0+tl] += templates[k,:]



    if noise:
        stream_segment += noise_generator.generate(T1-T0+tl)
    
    return stream_segment, true_ar, templates 







exclude = []


colors = ['blue','red','green',
'cyan',  'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink']






    


def plot_nmp_cmp(T0,T1,exclude = [], seed=None, fignum=0):    
    
    col_count = 0
    colors_dict = {}    
    

    T = T1-T0    

    stream_segment, true_ar, templates  = generate_stream(T0,T1,seed,noise)

    stream= stream_segment.copy()    
    #ptps = templates.ptp(1)
    
    exclude = list(set(range(num_templates)).difference(set(true_ar[:,1])))
    nmp_ar, nmp_probs, _ = get_nmp(stream_segment, templates, model, threshold, exclude, c0_max)    

    c0 = nmp_ar.shape[0]
    

    print('\n')    
    T = T1-T0    
    stream= stream_segment.copy()    
    #denoised_stream  = denoise_stream(stream, denoiser)    
    
    plt.figure(fignum,figsize=(19,9.6))
    plt.clf()

    

    for i in range(1,3):
        plt.subplot(c0+1,2,i)
        if i==1:
            plt.plot(stream, '-', color='grey',  label='Observed')
        else:
            plt.plot(stream, '-', color='grey',  label='Observed')
            
        for x,ind in zip(true_ar[:,0],true_ar[:,1]) :            
            if i==1:
                print(x,ind)
            xspike = range(x-T0, x-T0+tl)

            if ind not in colors_dict:            
                colors_dict[ind] = colors[col_count]
                col_count +=1
                
            color = colors_dict[ind]
                
                
                
            plt.plot(xspike,templates[ind,:], '-', color = color)
    
        #plt.legend()
    
        ax = plt.gca()    
        ax.set_xticks([])
        ax.set_xticklabels([])
        # ax.set_yticks([])
        # ax.set_yticklabels([])
        if i ==1:
            plt.title('NMP')
        else:
            plt.title('CMP')        


    print('\n')
    c1=0    

    while True:
                
        tte = nmp_ar[c1,1]
        loc = nmp_ar[c1,0] -T0

        print(T0+loc, tte, '{:.2f}'.format(templates[tte,:].ptp()), nmp_probs[c1])


        plt.subplot(c0+1,2,2*c1+3)    
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_yticklabels([])


        plt.plot(stream, 'silver')
        plt.ylabel('{0:.2f}'.format(nmp_probs[c1]))        
        
        if c1==0:
            axes = plt.gca()        
            ylims= axes.get_ylim()

        else:
            plt.ylim(ylims)
                

        if tte not in colors_dict:            
            colors_dict[tte] = colors[col_count]
            col_count +=1
            
        color = colors_dict[tte]


        if c1==c0-1: 
             plt.plot(range(loc,loc+tl),templates[tte,:],'--', color = color)
             break   

        ax.set_xticks([])
        ax.set_xticklabels([])
        plt.plot(range(loc,loc+tl),templates[tte,:],color = color)

        stream[loc:loc+tl] -= templates[tte,:]
        c1+=1

    


    stream= stream_segment.copy()    

    c2=0
    while True:

        convs = np.zeros([num_templates,T+1])
        for i in range(num_templates):
            convs[i,:] = 2*np.correlate(stream, templates[i,:], 'valid') - norm(templates[i,:])**2


        convs[exclude,:]=-1
        
        amax= convs.argmax(1)
        tte = convs[range(num_templates),amax].argmax()
        loc = amax[tte]


        plt.subplot(c0+1,2,2*c2+4)    
        ax = plt.gca()
        ax.set_yticks([])
        ax.set_yticklabels([])

        plt.plot(stream, 'silver')
        plt.ylabel('{0:.2f}'.format(convs[tte,loc]))        
        
        if c2==0:
            axes = plt.gca()        
            ylims= axes.get_ylim()

        else:
            plt.ylim(ylims)


        if tte not in colors_dict:            
            colors_dict[tte] = colors[col_count]
            col_count +=1
            
        color = colors_dict[tte]

                
        if c2==c0-1:

            plt.plot(range(loc,loc+tl),templates[tte,:],'--', color = color)
            break   

        
        plt.plot(range(loc,loc+tl),templates[tte,:],color = color)
        ax.set_xticks([])
        ax.set_xticklabels([])

        stream[loc:loc+tl] -= templates[tte,:]
        c2+=1

    
    save_name ='./figures/NMP_vs_CMP_T0_' + str(T0) + '_T1_' + str(T1) + '_cats_data.pdf'
#    plt.savefig(save_name, bbox_inches="tight")            
    







#################################################################
# compute NMP times and identities  (fast version)


def get_nmp(stream_segment, templates, model, threshold, exclude, c0_max):
    
    
    stream = stream_segment.copy()
    T = stream.shape[0]-75
    
    chosen_probs = []    
    nmp = []
    c0=1

    # initial probabilities
    segments = np.zeros([T,75])
    for i in range(T):
        segments[i,:] = stream[i: i+75]
    probs = model.evaluate(segments, templates)
    
    while True:


        probs[exclude,:] = -1
    
        amax= probs.argmax(1)
        tte = probs[range(num_templates),amax].argmax()
        loc = amax[tte]
        chosen_probs.append(probs[tte,loc])
        
        nmp.append((loc+T0,tte,probs[tte,loc]))
        
        if probs[tte,loc] < threshold or c0==c0_max:
            break   

        c0+=1
    
    
        # update stream
        stream[loc:loc+75] -= templates[tte,:]
    
        # pepare data segments whose probabilities need to be updated
        b0 = max(0,loc-75+1)    
        b1 = min(loc+75,T)
        xb = range(b0,b1)
        bsegments = np.zeros([b1-b0,75])
        for i in range(b1-b0):
            bsegments[i,:] = stream[i+b0:i+b0+75]
            
        # update probabilities in the changed segments        
        bprobs = model.evaluate(bsegments, templates)    
        probs[:,xb] = bprobs
    
    
    ln = len(nmp)
    nmp_ar = np.zeros([ln,2],dtype=np.int32)    #[time locations, template ids]
    nmp_probs = np.zeros(ln)
    
    for i in range(ln):
        nmp_ar[i,0] = nmp[i][0]
        nmp_ar[i,1] = nmp[i][1]
        nmp_probs[i] = nmp[i][2]
    
    return nmp_ar, nmp_probs, stream




    
c0_max = 10
threshold = .10


T0 =39000
T1= 39600
seed = 3
noise = True




fignum=80
model = model2
plot_nmp_cmp(T0,T1,exclude = [], seed=seed,fignum=fignum)



