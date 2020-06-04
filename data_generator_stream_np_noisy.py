from collections import defaultdict
import numpy as np
import torch

from noise_generator import Noise_Generator


class generate_stream():    
    
    def __init__(self):
        

        self.noise_generator = Noise_Generator()


        data_dir = './data/'
        data_set = 'data_raw_30000hz_300sec_5hz_to_25hz'
        fname_times = data_dir + data_set + '_times.npy'
        fname_templates = data_dir + 'templates_np.npy'

        cat_times= np.load(fname_times, allow_pickle = True)        

        self.templates = np.load(fname_templates)        
        self.t = self.templates.shape[1]   # should be 80
        self.num_templates = self.templates.shape[0]  

        times = []
        for i in range(self.num_templates):
            times.append(cat_times[i])

        # make dictionary td[spike_time] = num_of_a_template btw 0 and 7 
        
        td = {}
        for i in range(self.num_templates):
            ti = times[i]
            for j in ti:
                td[j] = i        

        self.all_spike_times = np.sort(list(set(td.keys())))
        self.num_events = len(self.all_spike_times)-10
        self.td = td
        self.M = 12 # num of templates present + nums of negative templates

        
        # self.matrix_diffs = np.zeros([self.num_templates,self.num_templates])
        # for i in range(self.num_templates):
        #     for j in range(i+1):
        #         u = self.templates[i,:]
        #         v = self.templates[j,:]
        #         self.matrix_diffs[i,j] =  np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        #         self.matrix_diffs[j,i] =  self.matrix_diffs[i,j]




    def generate(self, batch_size=10, seed=None):
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        all_segments = []
        all_noisy_segments = []
        all_templs = []
        all_targets =[]
        t = self.t 

        dilations = .8 +.4*np.random.rand(self.num_templates)         
        
        for _ in range(batch_size):
    
            i = np.random.choice(self.num_events,1)[0]
            Ti = self.all_spike_times[i]
            
            inds_big = np.where((self.all_spike_times > Ti-2*t) & (self.all_spike_times < Ti+2*t) )[0]
            inds_target = np.where((self.all_spike_times > Ti-t) & (self.all_spike_times < Ti+t) )[0]   
            

            ss = set()
            for ii in self.all_spike_times[inds_target]:
                ss.add(self.td[ii]) 

            
            mmx = np.random.choice(len(ss),1)[0]+1    # how many different templates to include
            
 
            
            included_temps = np.array(list(ss))[np.random.choice(len(ss),mmx)]
                       
            big_segment = np.zeros(5*t)
            for k in self.all_spike_times[inds_big]:
                temp_ind = self.td[k]
                if temp_ind in included_temps:
                    start = k-(Ti-2*t)                    
                    #print(temp_ind)
                    big_segment[start:start+t] += dilations[temp_ind]*self.templates[temp_ind,:]
                    
                    
            noisy_big_segment = big_segment + self.noise_generator.generate(5*t)                    
        
            segments = np.zeros([2*t,t])
            noisy_segments = np.zeros([2*t,t])
            
            for i in range(2*t):
                segments[i,:] = big_segment[t+i:2*t+i]
                noisy_segments[i,:] = noisy_big_segment[t+i:2*t+i]
            
            
            s = defaultdict(list)
            for ii in self.all_spike_times[inds_target]:
                temp_ind = self.td[ii]
                if temp_ind in included_temps:
                    s[temp_ind].append(ii)
            
            m = len(s)    # number of different spike times in this segment
            
            M = max(m,self.M)
            templs = np.zeros([M,t])
            targets = np.ones([M,2*t])
            
            for i,k in enumerate(s.keys()):

                for tt in s[k]:                
                    j = tt-Ti+t
                    targets[i,j] = 0
 
                templs[i,:] = dilations[k]*self.templates[k,:]
    
            ms = np.array(list(s.keys()), dtype= np.int32)
            

            nm = np.random.choice(self.num_templates-m,size = M-m,replace=False)                   
            nn = np.array(list(set(range(self.num_templates)).difference(set(ms)) ) )
            
            templs[m:,:] = self.templates[nn[nm],:]
            
            #print(s)

            all_segments.append(segments)
            all_noisy_segments.append(noisy_segments)
            all_targets.append(targets)
            all_templs.append(templs)
            
        return all_segments, all_noisy_segments, all_templs, all_targets

        
                              


