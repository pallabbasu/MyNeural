#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 31 07:42:33 2020

@author: aripakman
"""

import numpy as np

class Noise_Generator():
    
    
    def __init__(self):


        temp_cov = np.load('./data/temp_cov_161.npy')

        n_raw_timesteps = 160
        temp_cov = temp_cov[:,:n_raw_timesteps]
        temp_cov = temp_cov[:n_raw_timesteps,:]

        T = temp_cov.shape[0]
        for t in range(1,T):
            c = .5*temp_cov.diagonal(t).mean() + .5*temp_cov.diagonal(-t).mean()
            np.fill_diagonal(temp_cov[:,t:],c)
            np.fill_diagonal(temp_cov[t:,:],c)



        S11 = temp_cov[:,:80]
        S11 = S11[:80,:]
        S22i = np.linalg.inv(S11)
        S12 = temp_cov[:,80:]
        S12 = S12[:80,:]
        S21 = S12.transpose()
        
        C = S11 - np.matmul(S12,np.matmul(S22i,S21))        
        self.Ch = np.linalg.cholesky(C).astype(np.float32)
        
        self.Ih = np.linalg.cholesky(S11).astype(np.float32)        
        self.A = np.matmul(S12,S22i)



    def generate(self,T):
        
        b = T//80 +1
        
        noise = np.zeros([b*80])
        
        e = np.random.normal(0, 1, 80)
        noise[:80] = np.matmul(self.Ih,e)
        
        c = 1
        
        while c < b:
            e = np.random.normal(0, 1, 80)
            noise[80*c:80*(c+1)] = np.matmul(self.Ch,e) + np.matmul(self.A,noise[80*(c-1):80*c])
            c+=1
            
        return noise[:T]            
            




            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        