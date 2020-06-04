
import numpy as np
import torch
from torch import nn
import torch.distributions as dist



# model based on https://arxiv.org/abs/1612.00410

class matcher(nn.Module):

    def __init__(self, params):
        super(matcher, self).__init__()


        self.params = params

        H = params['H_dim']
        self.z_dim = params['z_dim']
        self.h_dim = params['h_dim']
        self.device = params['device']
        
        self.n_chan = 1#params['n_channels']
        self.n_steps = params['n_template_timesteps']

        self.t = 75


        self.h = torch.nn.Sequential(        
            nn.Conv1d(1, 24, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(24, 36, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),            
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        
         
        self.pz_x = torch.nn.Sequential(
            torch.nn.Linear(36*17, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, 2*self.z_dim),
            )
            
            
        self.py_z = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, 2*self.t)
            )


        self.h2 = torch.nn.Sequential(        
            nn.Conv1d(1, 24, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(24, 36, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),            
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        
         
        self.pz_x2 = torch.nn.Sequential(
            torch.nn.Linear(36*17, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, 2*self.z_dim),
            )
            
            
        self.py_z2 = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),                                    
            torch.nn.Linear(H, 2*self.t)
            )


        self.log_prob = torch.nn.Sequential(
            torch.nn.Linear(2*self.z_dim, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, H),
            torch.nn.PReLU(),
            torch.nn.Linear(H, 1, bias=False),
        )


    def encode1(self, data):


        h0 = self.h(data)        
        h1 = h0.view(h0.size(0), -1)

        mu_logstd_z  = self.pz_x(h1)
        
        
        pz_mu = mu_logstd_z[:,self.z_dim:]
        pz_sigma = mu_logstd_z[:,:self.z_dim].exp()
        
        pz = dist.Normal(pz_mu,pz_sigma)
        z = pz.rsample()  

        mu_logstd_y = self.py_z(z)                  
        
        py_mu = mu_logstd_y[:,:self.t]
        py_sigma = mu_logstd_y[:,self.t:].exp()
        
        py = dist.Normal(py_mu,py_sigma)
        
        
        loss1 = -py.log_prob(data[:,0,:]).sum(1).mean()
        
        KL = - pz_sigma.log() + (pz_sigma.pow(2) + pz_mu.pow(2))/2
        KL = KL.sum(1).mean()
        
        loss = loss1+ 3*KL


        return loss, pz_mu

    def encode2(self, data,noisy_data):

        h0 = self.h2(noisy_data)        
        h1 = h0.view(h0.size(0), -1)
        mu_logstd_z  = self.pz_x2(h1)        
        
        pz_mu = mu_logstd_z[:,self.z_dim:]
        pz_sigma = mu_logstd_z[:,:self.z_dim].exp()
        
        pz = dist.Normal(pz_mu,pz_sigma)
        z = pz.rsample()  

        mu_logstd_y = self.py_z2(z)                  
        
        py_mu = mu_logstd_y[:,:self.t]
        py_sigma = mu_logstd_y[:,self.t:].exp()
        
        py = dist.Normal(py_mu,py_sigma)
        
        
        loss1 = -py.log_prob(data[:,0,:]).sum(1).mean()
        
        KL = - pz_sigma.log() + (pz_sigma.pow(2) + pz_mu.pow(2))/2
        KL = KL.sum(1).mean()
        
        loss = loss1+ .1*KL


        return loss, pz_mu
    
    

    def forward(self, all_segments, all_noisy_segments, all_templs, all_targets):


        # the inputs are numpy arrays
        # segments = [num_segments, 55] 
        # templates = [num_templates, 55] 

        all_loss = 0
        lps = []
        ms = []
        
        batch_size = len(all_segments)
        for i in range(batch_size):
            
            segments = all_segments[i]
            noisy_segments = all_noisy_segments[i]
            templs = all_templs[i]
            targets = all_targets[i]



            assert segments.shape[1] == templs.shape[1]
            assert targets.shape[0] == templs.shape[0]
            
            num_segments = segments.shape[0]            
            num_templates = templs.shape[0]        
            num_targets  = targets.shape[0]        
            templ_size = templs.shape[1]    

            
            t_segments = torch.tensor(segments).to(self.params['device']).type(torch.float32)
            t_noisy_segments = torch.tensor(noisy_segments).to(self.params['device']).type(torch.float32)            
            t_templates = torch.tensor(templs).to(self.params['device']).type(torch.float32)
            t_targets = torch.tensor(targets).to(self.params['device']).type(torch.float32)
    
            big_segments = t_segments.unsqueeze(0).expand([num_templates,num_segments,templ_size]).reshape([num_templates*num_segments,1,templ_size])                
            big_noisy_segments = t_noisy_segments.unsqueeze(0).expand([num_templates,num_segments,templ_size]).reshape([num_templates*num_segments,1,templ_size])
            big_templates = t_templates.unsqueeze(1).expand([num_templates,num_segments,templ_size]).reshape([num_templates*num_segments,1,templ_size])                
    
    
    
            loss_temps, z_templates  = self.encode1(big_templates)        

            loss_data, z_data  = self.encode2(big_segments, big_noisy_segments)        


    
            enc = torch.cat([z_data,z_templates], dim =1 )            
            logits = self.log_prob(enc).view([num_templates*num_segments])        
            
            t_targets = t_targets.view([num_templates*num_segments])
            
            
            pb = dist.Bernoulli(logits=logits)
            nll= -pb.log_prob(t_targets)
            
            
            mask = (1-targets).copy()
            tt = np.where(targets==0)

            tx = tt[0]
            tz = tt[1]

            mask[tx[tz<num_segments-1],tz[tz<num_segments-1] +1] =1
            mask[tx[tz>0],tz[tz>0] -1] =1
            
            tzp = tz[tz<num_segments-1] +1
            tzm = tz[tz>0] -1            
            m = len(set(tx))
            mask[m:,tz] = 1
            mask[m:,tzp] = 1
            mask[m:,tzm] = 1

            t_mask = torch.tensor(mask).to(self.params['device']).type(torch.float32).bool()
            t_mask = t_mask.view([num_templates*num_segments])
    
    
            nll_tt= nll[t_mask]
            bern_losses = nll.mean() + 20*nll_tt.mean()
            
    
            tt = torch.zeros([num_templates*num_segments]).to(self.params['device'])
            lp  = pb.log_prob(tt).view([num_templates,num_segments]).cpu().detach().numpy() 
            #mm = lp.argmax(1)                        
    
            
            loss = loss_data + loss_temps + 50*bern_losses
            
            
            vae_loss = loss_data.item() + loss_temps.item()
            b_loss = bern_losses.item()



            all_loss += loss
            lps.append(lp)
            ms.append(m)
        
        return all_loss/batch_size, lps, ms



    def evaluate(self, noisy_segments, templates):   
        
        # the inputs are numpy arrays
        # segments = [num_segments, self.t] 
        # templates = [num_templates, self.t] 
        
        assert noisy_segments.shape[1] == templates.shape[1]

        num_segments = noisy_segments.shape[0]
        num_templates = templates.shape[0]        
        
        t_segments = torch.tensor(noisy_segments).to(self.params['device']).type(torch.float32)
        t_templates = torch.tensor(templates).to(self.params['device']).type(torch.float32)
        
        big_segments = t_segments.unsqueeze(0).expand([num_templates,num_segments,self.t]).reshape([num_templates*num_segments,1,self.t])                
        big_templates = t_templates.unsqueeze(1).expand([num_templates,num_segments,self.t]).reshape([num_templates*num_segments,1,self.t])                
        
        with torch.no_grad():        
        
            _, z_templates  = self.encode1(big_templates)        
            
            h0 = self.h2(big_segments)        
            h1 = h0.view(h0.size(0), -1)
            mu_logstd_z  = self.pz_x2(h1)                    
            pz_mu = mu_logstd_z[:,self.z_dim:]
            
            enc = torch.cat([pz_mu, z_templates], dim =1 )            
            logits = self.log_prob(enc).view([num_templates,num_segments])        
            
        logits = logits.cpu().numpy()            
        logits[logits>70] = 70        # to avoid overflow warnings
        probs = 1/(1+np.exp(logits))
        
        return probs
        
        





    
    
    
