import numpy as np
import torch
import time
import os


from models.matcher_stream_np_noisy2 import matcher

from plot_functions  import *


from config import params



model_name = 'matcher_stream_np_noisy2' 
load_from_file = False


from data_generator_stream_np_noisy import generate_stream

data_generator = generate_stream()

model = matcher(params).to(params['device'])



# define containers to collect statistics
losses = []      # NLLs
times = []       # Runtime in seconds for each iteration
accs1 = []
max_ps = []
min_ps = []

# training parameters;
learning_rate = 1e-4
weight_decay = 0.01
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)


if 'fname' in vars():
    del fname

it  = -1
# total number of iterations
it_terminate = 400000
# at these points decrease learning rate by lr_decay
milestones = [it_terminate // 2, it_terminate * 7 // 8]
lr_decay = 0.5
# a callback-function to schedule the LR decay
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones, gamma=lr_decay, last_epoch=it)

it, itt = 0,0

if load_from_file:
    model_checkpoint = './saved_models/matcher/matcher_stream_np_noisy_5_25_34000.pt'
    checkpoint = torch.load(model_checkpoint)    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(params['device'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    it = checkpoint['it']
    losses = checkpoint['losses']
    accs1 = checkpoint['accs1']
    min_ps = checkpoint['min_ps']
    max_ps = checkpoint['max_ps']



if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')


if not os.path.isdir('figures'):
    os.mkdir('figures')

if 'fname' in vars():
    del fname 


batch_size = 6


# set the model parameters to the training mode (vs. eval mode)
model.train()
t_start = time.time()
itt = it

# training iterations
while True:

    it += 1
    # plot training curve to monitor training progress 
    if (it % 500 == 0 and it >= 500) or it == it_terminate+1:
        torch.cuda.empty_cache()

        # plot learning curve         
        #plot_losses2(losses, accs1, 100, save_name='./figures/train_avgs2_' + model_name + '.pdf')        
        #plot_losses22(losses, accs1, max_ps,100, save_name='./figures/train_avgs2_' + model_name + '.pdf')        
        plot_losses23(losses, accs1, max_ps, min_ps, 100, save_name='./figures/train_avgs2_' + model_name + '.pdf')        
        
    # save model checkpoints
    if it % 1000 == 0:
        # remove previous checkpoints 
        if 'fname' in vars():
            os.remove(fname)
        model.params['it'] = it
        fname = 'saved_models/' + model_name + '_' + str(it) + '.pt'        
        torch.save({
            'it': it,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,            
            'accs1': accs1,            
            'max_ps': max_ps,
            'min_ps': min_ps,
            
        }, fname)


    # terminate training 
    if it == it_terminate:
        break


    all_segments, all_noisy_segments, all_templs, all_targets = data_generator.generate(batch_size)    
    loss, lps, ms = model(all_segments, all_noisy_segments, all_templs, all_targets)
        
    optimizer.zero_grad()
    loss.backward()    
    losses.append(loss.item())
    


    # update all parameters by computed gradients
    optimizer.step()    
    # callback for updating learning rate at milestones
    scheduler.step()  

    #########################
    # compute accuracy and max_p 

    max_p = -1
    sum_acc = 0
    for i in range(batch_size):

        lp = lps[i]        
        targets = all_targets[i]        
        
        min_p = np.exp(lp[targets==0]).min()
        
        m = ms[i] 
        mlp = np.argmax(lp[:m,:],1)        
        if m<lp.shape[0]:
            max_p = max(max_p, np.exp(lp[m:,:]).max()  )
        mtg =np.where(targets[:m,:]==0)
        acc=0
        
        for b in np.sort(list(set(mtg[0]))):
            if mlp[b] in mtg[1][mtg[0]==b]:
                acc+=1
        
        acc /=m
        sum_acc +=acc
        
    accs1.append(sum_acc/batch_size)
    min_ps.append(min_p)
    max_ps.append(max_p)                

    lr_curr = optimizer.param_groups[0]['lr']
    
    print('{0}  Mean Loss:{1:.3f}  Acc1:{2:.3f}  Max_p:{3:.3f}  Min_p:{4:.3f} Time/It: {5:.2f} lr: {6}'\
          .format(it, np.mean(losses[-50:]), \
          np.mean(accs1[-50:]),\
          np.mean(max_ps[-50:]),\
          np.mean(min_ps[-50:]),\
          (time.time()-t_start)/(it-itt), lr_curr))


############################













