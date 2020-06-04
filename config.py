import torch 

params = {   

    
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #'device': torch.device("cpu"),
    
    'beta': 0.1,
    
    'n_raw_timesteps': 150,
    'n_template_timesteps': 121,
    
    # neural net architecture
    'z_dim': 256,
    'h_dim': 512,
    'H_dim': 256,
}



        



    



