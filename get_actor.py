import torch


torch.save(torch.load('./pred_ddpg.pt', map_location='cpu')['actor'], 'pred_ddpg_actor.pt')
torch.save(torch.load('./prey_ddpg.pt', map_location='cpu')['actor'], 'prey_ddpg_actor.pt')