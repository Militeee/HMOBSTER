import torch
from mobster.ParetoBinomial import ParetoBinomial

L = torch.tensor(0.05)
H = torch.tensor(1)
alpha = torch.tensor(0.5)
trials = torch.tensor(40)



pb = ParetoBinomial(alpha,H,L,trials)

values = torch.linspace(1,40,steps= 40 )

pb.log_prob(values)