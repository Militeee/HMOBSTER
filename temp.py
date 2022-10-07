import torch
from mobster.ParetoBinomial import ParetoBinomial

alpha = torch.tensor(1)
L = torch.tensor(0.01)
U = torch.tensor(1)
trials = torch.tensor(40)


value = torch.linspace(1,40, 40, dtype=torch.int)

pb = ParetoBinomial(alpha, U, L, trials)