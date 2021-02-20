import pickle
import torch

def load_example_data(directory = "./"):
    flh = open(directory + "example.pkl", "rb")
    inp = pickle.load(flh)
    inp = {k: (torch.tensor(v) - 0.001) for k, v in zip(inp.keys(), inp.values())}
    return inp