import pickle
import torch

def load_example_data(directory = "./"):
    flh = open(directory + "example.pkl", "rb")
    inp = pickle.load(flh)
    inp = {k: v.float().round() for k, v in zip(inp.keys(), inp.values())}
    return inp