from torch import nn
import torch

class Attention:
    def __init__(self, dim, n_vocab):
        self.dim = dim
        self.n_vocab = n_vocab
        
        self.Wq = None
        self.Wk = None
        self.Wv = None
        