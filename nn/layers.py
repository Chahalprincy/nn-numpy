import numpy as np

class Layer():
    def __init__(self,inp,out):
        self.inp = inp
        self.out = out
        self.Weights = np.random.rand(self.out, self.inp)
        self.bias = np.random.rand(self.out,1)
        self.diff_bias = None
        self.diff_weights = None
        self.diff_out = None

    def forward(self, other):
        
        # print(Weights.shape)
        # print(bias.shape)
        self.other = other
        return self.Weights@self.other + self.bias

    def backward(self, dout):
        self.diff_out = np.copy(self.Weights)
        #print(self.Weights.shape)
        self.diff_bias = dout
        #print(dout.transpose().shape)
        #print(self.other.transpose().shape)
        self.diff_Weights = dout.transpose()@self.other.transpose()
        return dout@self.diff_out



        



