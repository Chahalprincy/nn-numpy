import numpy as np

class Layer():
    def __init__(self,inp,out):
        self.inp = inp
        self.out = out
        self.Weights = np.random.uniform(-0.1,0.1,(self.out, self.inp))
        self.bias = np.random.uniform(-0.1,0.1,(self.out,1))
        self.diff_bias = None
        self.diff_weights = None
        self.diff_out = None

    def forward(self, other):
        self.other = other
        return self.Weights@self.other + self.bias

    def backward(self, dout):
        self.diff_out = np.copy(self.Weights)
        self.diff_bias = dout.sum(axis=0).T
        y = np.reshape(dout,(dout.shape[0],dout.shape[2],1))
        a = np.expand_dims(self.other.T, axis=(1))
        self.diff_Weights = (y@a).sum(axis=0)
        return dout@self.diff_out

    def update(self, y):
        self.Weights = self.Weights - y * self.diff_Weights
        self.bias = self.bias - y*self.diff_bias
        



        



