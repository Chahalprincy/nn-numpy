import numpy as np
class loss():

    def forward(self,y,inp):
        self.y = y
        self.inp = inp
        loss = -self.inp[y]*np.log(self.inp[y])
        return loss
        
    def backward(self):
        diff_loss = np.zeros(self.inp.shape)
        diff_loss[self.y] = -np.log(self.inp[self.y])-1
        return diff_loss
