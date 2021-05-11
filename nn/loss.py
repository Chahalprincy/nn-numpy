import numpy as np
class loss():

    def forward(self,y,inp):
        self.y = y
        self.inp = inp
        loss = -self.inp*np.log(self.inp)
        self.j = np.arange(self.inp.shape[1])
        return loss[self.y,self.j]
        
    def backward(self):
        diff_loss = np.zeros(self.inp.shape)
        diff_loss[self.y, self.j] = -np.log(self.inp[self.y,self.j])-1
        return np.reshape(diff_loss.T, (diff_loss.shape[1],1,diff_loss.shape[0]))