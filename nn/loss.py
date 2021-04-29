import numpy as np
class loss():

    def forward(self,y,inp):
        self.y = y
        self.inp = inp
        self.loss = self.inp[y]*np.log(self.inp[y])
        diff_loss = np.zeros(self.inp.shape)
        diff_loss[y] = -np.log(self.inp[y])-1
        return diff_loss
        