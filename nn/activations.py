import numpy as np

class Relu():

    def forward(self, arr1):
        self.arr1 = arr1
        self.a1 = np.clip(self.arr1,0,None)
        return np.clip(self.arr1,0,None)

    def backward(self, dout):
        
        a = np.expand_dims(self.a1.T, axis=2)
        b = np.expand_dims(self.a1.T, axis=1)
        c = np.zeros(np.shape(a@b))
        d = np.where(b == 0, 0,1)
        c[:,range(self.a1.shape[0]),range(self.a1.shape[0])] = np.squeeze(d, axis = 1)
        return dout@c

class softmax():

    def forward(self, arr):
        self.arr = arr
        self.new_arr = np.exp(self.arr)
        self.summ = self.new_arr.sum(axis = 0)
        return self.new_arr/(self.summ+1e-6)

    def backward(self, dout):
        y = np.expand_dims(self.new_arr.T, axis=1)
        self.new_arr2 = np.expand_dims(self.new_arr.T, axis=2)
        l = np.expand_dims(self.summ, axis=(1,2))
        dia = (y-np.ones(y.shape)*l)*(y)
        b = self.new_arr2 @ y
        b[:,range(self.new_arr2.shape[1]),range(self.new_arr2.shape[1])] = np.squeeze(dia, axis = 1)
        c = (-1/l**2)*b
        return dout@c