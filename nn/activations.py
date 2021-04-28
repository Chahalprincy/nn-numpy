import numpy as np

class Relu():

    def forward(self, arr1):
        self.arr1 = arr1
        self.a1 = np.clip(self.arr1,0,None)

        return np.clip(self.arr1,0,None)

    def backward(self, dout):
        c = np.zeros(np.shape(self.a1@self.arr1.transpose()))
        d = np.where(self.a1 == 0, 0,1)
        np.fill_diagonal(c,d)
        #print(c.shape)
        return dout@c

class softmax():

    def forward(self, arr):
        self.arr = arr
        self.new_arr = np.exp(self.arr)
        self.summ = np.sum(self.new_arr)
        self.a2 = (self.new_arr/self.summ)
        return self.new_arr/self.summ

    def backward(self, dout):
        self.new_arr_T = self.new_arr.transpose()
        dia = np.ones(np.shape(self.arr))-((np.ones(np.shape(self.new_arr))/self.new_arr)*self.summ)
        b = self.new_arr @ self.new_arr_T
        np.fill_diagonal(b,dia)
        #print(b.shape)
        return dout@b



          

          


