from nn.layers import Layer
from nn.activations import Relu
from nn.activations import softmax
from nn.loss import loss

import numpy as np

layer1 = Layer(2,32)
c = np.array([[2, 3],[3, 4]])
out = layer1.forward(c)
#print(out)
Relu_new = Relu()
out1 = Relu_new.forward(out)
#print(out1)
layer2 = Layer(32,4)
out2 = layer2.forward(out1)
#print(out2)
softmax_new = softmax()
out3 = softmax_new.forward(out2)
#print(out3)
loss1 = loss()
b = np.array([1,0])
fl = loss1.forward(b, out3)
L = loss1.backward()
#print(L.shape)
#print(L.shape)
back1 = softmax_new.backward(L)
#print(back1)
back2 = layer2.backward(back1)
#print(back2.shape)
back3 = Relu_new.backward(back2)
#print(back3.shape)
back4 = layer1.backward(back3)
print(back4.shape)
