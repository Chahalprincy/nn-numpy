from nn.layers import Layer
from nn.activations import Relu
from nn.activations import softmax
from nn.loss import loss

import numpy as np

train_features = np.load("/Users/princychahal/Downloads/swiss-roll-data/train_features.npy")
train_labels = np.load("/Users/princychahal/Downloads/swiss-roll-data/train_labels.npy")
test_features = np.load("/Users/princychahal/Downloads/swiss-roll-data/test_features.npy")
test_labels = np.load("/Users/princychahal/Downloads/swiss-roll-data/test_labels.npy")

j = 0
layer1 = Layer(3,32)
layer2 = Layer(32,3)
while j < 10:
        for i in range(0,13000,3):
                c = np.array(train_features[i:i+3].T)
                out = layer1.forward(c)
                Relu_new = Relu()
                out1 = Relu_new.forward(out)
                out2 = layer2.forward(out1)
                softmax_new = softmax()
                out3 = softmax_new.forward(out2)
                max_index = np.argmax(out3.T,axis = 1)
                a = max_index-(train_labels[i:i+3])
                k = np.count_nonzero(a)
                loss1 = loss()
                b = np.array(train_labels[i:i+3])
                fl = loss1.forward(b, out3)
                L = loss1.backward()
                back1 = softmax_new.backward(L)
                back2 = layer2.backward(back1)
                layer2.update(0.1)
                back3 = Relu_new.backward(back2)
                back4 = layer1.backward(back3)
                layer1.update(0.1)
        c = np.array(test_features.T)
        out = layer1.forward(c)
        Relu_new = Relu()
        out1 = Relu_new.forward(out)
        out2 = layer2.forward(out1)
        softmax_new = softmax()
        out3 = softmax_new.forward(out2)
        max_index = np.argmax(out3.T,axis = 1)
        a = max_index-test_labels
        print("test accuracy:")
        print(((2000-np.count_nonzero(a))/2000)*100)
        j = j +1
     


