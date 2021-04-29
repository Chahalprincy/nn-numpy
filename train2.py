from nn.convolution1 import Convolution
from nn.layers import Layer
from nn.activations import Relu
from nn.activations import softmax
from nn.loss import loss

import numpy as np
i = 0 
list1 = []
with open("/Users/princychahal/Documents/MNIST Database/train-images-idx3-ubyte", "rb") as f:
    f.seek(16,0)
    while i < 47040000:
        read_file = f.read(1)
        convert_int = int.from_bytes(read_file,"big")
        i = i +1
        list1.append(convert_int)
intm = np.array(list1)
Image = intm.reshape(60000,1,28,28)

i = 0 
test_list = []
with open("/Users/princychahal/Documents/MNIST Database/t10k-images-idx3-ubyte", "rb") as f:
    f.seek(16,0)
    while i < 7840000:
        read_file = f.read(1)
        convert_int = int.from_bytes(read_file,"big")
        i = i +1
        test_list.append(convert_int)
intm = np.array(test_list)
test_image = intm.reshape(10000,1,28,28)

i = 0
list2 = []
with open("/Users/princychahal/Documents/MNIST Database/train-labels-idx1-ubyte", "rb") as l:
    l.seek(8,0)
    while i < 60000:
        read_file1 = l.read(1)
        convert_int1 = int.from_bytes(read_file1,"big")
        i = i +1
        list2.append(convert_int1)
label = np.array(list2)

i = 0
test_list2 = []
with open("/Users/princychahal/Documents/MNIST Database/train-labels-idx1-ubyte", "rb") as l:
    l.seek(8,0)
    while i < 10000:
        read_file1 = l.read(1)
        convert_int1 = int.from_bytes(read_file1,"big")
        i = i +1
        test_list2.append(convert_int1)
test_label = np.array(test_list2)


layer1 = Layer(676,28)
layer2 = Layer(28,10)
conv = Convolution(1,1,3,1)
for k in range(2):
    for j in range(2000):
        out1 = conv.forward(Image[j:j+30,:,:,:])
        #print("out1",out1.shape)
        relu_n = Relu()
        outj = relu_n.forward(out1)
        out2 = layer1.forward(outj)
        #print("layer1 done")
        #print("out2",out2.shape)
        relu_new = Relu()
        out3 = relu_new.forward(out2)
        #print("relu done")
        #print("out3",out3.shape)
        out4 = layer2.forward(out3)
        #print("layer2 done")
        #print("out4",out4.shape)
        softmax1 = softmax()
        out5 = softmax1.forward(out4)
        #print("softmax done")
        #print("out5",out5.shape)
        loss1 = loss()
        out6 = loss1.forward(label[j:j+30],out5)
        #print("loss done")
        back1 = loss1.backward()
        #print("loss backward")
        #print("back1",back1.shape)
        back2 = softmax1.backward(back1)
        #print("softmax back")
        #print("back2",back2.shape)
        back3 = layer2.backward(back2)
        #print("layer2 done")
        #print("back3",back3.shape)
        layer2.update(0.1)
        back4 = relu_new.backward(back3)
        #print("back4",back4.shape)
        back5 = layer1.backward(back4)
        #print("back5",back5.shape)
        layer1.update(0.1)
        backj = relu_n.backward(back5)
        back6 = conv.backward(backj)
        #print("back6",back6.shape)
        conv.update(0.1)

accuracy_list = []
for j in range(500):
    out1 = conv.forward(test_image[j:j+20,:,:,:])
    #print("out1",out1.shape)
    relu_n = Relu()
    outj = relu_n.forward(out1)
    out2 = layer1.forward(outj)
    #print("layer1 done")
    #print("out2",out2.shape)
    relu_new = Relu()
    out3 = relu_new.forward(out2)
    #print("relu done")
    #print("out3",out3.shape)
    out4 = layer2.forward(out3)
    #print("layer2 done")
    #print("out4",out4.shape)
    softmax1 = softmax()
    out5 = softmax1.forward(out4)
    max_index = np.argmax(out5.T,axis = 1)
    a = max_index-test_label[j:j+20]
    accuracy_list.append(a)

print("test accuracy:")
print(((10000-np.count_nonzero(accuracy_list))/10000)*100)


# layer1 = Layer(676,28)
# layer2 = Layer(28,10)
# conv = Convolution(1,1,3,1)
# Image = np.random.rand(30,1,28,28)
# out1 = conv.forward(Image)
# out2 = layer1.forward(out1)
# relu_new = Relu()
# out3 = relu_new.forward(out2)
# out4 = layer2.forward(out3)
# softmax1 = softmax()
# out5 = softmax1.forward(out4)
# print(out5.shape)
# print(out5)
# max_index = np.argmax(out5.T,axis = 1)
# print(max_index)







