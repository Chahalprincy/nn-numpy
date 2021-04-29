import numpy as np
class Convolution:
    def __init__(self,hstride, vstride,size,dimn):
        self.hstride = hstride
        self.vstride = vstride
        self.weight_size = size
        self.dimn = dimn
        self.weights = np.random.uniform(-0.1,0.1,(self.dimn, self.weight_size, self.weight_size))
        # print("weights",self.weights)

    def forward(self, image):
        self.image = image
        # print("image", self.image)
        self.Ij = self.image.shape[2]
        self.Ii = self.image.shape[3]
        self.Wj = self.weights.shape[1]
        self.Wi = self.weights.shape[2]
        c = np.zeros((self.dimn,self.Wj,self.Ii-self.Wi)) 
        #print("c",c)
        z = np.concatenate((self.weights,c), axis = 2).reshape(self.dimn,self.Ii*self.Wj)
        #print("z",z)
        #d = z[:,:self.Ii*self.Wj-self.Wi]
        #jkl = self.Ii*(self.Wj-1)+self.Wi
        #print("jkl",jkl)
        d = z[:,:(self.Ii*(self.Wj-1))+self.Wi]

        #print("d",d)
        #print("shape",d.shape)
        self.x = int((self.Ii-self.Wi/self.hstride))+1
        #print("x",self.x)
        self.y = int((self.Ij-self.Wj/self.vstride))+1
        #print("y",self.y)
        l = np.zeros((self.dimn,self.x*self.y,self.Ii*self.Ij))
        #print(l[:,0,0:0+84].shape)
        #print("l",l)

        m = 0
        for j in range(self.y):
            for i in range(self.x):
                l[:,m,(i*self.hstride+j*self.vstride*self.Ii):(i*self.hstride+j*self.vstride*self.Ii)+d.shape[1]] = d 
                m = m+1
        # print("l",l)
        self.nweight = l 
        # print("weight shape",self.nweight)
        Image2 = self.image.reshape(self.image.shape[0],self.dimn,1,self.Ij*self.Ii)
        out = Image2@self.nweight.transpose((0,2,1))
        # print("out",out)
        outf = np.sum(out,axis = 1)
        # print(outf)
        # print(outf.reshape(outf.shape[0],outf.shape[2]).T)
        return outf.reshape(outf.shape[0],outf.shape[2]).T


    def backward(self,dout):
        doutR =dout.reshape(dout.shape[0],1,dout.shape[1],dout.shape[2])
        doutF = np.tile(doutR,(1,self.dimn,1,1))
        doutR2 = dout.reshape(dout.shape[0],1,dout.shape[2],dout.shape[1])
        doutF2 = np.tile(doutR2,(1,self.dimn,1,1))
        a = np.zeros((self.image.shape[0],self.dimn,self.Wj*self.Wi,self.x*self.y))
        for z in range(self.image.shape[0]):
            n = -1
            for y in range(self.Wj):
                for x in range(self.Wi):
                    n = n + 1
                    m = -1
                    for j in range(self.y):
                        for i in range(self.x):
                            m = m+1
                            a[z,:,n,m] = self.image[z,:,y+j*self.vstride,x+i*self.hstride]
        diff_weight = a @ doutF2
        self.diff_weight = np.sum(diff_weight,axis = 0).reshape(self.weights.shape[0],self.weights.shape[1],self.weights.shape[2])
        self.diff_out = doutF@self.nweight
        return self.diff_out

    def update(self, y):
        self.weights = self.weights - y * self.diff_weight.reshape(self.dimn,self.Wi,self.Wj)
        

