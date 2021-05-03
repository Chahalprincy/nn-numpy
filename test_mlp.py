from nn.layers import Layer
from nn.activations import Relu
from nn.activations import softmax
from nn.loss import loss

import numpy as np
import torch


def create_nn_layers():
    return {
        "layer1": Layer(2,32),
        "relu1": Relu(),
        "layer2": Layer(32, 4),
        "softmax1": softmax(),
        "loss1": loss(),
    }


def main():
    nn_input = np.array([[2], [3]])
    y_label = 2
    nn_layers = create_nn_layers()
    # forward
    out_layer1 = nn_layers["layer1"].forward(nn_input)
    out_relu1 = nn_layers["relu1"].forward(out_layer1)
    out_layer2 = nn_layers["layer2"].forward(out_relu1)
    out_softmax1 = nn_layers["softmax1"].forward(out_layer2)
    out_loss1 = nn_layers["loss1"].forward(y_label, out_softmax1)
    # backward
    grad_l = nn_layers["loss1"].backward()
    grad_softmax1 = nn_layers["softmax1"].backward(grad_l)
    grad_layer2 = nn_layers["layer2"].backward(grad_softmax1)
    grad_relu1 = nn_layers["relu1"].backward(grad_layer2)
    grad_layer1 = nn_layers["layer1"].backward(grad_relu1)

    t_input = torch.as_tensor(nn_input, dtype=torch.float64)
    ty_label = y_label
    tlayer1_w = torch.tensor(nn_layers["layer1"].Weights, requires_grad=True)
    tlayer1_b = torch.tensor(nn_layers["layer1"].bias, requires_grad=True)
    tlayer2_w = torch.tensor(nn_layers["layer2"].Weights, requires_grad=True)
    tlayer2_b = torch.tensor(nn_layers["layer2"].bias, requires_grad=True)
    tout_layer1 = tlayer1_w @ t_input + tlayer1_b
    tout_relu1 = torch.clamp(tout_layer1, min=0)
    tout_layer2 = tlayer2_w @ tout_relu1 + tlayer2_b
    tout_softmax = torch.exp(tout_layer2) / tout_layer2.exp().sum()
    tloss = -tout_softmax[2, 0] * torch.log(tout_softmax[2, 0])
    tloss.backward()

    print(out_loss1, tloss.item())
    print("-" * 50)

    print(nn_layers["layer2"].diff_Weights)
    print(tlayer2_w.grad)

    print(np.isclose(nn_layers["layer2"].diff_Weights, tlayer2_w.grad.numpy()))


if __name__ == "__main__":
    main()
