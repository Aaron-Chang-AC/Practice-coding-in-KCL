import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#


def my_loss(output, target):
    # ord = 2 is euclidean distance
    loss = 0.5 * torch.linalg.norm(output-target, dim=1, ord=2)**2
    print(loss)
    return loss

def tansigmoid(x):
    activation = 2/(1+torch.exp(-2 * x))-1
    return activation

def logarithmic_sigmoid_function(x):
    activation = 1/(1+torch.exp(-x))
    return activation

def sgn(x):
    for i in range(len(x)):
        if x[i]>= 0:
            x[i] = 1
        else:
            x[i] = -1
    return x

def sigmoid(input):
  return 1 / (1 + torch.exp(-input))

class my_network():
    def __init__(self):
        print("**************PLEASE CHECK YOUR ACTIVATION AND LOSS FUNCTION*****************\n")
        #=====================================================
        self.w11 = torch.tensor([5], dtype=torch.float, requires_grad=True)
        self.w21 = torch.tensor([3], dtype=torch.float, requires_grad=True)
        self.w12 = torch.tensor([1], dtype=torch.float, requires_grad=True)
        self.w22 = torch.tensor([-4], dtype=torch.float, requires_grad=True)
        self.w13 = torch.tensor([-5], dtype=torch.float, requires_grad=True)
        self.w23 = torch.tensor([-2], dtype=torch.float, requires_grad=True)
        self.w10 = torch.tensor([3], dtype=torch.float, requires_grad=True)
        self.w20 = torch.tensor([3], dtype=torch.float, requires_grad=True)
        self.w30 = torch.tensor([-4], dtype=torch.float, requires_grad=True)
        # weight for layer 1 -> 2
        self.m11 = torch.tensor([3], dtype=torch.float, requires_grad=True)
        self.m12 = torch.tensor([1], dtype=torch.float, requires_grad=True)
        self.m13 = torch.tensor([2], dtype=torch.float, requires_grad=True)
        self.m21 = torch.tensor([-5], dtype=torch.float, requires_grad=True)
        self.m22 = torch.tensor([3], dtype=torch.float, requires_grad=True)
        self.m23 = torch.tensor([-5], dtype=torch.float, requires_grad=True)
        self.m10 = torch.tensor([-1], dtype=torch.float, requires_grad=True)
        self.m20 = torch.tensor([-1], dtype=torch.float, requires_grad=True)


        x = [[-0.3, 0.5]]  # input_sample
        t = [[-1, -2]]  # target

        self.x_data = torch.tensor(x, dtype=torch.float)
        self.target_data = torch.tensor(t, dtype=torch.float)
        # =========================================================

        # initialize activation function
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()



    def forward(self):
        y1 = self.w11 * self.x_data[:,0] + self.w21 * self.x_data[:,1] + self.w10
        y2 = self.w12 * self.x_data[:,0]+ self.w22 * self.x_data[:,1] + self.w20
        y3 = self.w13 * self.x_data[:,0]+ self.w23 * self.x_data[:,1] + self.w30
        print(y1)
        print(y2)

        y1 = sigmoid(y1)
        y2 = sigmoid(y2)
        y3 = sigmoid(y3)

        z1 = self.m11 * y1 + self.m12 *y2 + self.m13* y3 + self.m10
        z2 = self.m21 * y1 + self.m22 *y2 + self.m23* y3 + self.m20
        z = torch.stack([z1, z2])
        return z



    def seperate_loss(self):
        # Just to check
        z = self.forward()
        temp = torch.tensor([0],dtype=torch.float)
        for i in range(len(z)):
            loss_t = my_loss(torch.unsqueeze(z[i], 0), torch.unsqueeze(self.target_data[i], 0))
            print(f"Loss is: {loss_t}")
            loss_t.sum().backward(retain_graph=True)
            print(f"Backpropagation for J {i+1}: {self.w11.grad-temp}")
            temp = self.w11.grad.clone()
            # print(f"temp is {temp}")
    def loss(self):
        # define own loss
        z = self.forward()
        print(f"Z is: {z}")
        loss = my_loss(z, self.target_data)
        print(f"Loss is: {loss}")
        loss.sum().backward()
        # print(f"Total backpropagation (dJ/dw11): {self.w11.grad}")
        print(f"Total backpropagation (dJ/dw11): {self.w11.grad}")
        # print(f"Total backpropagation (dJ/dm10): {self.m10.grad}")

        # print(w21.grad)
        # print(w22.grad)
        # print(w10.grad)
        # print(m11.grad)
        # print(m21.grad)
        # print(m22.grad)
        # print(m20.grad)

    def update_weight(self, learning_rate= 0.1):
        # update weight
        # self.w11 = self.w11 - learning_rate * self.w11.grad
        self.w11 = self.w11 - learning_rate * self.w11.grad
        # self.m10 = self.m10 - learning_rate * self.m10.grad
        print(f"Updated weight of w11: {self.w11}")
        # print(f"Updated weight of m11: {self.m10}")


        # print(f"Updated weight of w11: {self.w11}")

    def test(self):
        print("============================TEST RESULT==============================")
        y1 = self.w11 * self.x_data[:, 0] + self.w21 * self.x_data[:, 1] + self.w10
        y2 = self.w12 * self.x_data[:, 0] + self.w22 * self.x_data[:, 1] + self.w20
        y3 = self.w13 * self.x_data[:, 0] + self.w23 * self.x_data[:, 1] + self.w30
        print(y1)
        print(y2)

        y1 = sigmoid(y1)
        y2 = sigmoid(y2)
        y3 = sigmoid(y3)

        z1 = self.m11 * y1 + self.m12 * y2 + self.m13 * y3 + self.m10
        z2 = self.m21 * y1 + self.m22 * y2 + self.m23 * y3 + self.m20
        z = torch.stack([z1, z2])
        return z


Net = my_network()
check = False # check for detail dJ1 and dJ2 use True, if combined use False
if check:
    Net.seperate_loss()
else:
    Net.loss()
Net.update_weight(learning_rate=0.1)

print(Net.test())