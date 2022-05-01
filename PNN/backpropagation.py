import torch
import torch.nn as nn
import torch.nn.functional as F
#


def my_loss(output, target):
    loss = 0.5 * torch.linalg.norm(output-target, dim=1, ord=2)**2
    print(loss)
    return loss

def tansigmoid(x):
    activation = 2/(1+torch.exp(-2 * x))-1
    return activation

class my_network():
    def __init__(self):
        print("**************PLEASE CHECK YOUR LOSS FUNCTION*****************\n")
        # weight for layer 0 -> 1
        # self.w11= torch.tensor([1], dtype=torch.float, requires_grad=True)
        # self.w21= torch.tensor([2], dtype=torch.float, requires_grad=True)
        # self.w22= torch.tensor([3], dtype=torch.float, requires_grad=True)
        # self.w10= torch.tensor([4], dtype=torch.float, requires_grad=True)
        # # weight for layer 1 -> 2
        # self.m11= torch.tensor([5], dtype=torch.float, requires_grad=True)
        # self.m21= torch.tensor([6], dtype=torch.float, requires_grad=True)
        # self.m22= torch.tensor([7], dtype=torch.float, requires_grad=True)
        # self.m20= torch.tensor([8], dtype=torch.float, requires_grad=True)

        x = [[-1, 2], [-3, 4], [-3, 4]]  # input_sample
        t = [[-6, 8], [-2, 4], [-2, 4]]  # target

        self.x_data = torch.tensor(x, dtype=torch.float)
        self.target_data = torch.tensor(t, dtype=torch.float)

        #=====================================================
        self.w11 = torch.tensor([0.5], dtype=torch.float, requires_grad=True)
        self.w21 = torch.tensor([0.3], dtype=torch.float, requires_grad=True)
        self.w22 = torch.tensor([-0.7], dtype=torch.float, requires_grad=True)
        self.w10 = torch.tensor([0.2], dtype=torch.float, requires_grad=True)
        # weight for layer 1 -> 2
        self.m11 = torch.tensor([0.8], dtype=torch.float, requires_grad=True)
        self.m12 = torch.tensor([1.6], dtype=torch.float, requires_grad=True)
        self.m10 = torch.tensor([-0.4], dtype=torch.float, requires_grad=True)

        x = [[0.1, 0.9]]  # input_sample
        t = [[0.5]]  # target

        self.x_data = torch.tensor(x, dtype=torch.float)
        self.target_data = torch.tensor(t, dtype=torch.float)
        # =========================================================

        # initialize activation function
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()


    def forward(self):
        y1 = self.w11 * self.x_data[:,0] + self.w10
        y2 = self.w21 * self.x_data[:,0]+ self.w22 * self.x_data[:,1]
        y1 = tansigmoid(y1)
        y2 = tansigmoid(y2)
        print(y1)
        print(y2)
        z1 = self.m11 * y1 + self.m12 *y2 + self.m10
        z = tansigmoid(z1)
        return z

    # def forward(self):
    #     y1 = self.w11 * self.x_data[:,0] + self.w10
    #     y2 = self.w21 * self.x_data[:,0]+ self.w22 * self.x_data[:,1]
    #     z1 = self.m11 * y1
    #     z2 = self.m21 * y1 + self.m22 * y2 + self.m20
    #     z = torch.stack([z1,z2], dim=0).transpose(0,1)
    #     return z

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
        print(f"Total backpropagation (dJ/dw21): {self.w21.grad}")
        print(f"Total backpropagation (dJ/dm10): {self.m10.grad}")

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
        self.w21 = self.w21 - learning_rate * self.w21.grad
        self.m10 = self.m10 - learning_rate * self.m10.grad
        print(f"Updated weight of w21: {self.w21}")
        print(f"Updated weight of m11: {self.m10}")


        # print(f"Updated weight of w11: {self.w11}")

    def test(self):
        y1 = self.w11 * self.x_data[:, 0] + self.w10
        y2 = self.w21 * self.x_data[:, 0] + self.w22 * self.x_data[:, 1]
        y1 = tansigmoid(y1)
        y2 = tansigmoid(y2)
        z1 = self.m11 * y1 + self.m12 * y2 + self.m10
        z = tansigmoid(z1)
        print(f"The tested result after update is: {z}")
        loss = my_loss(z, self.target_data)
        print(f"The updated result loss now is: {loss}")
        return z

Net = my_network()
check = False # check for detail dJ1 and dJ2 use True, if combined use False
if check:
    Net.seperate_loss()
else:
    Net.loss()
Net.update_weight(learning_rate=0.25)

Net.test()