import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import random
import pickle

def decimalToBinary(n):
    x = bin(n).replace("0b", "")
    return x.zfill(10)

def input_data(filename):
    input_samples = []
    with open(filename) as openfileobject:
        for line in openfileobject:
            line = line.rstrip("\n")
            input_samples.append(line)
    train_input = input_samples
    for i, item in enumerate(input_samples):
        train_input[i] = decimalToBinary(int(item))
    for i, item in enumerate(train_input):
        train_input[i] = list(item)
    train_input = np.array(train_input, dtype=float)
    train_input = torch.tensor(train_input, dtype=torch.float32)
    return train_input

def output_data(filename):
    output_samples = []
    with open(filename) as openfileobject:
        for line in openfileobject:
            line = line.rstrip("\n")
            output_samples.append(line)
    train_output = output_samples
    for i, item in enumerate(train_output):
        if (item == "Fizz"):
            train_output[i] = "0001"
        elif (item == "Buzz"):
            train_output[i] = "0010"
        elif (item == "FizzBuzz"):
            train_output[i] = "0100"
        else:
            train_output[i] = "1000"
    for i, item in enumerate(train_output):
        train_output[i] = list(item)
        
    train_output = np.argmax(train_output, axis = 1)
    train_output = torch.tensor(train_output, dtype=torch.long)
    return train_output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10,15)
        self.fc2 = nn.Linear(15,8)
        self.fc3 = nn.Linear(8,4)
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__== "__main__":
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay = 0.00001)
    w = torch.tensor([0.6,1,2,0.2],dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight = w)
    X = input_data('train_input.txt')
    Y = output_data('train_output.txt')
    
    index = 0
    for batch_size in [32]: 
        for epoch in range(50000):
            optimizer.zero_grad()
            order=np.arange(900)
                    
            Y_pred = net(X[index:index+batch_size])
            loss = criterion(Y_pred, Y[index:index+batch_size])
            a = accuracy_score(np.argmax(np.array(Y_pred.detach().numpy()), axis=1),
                               np.array(Y[index:index+batch_size].detach().numpy()))
            if (epoch % 100 == 0):
                print('epoch: ', epoch, ' loss: ', loss.item(),
                      ' Train_Acc: ', a*100)
                np.random.shuffle(order)
                X=X[order]
                Y=Y[order]
            index = random.randrange(len(Y)-64)
            loss.backward()
            optimizer.step()
    pickle.dump(net, open("software2", 'wb'))