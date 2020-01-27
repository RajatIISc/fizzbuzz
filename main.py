import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pickle
import getopt, sys

def usage():
  print ("\nThis is the usage function\n")
  print ('Usage: '+sys.argv[0]+' -i <file1> [option]')


def decimalToBinary(n):
    x = bin(n).replace("0b", "")
    return x.zfill(10)[-10:]

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
        if (item == "fizz" or item == "Fizz"):
            train_output[i] = "0001"
        elif (item == "buzz"or item == "Buzz"):
            train_output[i] = "0010"
        elif (item == "fizzbuzz"or item == "FizzBuzz"):
            train_output[i] = "0100"
        else:
            train_output[i] = "1000"
    for i, item in enumerate(train_output):
        train_output[i] = list(item)
        
    train_output = np.argmax(train_output, axis = 1)
    train_output = torch.tensor(train_output, dtype=torch.long)
    return train_output

def software1(filename):
    input = []
    with open(filename) as openfileobject:
        for line in openfileobject:
            line = line.rstrip("\n")
            input.append(line)
    input = [int(i) for i in input] 
    output = input
    for i in range(len(input)):
        if(input[i]%3 == 0 and input[i]%5!=0):
            output[i] = 'fizz'
        elif(input[i]%5 == 0 and input[i]%3!=0):
            output[i] = 'buzz'
        elif(input[i]%3 == 0 and input[i]%5==0):
            output[i] = 'fizzbuzz'
    with open('Software1.txt', 'w') as f:
        for item in output:
            f.write("%s\n" % item)       

def software2(filename):
    software2 = pickle.load(open("Model/software2","rb"))
    X = input_data("test_input.txt")
    Y_pred = software2(X)
    Y_pred = np.argmax(np.array(Y_pred.detach().numpy()), axis=1)
    test_output = [str(i) for i in Y_pred]
    for i, item in enumerate(Y_pred):
        if(item==0):
            test_output[i] = i+1
        elif(item==1):
            test_output[i] = "fizzbuzz"
        elif(item==2):
            test_output[i] = "buzz"
        elif(item==3):
            test_output[i] = "fizz"
    with open('Software2.txt', 'w') as f:
        for item in test_output:
            f.write("%s\n" % item)       


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10,8)
        self.fc2 = nn.Linear(8,6)
        self.fc3 = nn.Linear(6,4)
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x



def main(argv):
    print("Name: Rajat Nagpal")
    print("Email Id: rajatnagpal@iisc.ac.in")
    print("SR No.: 04-03-02-10-42-18-1-15533")
    print("Course: M.Tech Systems Engineering (2nd year)")
    try:
        opts, args = getopt.getopt(argv, '', ['help', 'test-data='])
        if not opts:
            print ('No options supplied')
            usage()
    except getopt.GetoptError as e:
        print (e)
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit(2)
        else:
            print(".......................................")
            print("Generating software1.txt ")
            print(".......................................")
            print("Generating software2.txt ")
            print(".......................................")
            software1(arg)
            software2(arg)
    
    
    

if __name__ =='__main__':
    main(sys.argv[1:])