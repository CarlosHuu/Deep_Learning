from .layer import *

class Network(object):
    def __init__(self):

        ## by yourself .Finish your own NN framework
        ## Just an example.You can alter sample code anywhere. 
        self.fc1 = FullyConnected(28*28, 2352) ## Just an example.You can alter sample code anywhere. 
        self.relu1 = relu()
        self.fc2 = FullyConnected(2352, 1568)
        self.relu2 = relu()
        self.fc3 = FullyConnected(1568, 784)
        self.relu3 = relu()
        self.fc4 = FullyConnected(784, 10)
        # self.relu4 = relu()
        # self.fc5 = FullyConnected(64, 10)
        self.loss = SoftmaxWithloss()
        

    def forward(self, input, target):
        x = self.fc1.forward(input)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        x = self.relu3.forward(x)
        x = self.fc4.forward(x)
        # x = self.relu4.forward(x)
        # x = self.fc5.forward(x)
        pred, loss = self.loss.forward(x, target)

        return pred, loss

    def backward(self):
        ## by yourself .Finish your own NN framework
        grad = self.loss.backward()
        # grad = self.fc5.backward(grad)
        # grad = self.relu4.backward(grad)
        grad = self.fc4.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.fc3.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.fc1.backward(grad)

    def update(self, lr):
        ## by yourself .Finish your own NN framework
        
        self.fc1.weight -= lr * self.fc1.weight_grad
        self.fc1.bias -= lr * self.fc1.bias_grad
        self.fc2.weight -= lr * self.fc2.weight_grad
        self.fc2.bias -= lr * self.fc2.bias_grad
        self.fc3.weight -= lr * self.fc3.weight_grad
        self.fc3.bias -= lr * self.fc3.bias_grad
        self.fc4.weight -= lr * self.fc4.weight_grad
        self.fc4.bias -= lr * self.fc4.bias_grad
        # self.fc5.weight -= lr * self.fc5.weight_grad
        # self.fc5.bias -= lr * self.fc5.bias_grad
        
        
