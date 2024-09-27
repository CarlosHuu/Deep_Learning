import numpy as np

## by yourself .Finish your own NN framework
## Just an example.You can alter sample code anywhere. 


class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
## by yourself .Finish your own NN framework
class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        # self.in_features = in_features
        # self.out_features = out_features
        # # Initialize weights using He initialization
        # self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros((1, out_features))

    def forward(self, input):
        self.input = input
        output = np.dot(self.input, self.weight) + self.bias
        return output

    def backward(self, output_grad):
        input_grad = np.dot(output_grad, self.weight.T)
        self.weight_grad = np.dot(self.input.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0, keepdims=True)

        return input_grad

## by yourself .Finish your own NN framework
class relu(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        output = np.maximum(0, self.input)
        return output

    def backward(self, output_grad):
        relu_grad = (self.input > 0).astype(float)
        input_grad = output_grad * relu_grad
        return input_grad

class SoftmaxWithloss(_Layer):
    def __init__(self):
        pass
        ''' Softmax '''
    def softmax(self, inputs):
        stable_input = inputs - np.max(inputs, axis=1, keepdims=True)
        exps = np.exp(stable_input)
        prob = exps / np.sum(exps, axis=1, keepdims=True)
        return prob
        '''Average loss'''
    def forward(self, input, target):
        
        self.input = input
        self.predict = self.softmax(self.input)
        self.target = target
        # class_num = self.input.shape[0]
        # loss = -np.sum(self.target * np.log(self.predict + 1e-10)) / class_num
        loss = -np.mean(np.sum(self.target * np.log(self.predict + 1e-10), axis=1))
        return self.predict, loss

    def backward(self):
        input_grad = self.predict - self.target
        input_grad /= input_grad.shape[0]
        return input_grad
    
# class Dropout(_Layer):
#     def __init__(self, p=0.5):
#         self.p = p

#     def forward(self, input, train=True):
#         if train:
#             self.mask = np.random.binomial(1, 1 - self.p, size=input.shape)
#             return input * self.mask / (1 - self.p)
#         else:
#             return input

#     def backward(self, output_grad):
#         return output_grad * self.mask / (1 - self.p)