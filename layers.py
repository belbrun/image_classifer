import numpy as np


@abstract
class Layer:

    def __init__(self, weights, activationFunction):
        self.weights = weights
        self.activationFunction = activationFunction

    def propagateForward(self):
        pass

    def propagateBackwards(self):
        pass


class ConvolutionLayer(Layer):


    def __init__(self, filterNumber, filterSize, stride, activationFunction):
        self.filters = []
        self.bias = 0
        for i in range(0, filterNumber):
            self.filters.append(np.matrix(np.array((filterSize,filterSize))))
        self.stride = stride
        self.activationFunction = activationFunction

    @static
    def generateConvolutionLayer():
    """
        Returns a new convolution layer given the input data parameters.
    """

    def propagateForward(self, input):
        outputShape = input[0].shape[0] - self.filterSize + 1
        y = np.matrix(np.array(outputShape,outputShape))
        for i in range(0, x.shape[0] - self.filterSize):
            for j in range(0, x.shape[1] - self.filterSize):
                for a in range(0, self.filterSize):
                    for b in range(0, self.filterSize):
                        for (d,x) in enumerate(input):
                            y[i,j] += self.filters[d][a,b] * \
                            x[i*self.stride + a, j*self.stride + b] + self.bias)
        return self.activationFunction(y)


    def propagateBackwards(self, error):
        
