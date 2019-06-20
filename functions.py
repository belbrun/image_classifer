import numpy as np

def getActivationFunction(id):
    return activationFunctions[id]

class Function:

    def activate(self, data):
        pass

    def derived():
        pass

    def forEachLayer(self, data, function):
        outputs = []
        for x in data:
            y = function(x)
            outputs.append(y)

        return outputs

class Identity(Function):

    def activate(self, x):
        return 1


class Sigmoid(Function):

    def activate(self, data):
        return self.forEachLayer(data,
            lambda x : 1. / (1. + np.exp(-x)))

    def derived(self, x):
        sigmX = 1. / (1. + np.exp(-x))
        return sigmX * (1 - sigmX)

    def getName(self):
        return 'sig'


class ReLU(Function):

    def activate(self, data):
        return self.forEachLayer(data,
            lambda x : x * (x > 0))

    def derived(self, x):
        return 1 * (x > 0)

    def getName(self):
        return 'relu'


class LeakyReLU(Function):

    def activate(self, data):
        #print('DATA: ', data)
        return self.forEachLayer(data,
            lambda x : x * (x > 0) + (.1 * x) * (x < 0))

    def derived(self, x):

        return 1 * (x > 0) + 0.1 * (x < 0)

    def getName(self):
        return 'lrelu'

class TanHiperbolic(Function):

    def activate(self, data):
        return self.forEachLayer(data,
            lambda x : np.tanh(x))

    def derived(self, x):
        #print('TANH DER: ', x, 1/np.cosh(x)**2)
        return 1 / np.cosh(x)**2

    def getName(self):
        return 'tanh'


class SoftMax(Function):

    def activate(self, data):
        classValues = np.exp(data)
        return classValues/np.sum(classValues)

    def derived(self, x):
        values, index = x
        softmax = self.activate(values)
        result = 0
        for i in range(0, len(values)):
            result += softmax[i]*(1-softmax[i]) if i == index else \
                        -softmax[i]*softmax[index]
        return result

    def getName(self):
        return 'smax'

class RatioFunction(Function):

    def activate(self, data):
        data = abs(data[0])
        return [data/np.sum(data)]

    def derived(self, x):
        values, index = x
        return values[index] / (np.sum(values)**2)

    def getName(self):
        return 'ratio'

activationFunctions = {'sig':Sigmoid(), 'relu':ReLU(), 'lrelu': LeakyReLU(), \
'tanh': TanHiperbolic(), 'smax': SoftMax(), 'ratio': RatioFunction()}



def simpleCost(outputValue, correctValue):
    print('O: ', outputValue, ' C: ', correctValue)
    #print('Err: ', correctValue - outputValue)
    return (correctValue - outputValue)/correctValue

class CrossEntropy(Function):

    def activate(self, outputValue, correctValue):

        return -(correctValue*np.log(outputValue) + \
                (1-correctValue)*np.log(1-outputValue))

    def derived(self, outputValue, correctValue):
        #return (1-correctValue)/outputValue - correctValue/(1-outputValue) #change error values
        return (1-correctValue)/(1-outputValue) - correctValue/outputValue

    def getName(self):
        return 'ratio'

if __name__ == '__main__':
    x = RatioFunction()
    print(x.activate(np.array([-1,23])))
