import numpy as np

def getActivationFunction(id):
    return functions[id]

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
        print('Sig data:', data)
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
        return self.forEachLayer(data,
            lambda x : x * (x > 0) + (.1 * x) * (x < 0))

    def derived(self, data):
        pass

    def getName(self):
        return 'lrelu'

class TanHiperbolic(Function):

    def activate(self, data):
        return self.forEachLayer(data,
            lambda x : np.tanh(x))

    def derived(self, data):
        pass

    def getName(self):
        return 'tanh'

functions = {'sig':Sigmoid(), 'relu':ReLU(), 'lrelu': LeakyReLU(), \
'tanh': TanHiperbolic}



def simpleCost(outputValue, correctValue):
    print('O: ', outputValue, ' C: ', correctValue)
    return abs(correctValue - outputValue)



if __name__ == '__main__':
    x = LeakyReLU()
    print(x.activate([2]))
