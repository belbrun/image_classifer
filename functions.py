import numpy as np

def getActivationFunction(id):
    return activationFunctions[id]

class Function:
    """
        Model the requirements of a function in a neural network.
    """
    def activate(self, data):
        """
            Calculate the output of a function with given input data.
        """
        pass

    def derived(self, x):
        """
            Calculate the output of the functions derivative for given arguments.
        """
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
        softmax = self.activate(x)
        jacobian = np.zeros((len(x), len(x)))
        for index, value in enumerate(softmax):
            for i in range(0, len(x)):
                if i == index:
                    jacobian[index, i] += softmax[i]*(1-softmax[i])
                else:
                    jacobian[index, i] += -softmax[i]*softmax[index]

            #result += softmax[i]*(1-softmax[i]) if i == index else \
            #            -softmax[i]*softmax[index]
        return jacobian

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


activationFunctions = {'sig': Sigmoid(), 'relu': ReLU(), 'lrelu': LeakyReLU(), \
'tanh': TanHiperbolic(), 'smax': SoftMax(), 'ratio': RatioFunction()}



class CrossEntropy(Function):

    def activate(self, outputValue, correctValue):

        return -(correctValue*np.log(outputValue) + \
                (1-correctValue)*np.log(1-outputValue))

    def derived(self, outputValue, correctValue):
        #return (1-correctValue)/outputValue - correctValue/(1-outputValue) #change error values
        print('CEIN: ', outputValue, correctValue)
        return (1-correctValue)/(1-outputValue) - correctValue/outputValue

    def getName(self):
        return 'centropy'

class CategoricCrossEntropy(Function):

    def activate(self, outputValue, correctValue):
        result = 0
        for index, output in enumerate(outputValue):
            result += -output*correctValue[index]
        return result

    def derived(self, outputValue, correctValue):
        print('CEIN: ', outputValue, correctValue)
        return (1-correctValue)/(1-outputValue) - correctValue/outputValue




if __name__ == '__main__':
    x = RatioFunction()
    print(x.activate(np.array([-1,23])))
