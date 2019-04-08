
class NeuralNetwork():

    def __init__(self):
        self.layers = []


def addLayer(self, layer):
    """
        Add a layer to the neural network in sequential fashion.
    """
    self.layers.append(layer)

def output(self, x):
    """
        Calculate the output for a single input instance x (one row from
        the training or test set)
    """

    for layer in self.layers:
        x = layer.propagateForward(x)

    return x

def calculateOutputs(self, data):

    outputs = []
    for instance in data:
        outputs.append(self.output(instance))

    return outputs


def feed(self, data):

    outputs = self.calculateOutputs(data)
    return outputs


def calculateError(self, output, result):

    errors = []
    return errors.append(self.errorFunction(output, result))

def getOverallError(self, outputs, results):

    overallError = 0
    for i in range(0, len(outputs)):
        for error in self.calculateError(outputs[i], results[i])
            overallError += error #absoulte ?

    return error

def learn(errors):

    for layer in self.layers[::-1]:
        errors = layer.propagateBackwards(errors)


def train(self, data, results, learningRate):

    for i in range(0, len(data)):

        output = self.output(data[i])
        errors = self.calculateError(output, results[i])
        self.learn(errors)
