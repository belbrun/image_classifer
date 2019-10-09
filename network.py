from functions import *
from layers import *
import datautil


class NeuralNetwork():

    def __init__(self, errorFunction=CrossEntropy(), classificationLimit=0.5):
        """
            Default constructor.
            Uses cross entropy as a default error function
            and sets the classification limit to 0.5.
        """
        self.errorFunction = errorFunction
        self.layers = []
        self.classificationLimit = classificationLimit

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

        for i, layer in enumerate(self.layers):
            x = layer.propagateForward(x)
            # if i == 3:
            #    print(layer, x)

        return x

    def calculateOutputs(self, data):
        """
            Calculate outputs for a set of examples and return it as a list.
        """
        outputs = []

        for instance in data:
            outputs.append(self.output(instance))

        return outputs

    def classify(self, x):
        """
            Return output as a classified value mapped to 1 or 0.
            (used in binary classification).
        """
        return self.classifyResult(self.output(x)[0])

    def feed(self, data):
        outputs = self.calculateOutputs(data)
        return outputs

    def feedForError(self, x, result):
        """
            Return a tuple containing the output of the neural network for a
            given input, and its error depending on the correct result.
        """
        output = self.output(x)
        error = self.calculateError(output, result)
        return (output, error)

    def calculateError(self, output, result):
        """
            Calculate the errors of a set of examples, given the correct results
            for the set.
        """
        errors = []
        for i in range(0, result.shape[0]):
            errors.append(self.errorFunction.derived(output[i], result[i]))

        return np.array(errors)

    def getOverallError(self, outputs, results):

        overallError = 0
        for i in range(0, len(outputs)):
            for error in self.calculateError(outputs[i], results[i]):
                overallError += error  #absoulte ?

        return error

    def learn(self, errors, learningRate):
        """
            Use backpropagation algorithm of each layer to propagate the
            error thru the network and update the network weights using a
            given learning rate.
        """

        for index,layer in enumerate(self.layers[::-1]):
            base = 1.9
            #print(index, layer, learningRate * base**(index))
            errors = layer.propagateBackwards(errors, learningRate)
            print(layer, errors)

    def train(self, x, results, learningRate):
        """
            Train the network on a single example using a given learning rate.
        """
        output = self.output(x)
        print('O: ', output, 'C: ', results)
        errors = self.calculateError(output, results)
        self.learn(errors, learningRate)
        return errors



    def classifyResult(self, result):
        """
            Classify result as a 1 or 0 value depending on the networks
            classification limit.
        """
        print(result, self.classificationLimit, result > self.classificationLimit)
        return int(result > self.classificationLimit)

    def setClassificationLimit(self, classificationLimit):
        self.classificationLimit = classificationLimit

    def save(self, path, data = None):
        """
            Save the neural network to a given path.
            Add any additional information as data in a list format. It will
            be saved to a config text file.
        """
        if data:
            data.append('{:01.15f}'.format(self.classificationLimit))
        else:
            print(str((self.classificationLimit)))

            data = ['{:01.15f}'.format(self.classificationLimit)]
        datautil.saveData(path, data)
        for (index, layer) in enumerate(self.layers):
            newPath = path + str(index) + '/'
            datautil.makeDirectory(newPath)
            layer.save(newPath)

    def load(path, loadConfig = True):
        """
            Load a network from the given path. Set load config to false if the
            config text file should not be used to initialize the loaded network.
        """
        if loadConfig:
            classificationLimit = float(datautil.loadData(path, loadConfig)[-2])
        else :
            classificationLimit = 0.5
        network = NeuralNetwork(classificationLimit = classificationLimit)
        for (index, id) in enumerate(datautil.getLayerIds(path)):
            network.addLayer(
                getLayerById(id).load(path + str(index) + '/'))
        return network

    def printNetwork(self):
        for layer in self.layers:
            print(layer)
            if isinstance(layer, ConvolutionLayer):
                print(layer.filters)
                print(layer.bias)
            if isinstance(layer, FullyConnectedLayer):
                print(layer.weights)
                print(layer.bias)
