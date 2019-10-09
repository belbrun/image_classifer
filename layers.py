import numpy as np
import datautil
from functions import getActivationFunction

minWeight = -0.5
maxWeight = 0.5

class Layer:
    """
        Abstract class that models a basic neural network layer.
    """

    def __init__(self, activationFunction):
        self.activationFunction = activationFunction


    def propagateForward(self, input):
        """
            Use a forward propagation algorithm to get the layers output from
            input data.
        """
        pass

    def propagateBackwards(self, errors):
        """
            Use the layers backpropagation algorithm to calculate the error
            of the preceding layer, correct layers weights if there
            are any.
        """
        pass

    def save(self, path):
        """
            Save the data needed to recreate the layer to a given path.
        """
        pass

    def load(path):
        """
            Load the layer data from the given path and return a layer object.
            Can only load data saved with the layers save method.
        """
        pass


class ConvolutionLayer(Layer):

    """
        Layer type that uses convolution with a set of filters to create a
        vector of feature maps from a 2D map or a 3D volume.
    """

    def __init__(self, filterNumber, filterSize, stride, activationFunction,\
                 inputDepth, filters = None, bias = None):

        if filters is not None:
            self.filters = filters
        else:
            self.filters = ConvolutionLayer\
                .initializeFilters(filterNumber, inputDepth, filterSize)
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.uniform(minWeight, maxWeight, size=filterNumber)

        self.filterNumber = filterNumber
        self.filterSize = filterSize
        self.stride = stride
        self.activationFunction = activationFunction
        self.data = None
        self.inputSize = 0
        self.inputDepth = inputDepth

    def initializeFilters(filterNumber, inputDepth, filterSize, path = None):
        """
            Initialize and create the needed ammount of filters with random
            numbers.
        """
        filters = []
        for i in range(0, filterNumber):
            filterGroup = []
            for j in range(0, inputDepth):
                filter = None
                if path:
                    filter = np.load(path+'g'+str(i)+'f'+str(j)+'.npy')
                    #filter = np.load(path+str(i)+str(j)+'.npy')
                else:
                    filter = np.random.uniform(minWeight, maxWeight,size=(filterSize,filterSize))
                filterGroup.append(filter)
            filters.append(filterGroup)

        return filters




    def propagateForward(self, input):
        self.data = input
        self.inputSize = input[0].shape[0]
        self.z = []
        outputShape = int((self.inputSize - self.filterSize)/self.stride + 1)
        for (index,filterGroup) in enumerate(self.filters):

            self.z.append(np.zeros((outputShape,outputShape)))

            for i in range(0, outputShape):
                for j in range(0, outputShape):
                    for a in range(0, self.filterSize):
                        for b in range(0, self.filterSize):
                            for (d,x) in enumerate(input):

                                self.z[index][i,j] += filterGroup[d][a,b] * \
                                        x[i*self.stride + a, j*self.stride + b]

                    self.z[index][i,j] += self.bias[index]

        return self.activationFunction.activate(self.z)




    def propagateBackwards(self, errors, learningRate):

        weightErrors = [[]] * self.filterNumber
        previousErrors = []
        biasErrors = [0]* self.filterNumber
        outputShape = int((self.inputSize - self.filterSize)/self.stride + 1)

        for (d,x) in enumerate(self.data):

            previousErrors.append(np.zeros((self.inputSize,self.inputSize)))

            for (index, filterGroup) in enumerate(self.filters):

                weightErrors[index].append(np.zeros((self.filterSize,self.filterSize)))

                for i in range(0, outputShape):
                    for j in range(0, outputShape):

                        outputError = errors[index][i,j] * \
                            self.activationFunction.derived(self.z[index][i,j])

                        for a in range(0, self.filterSize):
                            for b in range(0, self.filterSize):

                                weightErrors[index][d][a,b] +=  \
                                    x[i+a, j+b] * outputError
                                previousErrors[d][i*self.stride +a, j*self.stride+b]\
                                    += outputError * self.filters[index][d][a,b]

                    biasErrors[index] += self.bias[index]*outputError

        self.correctWeights(weightErrors, biasErrors, learningRate)

        return previousErrors



    def correctWeights(self, weightErrors, biasErrors, learningRate):
        """
            Correct the values of the layers filters using given error values and
            a learning rate.
        """
        #print('Errors: ', weightErrors)
        for (index, filterGroup) in enumerate(self.filters):
            for (d, filter) in enumerate(filterGroup):
                for a in range(0, self.filterSize):
                    for b in range(0, self.filterSize):
                        #print('E:', learningRate * weightErrors[index][d][a,b])
                        #x = learningRate * weightErrors[index][d][a,b]
                        #if x > 0.1 : print(x)
                        filter[a,b] -= \
                            learningRate * weightErrors[index][d][a,b]

            self.bias[index] -= learningRate * biasErrors[index]



    def save(self, path):
        datautil.saveData(path, ['CONV', self.filterNumber, self.filterSize,\
        self.stride, self.inputDepth, self.activationFunction.getName()])
        for (i, filterGroup) in enumerate(self.filters):
            for (j, filter) in enumerate(filterGroup):
                np.save(path + 'g' + str(i) + 'f' + str(j), filter)
        np.save(path + 'b', self.bias)



    def load(path):
        data = datautil.loadData(path)
        filterNumber = int(data[1])
        filterSize = int(data[2])
        stride = int(data[3])
        inputDepth = int(data[4])
        activationFunction = getActivationFunction(data[5])
        filters = ConvolutionLayer\
            .initializeFilters(filterNumber, inputDepth, filterSize, path)
        bias = np.load(path +'b.npy')
        return ConvolutionLayer(filterNumber, filterSize, stride, activationFunction,\
             inputDepth, filters, bias)



class ExtremumPoolLayer(Layer):

    """
        Layer type that reduces the activation maps dimensions by taking the
        biggest or smallest value from a cluster.
    """

    comparationFunctions = { 'max': lambda x,y: x > y, 'min': lambda x,y: x < y, 'min\n': lambda x,y: x < y, "" : lambda x,y: x < y }

    def __init__(self, clusterSize = 2, type = 'min'):
        self.clusterSize = clusterSize
        self.dataLength = 0
        self.type = type
        self.comparationFunction = ExtremumPoolLayer.comparationFunctions[type]


    def propagateForward(self, input):

        self.dataLength = len(input)
        self.inputSize = input[0].shape[0]
        outputShape = self.inputSize - self.clusterSize + 1
        self.z = []
        self.extPositions = []


        for (d, data) in enumerate(input):

            self.z.append(np.empty((outputShape,outputShape)))
            self.extPositions.append([])

            for i in range(0, self.inputSize - self.clusterSize + 1 ):
                for j in range(0, self.inputSize - self.clusterSize + 1):

                    ext = data[i,j]
                    extPosition = (i,j)

                    for k in range(0, self.clusterSize):
                        for l in range(0, self.clusterSize):

                            if(self.comparationFunction(data[i+k, j+l], ext)):
                                ext = data[i+k, j+l]
                                extPosition = (i+k, j+l)

                    self.extPositions[d].append(extPosition)
                    self.z[d][i, j] = ext

        return self.z


    def propagateBackwards(self, errors, learningRate):

        previousErrors = []
        outputShape = self.inputSize - self.clusterSize + 1

        for d in range(0, self.dataLength):

            previousErrors.append(np.zeros((self.inputSize,self.inputSize)))

            for i in range(0, outputShape):
                for j in range(0, outputShape):

                    x, y = self.extPositions[d][i * outputShape + j]
                    previousErrors[d][x,y] += errors[d][i,j]

        return previousErrors

    def save(self, path):
        datautil.saveData(path, ['MAXP', self.clusterSize, self.type])

    def load(path):
        data = datautil.loadData(path)
        return ExtremumPoolLayer(int(data[1]), data[2])



class FlatteningLayer(Layer):

    """
        Layer type that redimensions a 2 or 3 dimensional entry into a 1D vector.
    """
    def __init__(self):
        self.inputSize = 0
        self.inputDepth = 0

    def propagateForward(self, input):

        self.inputDepth = len(input)
        self.inputSize = input[0].shape[0]

        flatLayers = []

        for layer in input:
            flatLayers.append(layer.flatten())

        return np.concatenate(input).ravel()

    def propagateBackwards(self, errors, learningRate):

        previousErrors = []
        layerLength = int(len(errors)/self.inputDepth)


        for layerIndex in range(0, self.inputDepth):
            layerErrors = \
                errors[layerIndex * layerLength: (layerIndex + 1) * layerLength]
            previousErrors.append\
                (np.resize(layerErrors,(self.inputSize, self.inputSize)))
        return previousErrors

    def save(self, path):
        datautil.saveData(path, ['FLAT'])

    def load(path):
        return FlatteningLayer()




class FullyConnectedLayer(Layer):

    """
        Layer type that uses a vector of weights to multiply the input vector.
    """

    def __init__(self, size, inputSize, activationFunction,  \
                    weights = None, bias = None, softmax = False):

        if weights is not None:
            self.weights = weights
        else :
            self.weights = FullyConnectedLayer.initializeWeights(size, inputSize)

        if bias is not None:
            self.bias = bias
        else :
            self.bias = np.random.uniform(minWeight, maxWeight,size=size)


        self.size = size
        self.inputSize = inputSize
        self.activationFunction = activationFunction
        self.softmax = softmax

    def initializeWeights(size, inputSize, path = None):
        """
            Initialize the needed ammount of weights with random numbers or
            with values loaded from the given path, if the path is not None.
        """
        weights = []
        for i in range(0, size):
            weightVector = None
            if path:
                weightVector = np.load(path + str(i) + '.npy')
            else :
                weightVector = np.random.uniform(minWeight, maxWeight,size=(inputSize,1))

            weights.append(weightVector)

        return weights

    def propagateForward(self, input):

        self.data = input
        self.z = np.empty(self.size)
        for i in range(self.size):
            self.z[i] = input.dot(self.weights[i]) + self.bias[i]

        return self.activationFunction.activate([self.z])[0]


    def propagateBackwards(self, errors, learningRate):

        weightErrors = []
        previousErrors = np.zeros(self.inputSize)
        biasErrors = [0] * self.size
        if self.softmax:
            outputError = np.matmul(self.activationFunction.derived(self.z), errors)
            print('OE: ', outputError)
        else:
            outputError = np.empty(self.size)
            for neuronIndex in range(0, self.size):
                outputError[neuronIndex] = errors[neuronIndex] * \
                        self.activationFunction.derived(self.z[neuronIndex])

        for neuronIndex in range(0, self.size):

            weightErrors.append(np.zeros(self.inputSize))
            for i in range(self.inputSize):

                weightErrors[neuronIndex][i] += \
                    self.data[i] * outputError[neuronIndex]
                previousErrors[i] += \
                    self.weights[neuronIndex][i] * outputError[neuronIndex]

            biasErrors += self.bias[neuronIndex] * outputError[neuronIndex]

        self.correctWeights(weightErrors, biasErrors, learningRate)
        return previousErrors

    def correctWeights(self, weightErrors, biasErrors, learningRate):
        """
            Correct the layers weights using given error values and learning rate.
        """
        #print('Errors FC: ', weightErrors)
        for neuronIndex in range(0, self.size):

            for i in range(0, self.inputSize):
                #x = learningRate * weightErrors[neuronIndex][i]
                #if x > 0.1 : print(x)
                self.weights[neuronIndex][i] -= \
                    learningRate * weightErrors[neuronIndex][i]

            self.bias[neuronIndex] -= \
                learningRate * biasErrors[neuronIndex]

    def save(self, path):
        datautil.saveData(path, ['FCON', self.size, self.inputSize, \
            self.activationFunction.getName(), int(self.softmax)])

        for (i, weight) in enumerate(self.weights):
            np.save(path + str(i), weight)

        np.save(path + 'b', self.bias)

    def load(path):
        data = datautil.loadData(path)
        size = int(data[1])
        inputSize = int(data[2])
        activationFunction = getActivationFunction(data[3])
        softmax = bool(int(data[4]))
        weights = FullyConnectedLayer.initializeWeights(size, inputSize, path)
        bias = np.load(path + 'b.npy')
        return FullyConnectedLayer(size, inputSize, activationFunction, \
            weights, bias, softmax)

    def __str__(self):
        return 'Fully Connected Layer\n size: ' + str(self.size) +'\n activation:' + \
                self.activationFunction.getName()

#dictionary used to map a layer to its short string code for saving and loading
#purpuses
layerMap = {'CONV': ConvolutionLayer, 'MAXP': ExtremumPoolLayer, \
        'FLAT': FlatteningLayer, 'FCON': FullyConnectedLayer}

def getLayerById(id):
    return layerMap[id]
