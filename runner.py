from sessions import *

# dataset info
datasetSize = 20000

# dataset division
trainingSetFactor = 0.95
validationSetFactor = 0.05


# training factors
drop = 0.99
learningRate = 0.01
epochs = 10
startEpoch = 1

datasetPath = 'dataset/2/ALL_IDB2/img/'
networkPath = 'network_data/mnist/'

def main():

    #train or test

    #train()
    test()

def train():
    #create training session
    session = TrainingSession(datasetSize, trainingSetFactor, validationSetFactor,
         datasetPath, networkPath, learningRate, drop, startEpoch, epochs)

    #initialize neural network
    session.setNeuralNet(initializeNN())

    #load neural network
    #session.loadNeuralNet(networkPath)

    #start training session
    trainingLog = session.start()

def test():
    #create testing session
    session = TestingSession(numOfExamples = 100)

    #initialize neural network
    #session.setNeuralNet(initializeNN())

    #load neural network
    session.loadNeuralNet(networkPath, False)

    results = session.start()


def initializeNN():
    neuralNet = NeuralNetwork()
    neuralNet.addLayer(ConvolutionLayer\
    (filterNumber = 5, filterSize = 4, stride = 2, activationFunction = Sigmoid(), inputDepth = 1))
    neuralNet.addLayer(ExtremumPoolLayer(4, 'min'))
    neuralNet.addLayer(FlatteningLayer())
    neuralNet.addLayer(FullyConnectedLayer(100, 500, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(30, 100, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(10, 30, SoftMax(), softmax = True))
    return neuralNet



if __name__ == '__main__':
    main()
