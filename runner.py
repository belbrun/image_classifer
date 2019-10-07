from sessions import *

#dataset info
datasetSize = 60000

#dataset division
trainingSetFactor = 0.8
validationSetFactor = 0.2


#training factors
drop = 1
learningRate = 0.002
epochs = 15
startEpoch = 1

datasetPath = 'dataset/2/ALL_IDB2/img/'
networkPath = 'network_data/configuration15/'

def main():

    #train or test

    train()
    #test()

def train():
    #create training session
    session = TrainingSession(datasetSize, trainingSetFactor, validationSetFactor,
         datasetPath, learningRate, drop, startEpoch, epochs)

    #initialize neural network
    session.setNeuralNet(initializeNN())

    #load neural network
    #session.loadNeuralNet(networkPath)

    #start training session
    trainingLog = session.start()

def test():
    #create testing session
    session = TestingSession(datasetPath, 100, 130, blastomCount, 1, gray, shape,
        rotations, avaraged)

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
