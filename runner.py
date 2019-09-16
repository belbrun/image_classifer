from sessions import *

#dataset info
blastomCount = 130
othersCount = 130

datasetSize = 260

#dataset division
trainingSetFactor = 0.5
validationSetFactor = 0.2
testSetFactor = 0.3


#training factors
drop = 1
learningRate = 0.002
epochs = 15
startEpoch = 7



#preprocessing factors
gray = True
avaraged = True
shape = 200,200
rotations = True

#blastomResults = np.array([1, 0])
#othersResults = np.array([0,1])
blastomResults = np.array([1])
othersResults = np.array([0])

datasetPath = 'dataset/2/ALL_IDB2/img/'
networkPath = 'network_data/configuration15/'

def initializeNN():
    neuralNet = NeuralNetwork()
    neuralNet.addLayer(ConvolutionLayer\
    (filterNumber = 10, filterSize = 6, stride = 2, activationFunction = Sigmoid(), inputDepth = 1))
    neuralNet.addLayer(ExtremumPoolLayer(4, 'min'))
    neuralNet.addLayer(ConvolutionLayer(30,5,2,Sigmoid(),10))
    neuralNet.addLayer(ExtremumPoolLayer(4, 'min'))
    neuralNet.addLayer(ConvolutionLayer(40,4,1,Sigmoid(),30))
    neuralNet.addLayer(ExtremumPoolLayer(3, 'min'))
    neuralNet.addLayer(ConvolutionLayer(50,4,1,Sigmoid(),40))
    neuralNet.addLayer(ExtremumPoolLayer(3, 'min'))
    neuralNet.addLayer(ConvolutionLayer(60,3,1,Sigmoid(),50))
    neuralNet.addLayer(ExtremumPoolLayer(2, 'min'))
    neuralNet.addLayer(ConvolutionLayer(70,3,1,Sigmoid(),60))
    neuralNet.addLayer(ExtremumPoolLayer(2, 'min'))
    neuralNet.addLayer(FlatteningLayer())
    neuralNet.addLayer(FullyConnectedLayer(50, 280, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(10, 50, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(4, 10, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(1, 4, Sigmoid()))

    return neuralNet

def train():
    #create training session
    session = TrainingSession(datasetSize, trainingSetFactor, validationSetFactor,
        blastomResults, othersResults, datasetPath, learningRate, drop,
        startEpoch, epochs, gray, shape, rotations, avaraged)

    #initialize neural network
    session.setNeuralNet(initializeNN())

    #load neural network
    #session.loadNeuralNet(networkPath)

    #start training session
    trainingLog = session.start()

def test():
    pass


def main():

    #train or test

    train()
    #test()

if __name__ == '__main__':
    main()
