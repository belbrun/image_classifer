from network import *
from layers import *
from datautil import *

blastomCount = 130
othersCount = 130

datasetSize = 260

trainingSetFactor = 0.6
validationSetFactor = 0.2
testSetFactor = 0.2

epochs = 10
learningRate = 0.2
drop = 0.1


blastomResults = [1,0]
othersResults = [0,1]

datasetPath = 'dataset/2/ALL_IDB2/img/'

def initializeNN():
    neuralNet = NeuralNetwork()
    #neuralNet.addLayer(ConvolutionLaye)
    neuralNet.addLayer(ConvolutionLayer(3,5,1,TanHiperbolic(),3))
    neuralNet.addLayer(MaxpoolLayer(5))
    neuralNet.addLayer(ConvolutionLayer(2,4,1,ReLU(),3))
    neuralNet.addLayer(MaxpoolLayer(3))
    neuralNet.addLayer(ConvolutionLayer(3,3,1,ReLU(),2))
    neuralNet.addLayer(MaxpoolLayer(2))
    neuralNet.addLayer(FlatteningLayer())
    neuralNet.addLayer(FullyConnectedLayer(100, 174243, TanHiperbolic()))
    neuralNet.addLayer(FullyConnectedLayer(25, 100, TanHiperbolic()))
    neuralNet.addLayer(FullyConnectedLayer(2, 25, Sigmoid()))
    return neuralNet

def fillIndex(index):
    if index < 10:
        return 'Im00' + str(index)
    elif index < 100:
        return 'Im0' + str(index)
    else:
        return 'Im' + str(index)

def getEntity(index, isBlastom):
    name = fillIndex(index) + '_1.tif' if isBlastom  else \
    fillIndex(index) + '_0.tif'
    return datautil.getInput(name)


def trainOnEntity(neuralNet, index, isBlastom):
    results = blastomResults if isBlastom else othersResults
    index += 0 if isBlastom else blastomCount
    return neuralNet.train(getEntity(index, isBlastom), results, learningRate)

def feedEntity(neuralNet, index, isBlastom):
    results = blastomResults if isBlastom else othersResults
    index += 0 if isBlastom else blastomCount
    return neuralNet.feedForError(getEntity(getEntity(index, isBlastom), results))

def train(neuralNet):
    trainingSetSize = int(round(datasetSize * trainingSetFactor / 2))
    validationSetSize = int(round(datasetSize * validationSetFactor))

    for i in range(1, epochs + 1):

        avgError = 0
        for index in range(1, trainingSetSize):
            error = trainOnEntity(neuralNet, index, True)
            avgError += error
            print('TRAINING -- Epoch: ', i, ' Example: ', index, ' Error: ', error)
            error = trainOnEntity(neuralNet, index, False)
            print('TRAINING -- Epoch: ', i, ' Example: ', index, ' Error: ', error)
            avgError += error

        print('TRAINING -- Epoch: ', i, ' Avarage error:', avgError/(trainingSetSize - 1))

        avgError = 0
        for index in range(trainingSetSize, trainingSetSize + validationSetSize):
            error = feedEntity(neuralNet, index, True)
            avgError += error
            print('VALIDATION -- Epoch: ', i, ' Example: ', index, ' Error: ', error)
            error = feedEntity(neuralNet, index, False)
            print('VALIDATION -- Epoch: ', i, ' Example: ', index, ' Error: ', error)
            avgError += error

        print('TRAINING -- Epoch: ', i, ' Avarage error:', avgError/(validationSetSize - 1))



def main():
    neuralNet = initializeNN()
    train(neuralNet)




if __name__ == '__main__':
    main()
