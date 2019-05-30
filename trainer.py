from network import *
from layers import *
from datautil import *
from functions import crossEntropyLoss

blastomCount = 130
othersCount = 130

datasetSize = 260

trainingSetFactor = 0.5
validationSetFactor = 0.1
testSetFactor = 0.3

epochs = 30


#preprocessing factors
gray = True
avaraged = True
shape = 200,200


#blastomResults = np.array([1, 0])
#othersResults = np.array([0,1])
blastomResults = np.array([1])
othersResults = np.array([0])

datasetPath = 'dataset/2/ALL_IDB2/img/'
networkPath = 'network_data/new_network/'

def initializeNN():
    neuralNet = NeuralNetwork(crossEntropyLoss)
    neuralNet.addLayer(ConvolutionLayer\
    (filterNumber = 3, filterSize = 5, stride = 2, activationFunction = Sigmoid(), inputDepth = 1))
    neuralNet.addLayer(ExtremumPoolLayer(4, 'min'))
    neuralNet.addLayer(ConvolutionLayer(2,4,2,Sigmoid(),3))
    neuralNet.addLayer(ExtremumPoolLayer(3, 'min'))
    neuralNet.addLayer(ConvolutionLayer(3,3,1,Sigmoid(),2))
    neuralNet.addLayer(ExtremumPoolLayer(2, 'min'))
    neuralNet.addLayer(FlatteningLayer())
    neuralNet.addLayer(FullyConnectedLayer(150, 768, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(50, 150, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(10, 50, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(3, 10, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(1, 3, Sigmoid(), softmax = False))
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
    entity = datautil.getInput(name, datasetPath, gray, shape, avaraged)
    #print('ENTITIY: ', entity)
    return entity


def trainOnEntity(neuralNet, index, isBlastom, learningRate):
    results = blastomResults if isBlastom else othersResults
    index += 0 if isBlastom else blastomCount
    return neuralNet.train(getEntity(index, isBlastom), results, learningRate)

def feedEntity(neuralNet, index, isBlastom):
    results = blastomResults if isBlastom else othersResults
    index += 0 if isBlastom else blastomCount
    return neuralNet.feedForError(getEntity(index, isBlastom), results)

def testEntity(neuralNet, index, isBlastom):
    index += 0 if isBlastom else blastomCount
    return neuralNet.output(getEntity(index, isBlastom))

def formatMessage(action, epoch, error, index = 0, isBlastom = True, output = None, isAvarage = False):
    message = ''
    message += action + '-- Epoch:' + str(epoch)
    message += ' Example: ' + str(index) if not isAvarage else \
            'AvarageError: ' + str(error)
    if not isAvarage:
        message += '_1 Error: ' + str(error) if isBlastom else \
                '_0 Error: ' + str(error)
    if output:
        message += ' Output: ' + str(output[0])
    return message

def saveEpoch(neuralNet, epoch, log):
    path = networkPath + 'epoch' + str(epoch) + '/'
    datautil.makeDirectory(path)
    neuralNet.save(path)
    datautil.writeLog(path, log)

def train(neuralNet):
    trainingSetSize = int(round(datasetSize * trainingSetFactor / 2))
    validationSetSize = int(round(datasetSize * validationSetFactor/2))
    trainingLog = []
    lastAvgError = None
    learningRate = 2.2
    drop = 0.1
    for i in range(21, epochs + 1):

        avgError = 0
        for index in range(1, trainingSetSize):
            error = trainOnEntity(neuralNet, index, True, learningRate)
            avgError += abs(error)
            trainingLog.append(formatMessage('TRAINING', i, error, index))
            print(trainingLog[-1])
            error = trainOnEntity(neuralNet, index, False, learningRate)
            trainingLog.append(formatMessage('TRAINING', i, error, index, False))
            print(trainingLog[-1])
            avgError += abs(error)

        trainingLog.append(formatMessage('TRAINING', i, \
             avgError/(trainingSetSize*2), isAvarage = True))
        print(trainingLog[-1])

        avgError = 0
        for index in range(trainingSetSize, trainingSetSize + validationSetSize):
            output, error = feedEntity(neuralNet, index, True)
            avgError += abs(error)
            trainingLog.append(formatMessage('VALIDATION', i, error, index, True, output))
            print(trainingLog[-1])
            output, error = feedEntity(neuralNet, index, False)
            trainingLog.append(formatMessage('VALIDATION', i, error, index, False, output))
            print(trainingLog[-1])
            avgError += abs(error)

        currentAvgError =  avgError/(validationSetSize*2)
        trainingLog.append(formatMessage('VALIDATION', i, \
            currentAvgError, isAvarage = True))
        print(trainingLog[-1])
        saveEpoch(neuralNet, i, trainingLog)
        if lastAvgError and currentAvgError * 0.9 > lastAvgError:
            break
        else :
            lastAvgError = currentAvgError

        learningRate -= 0.1

    return trainingLog

def isCorrect(value, isBlastom):

    #limit = 0.493338599275 new network 68%
    #limit = 0.48971075
    #limit = 0.4961616
    #limit = 0.4976952 #epoch3
    #limit = 0.498102467 #epoch4
    #limit = 0.4982373
    #limit = 0.49677685
    #limit = 0.500002
    #limit = 0.49941376 #e20

    print(value, limit)
    if value == limit: return None
    correct = value > limit if isBlastom else value < limit
    return correct


def test(neuralNet):
    counts = [0,0,0,0] #correct, false positives, false negatives, inconclusive
    avgValue = 0
    startIndex = 90
    endIndex = 130

    for index in range(startIndex, endIndex):
        output = testEntity(neuralNet, index, True)
        avgValue += output[0]
        correct = isCorrect(output[0], True)
        if correct:
            counts[0] += 1
        elif correct is not None:
            counts[2] += 1
        else:
            counts[3] += 1
        print('TEST----Example: ', index, '_1 Output: ', output, ' Overall: ', counts)

        output = testEntity(neuralNet, index, False)
        avgValue += output[0]
        correct = isCorrect(output[0], False)
        if correct:
            counts[0] += 1
        elif correct is not None:
            counts[1] += 1
        else:
            counts[3] += 1
        print('TEST----Example: ', index, '_0 Output: ', output, ' Overall: ', counts)
    print(avgValue/(2*(endIndex-startIndex)))

def testingProcedure():
    neuralNet = NeuralNetwork.load('network_data/new_network/epoch20/')
    test(neuralNet)

def trainingProcedure(new = True):
    if new :
        output = 0
        while output > 0.505 or output < 0.495:
            neuralNet = initializeNN()
            output = neuralNet.output(getEntity(1, True))
    else :
        neuralNet = NeuralNetwork.load('network_data/new_network/epoch20/')
    log = train(neuralNet)
    #neuralNet.save('network_data/new_network/')
    #datautil.writeLog('network_data/new_network/', log)

def printNetwork(path):
    NeuralNetwork.load(path).printNetwork()

def main():
    #testingProcedure()
    trainingProcedure(False)
    #printNetwork('network_data/new_network/')

if __name__ == '__main__':
    main()
