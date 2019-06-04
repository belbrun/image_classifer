from network import *
from layers import *
from datautil import *
from functions import crossEntropyLoss

blastomCount = 130
othersCount = 130

datasetSize = 260

trainingSetFactor = 0.7
validationSetFactor = 0.1
testSetFactor = 0.3

epochs = 20


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
networkPath = 'network_data/configuration6/'

def initializeNN():
    neuralNet = NeuralNetwork(crossEntropyLoss)
    neuralNet.addLayer(ConvolutionLayer\
    (filterNumber = 3, filterSize = 5, stride = 2, activationFunction = Sigmoid(), inputDepth = 1))
    neuralNet.addLayer(ExtremumPoolLayer(4, 'min'))
    neuralNet.addLayer(ConvolutionLayer(4,4,2,Sigmoid(),3))
    neuralNet.addLayer(ExtremumPoolLayer(3, 'min'))
    neuralNet.addLayer(ConvolutionLayer(5,4,1,Sigmoid(),4))
    neuralNet.addLayer(ExtremumPoolLayer(3, 'min'))
    neuralNet.addLayer(ConvolutionLayer(8,3,1,Sigmoid(),5))
    neuralNet.addLayer(ExtremumPoolLayer(2, 'min'))
    neuralNet.addLayer(ConvolutionLayer(10,3,1,Sigmoid(),8))
    neuralNet.addLayer(ExtremumPoolLayer(2, 'min'))
    neuralNet.addLayer(ConvolutionLayer(20,3,1,Sigmoid(),10))
    neuralNet.addLayer(ExtremumPoolLayer(2, 'min'))
    neuralNet.addLayer(FlatteningLayer())
    neuralNet.addLayer(FullyConnectedLayer(200, 500, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(80, 200, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(40, 80, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(20, 40, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(10, 20, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(5, 10, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(2, 5, Sigmoid()))
    neuralNet.addLayer(FullyConnectedLayer(1, 2, Sigmoid()))

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
    return datautil.getInput(name, datasetPath, gray, shape, rotations, avaraged)




def trainOnEntity(neuralNet, index, isBlastom, learningRate):
    results = blastomResults if isBlastom else othersResults
    index += 0 if isBlastom else blastomCount
    errors = []
    for rotation in getEntity(index, isBlastom):
        errors.append(neuralNet.train(rotation, results, learningRate))
    return errors

def feedEntity(neuralNet, index, isBlastom):
    results = blastomResults if isBlastom else othersResults
    index += 0 if isBlastom else blastomCount
    return neuralNet.feedForError(getEntity(index, isBlastom)[0], results)

def testEntity(neuralNet, index, isBlastom):
    index += 0 if isBlastom else blastomCount
    return neuralNet.classify(getEntity(index, isBlastom)[0])

def formatMessage(action, epoch, error, index = 0, isBlastom = True, output = None, isAvarage = False, rotation = False):

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
    drop = 0.05
    for i in range(1, epochs + 1):

        avgError = 0
        for index in range(1, trainingSetSize):
            errors = trainOnEntity(neuralNet, index, True, learningRate)
            rotationDirections = ['U', 'R', 'D', 'L'] if rotations else ['U']

            for r, rotation in enumerate(rotationDirections):
                avgError += abs(errors[r])
                trainingLog.append(formatMessage('TRAINING', i, errors[r], index))
                print(trainingLog[-1])

            errors = trainOnEntity(neuralNet, index, False, learningRate)
            for r, rotation in enumerate(rotationDirections):
                trainingLog.append(formatMessage('TRAINING', i, errors[r], index, False))
                print(trainingLog[-1])
                avgError += abs(errors[r])

        trainingLog.append(formatMessage('TRAINING', i, \
             avgError/(trainingSetSize*2), isAvarage = True))
        print(trainingLog[-1])

        avgError = 0
        blastomResults = []
        otherResults = []

        for index in range(trainingSetSize, trainingSetSize + validationSetSize):
            output, error = feedEntity(neuralNet, index, True)
            blastomResults.append(output)
            avgError += abs(error)
            trainingLog.append(formatMessage('VALIDATION', i, error, index, True, output))
            print(trainingLog[-1])
            output, error = feedEntity(neuralNet, index, False)
            otherResults.append(output)
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

        learningRate -= drop

    return trainingLog



    #limit = 0.493338599275 new network 68%
    #limit = 0.48971075
    #limit = 0.4961616
    #limit = 0.4976952 #epoch3
    #limit = 0.498102467 #epoch4
    #limit = 0.4982373
    #limit = 0.49677685
    #limit = 0.500002
    #limit = 0.49941376 #e20 62% conf1/2 ?
    #limit = 0.4994173
    #limit = 0.499426123 #e23 64% conf1/2?
    #limit = 0.49943805
    #limit = 0.4972952
    #limit = 0.49808655 conf3 e10
    #limit =  0.49856540954

def findClassificationLimit(epoch, startIndex = 1, endIndex = 10):
    neuralNet = NeuralNetwork.load(networkPath + 'epoch' + str(epoch) + '/', False)
    blastomResults = []
    otherResults = []
    for index in range(startIndex, endIndex):
        output, error = feedEntity(neuralNet, index, True)
        blastomResults.append(output)
        output, error = feedEntity(neuralNet, index, False)
        otherResults.append(output)
    results, limit = \
            NeuralNetwork.calculateClassificationLimit(blastomResults, otherResults)
    print(results, limit)

def test(neuralNet, startIndex = 90, endIndex = 130):
    counts = [0,0,0] #correct, false positives, false negatives
    print(neuralNet.classificationLimit)
    for index in range(startIndex, endIndex):
        correct = bool(testEntity(neuralNet, index, True))

        if correct:
            counts[0] += 1
        else:
            counts[2] += 1
        print('TEST----Example: ', index, '_1 Output: ', correct, ' Overall: ', counts)

        correct = not bool(testEntity(neuralNet, index, False))
        if correct:
            counts[0] += 1
        else:
            counts[1] += 1
        print('TEST----Example: ', index, '_0 Output: ', correct, ' Overall: ', counts)
    print([round(x/sum(counts)*100, 2) for x in counts])

def testingProcedure():
    neuralNet = NeuralNetwork.load(networkPath + 'epoch20/')
    test(neuralNet)
    print('RESULTS ON TRAINING SET:')
    test(neuralNet, 1, 10)


def trainingProcedure(new = True):
    if new :
        output = 0
        while output > 0.505 or output < 0.495:
            neuralNet = initializeNN()
            output = neuralNet.output(getEntity(1, True)[0])
    else :
        neuralNet = NeuralNetwork.load(networkPath + 'epoch22/')
    log = train(neuralNet)
    #neuralNet.save('network_data/new_network/')
    #datautil.writeLog('network_data/new_network/', log)

def printNetwork(path):
    NeuralNetwork.load(path).printNetwork()

def main():
    #testingProcedure()
    #trainingProcedure()
    #printNetwork('network_data/new_network/')
    findClassificationLimit(13)

if __name__ == '__main__':
    main()
