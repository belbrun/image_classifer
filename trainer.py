


def initializeNN2():
    neuralNet = NeuralNetwork()
    neuralNet.addLayer(ConvolutionLayer(15,5,10,Sigmoid(),3))
    neuralNet.addLayer(ExtremumPoolLayer(2, 'min'))
    neuralNet.addLayer(ConvolutionLayer(40,5, 2,Sigmoid(),15))
    neuralNet.addLayer(ExtremumPoolLayer(2, 'min'))
    neuralNet.addLayer(FlatteningLayer())
    neuralNet.addLayer(FullyConnectedLayer(1, 160, Sigmoid()))
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

def saveEpoch(neuralNet, epoch, log = None, data = None):
    path = networkPath + 'epoch' + str(epoch) + '/'
    datautil.makeDirectory(path)
    neuralNet.save(path)
    if log:
        datautil.writeLog(path, log)

def train(neuralNet):
    #makee training module indepentend with a class training session

    startEpoch = 7
    drop = 1
    learningRate = 0.002
    trainingLog = []

    trainingSetSize = int(round(datasetSize * trainingSetFactor / 2))
    validationSetSize = int(round(datasetSize * validationSetFactor/2))
    print('Training set size: ', trainingSetSize, 'validationSetSize: ', validationSetSize)

    if startEpoch == 1:
        saveEpoch(neuralNet, 0, data = [str(gray), str(avaraged), str(shape), \
            str(trainingSetFactor), str(validationSetFactor)])

    lastAvgError = None
    for i in range(startEpoch, epochs + 1):

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
            blastomResults.append(output[0])
            avgError += abs(error)
            trainingLog.append(formatMessage('VALIDATION', i, error, index, True, output))
            print(trainingLog[-1])
            output, error = feedEntity(neuralNet, index, False)
            otherResults.append(output[0])
            trainingLog.append(formatMessage('VALIDATION', i, error, index, False, output))
            print(trainingLog[-1])
            avgError += abs(error)

        limit, results = \
        NeuralNetwork.calculateClassificationLimit(blastomResults, otherResults)
        currentAvgError =  avgError/(validationSetSize*2)
        trainingLog.append\
        ('VALIDATION RESULTS : ' + str(results) + 'LIMIT: ' + str(limit))
        print(trainingLog[-1])
        trainingLog.append(formatMessage('VALIDATION', i, \
            currentAvgError, isAvarage = True))
        print(trainingLog[-1])
        neuralNet.setClassificationLimit(limit)

        print('Test on training set')
        trainingLog.append(str(test(neuralNet, 1, 10)))
        print(trainingLog[-1])
        saveEpoch(neuralNet, i, trainingLog, \
        [str(gray), str(avaraged), str(shape), str(trainingSetFactor), \
        str(validationSetFactor)])
        if lastAvgError and currentAvgError * 0.9 > lastAvgError:
            break
        else :
            lastAvgError = currentAvgError

        learningRate *= drop

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

def findClassificationLimit(neuralNet, startIndex = 1, endIndex = 10, forPlotting = False):
    blastomResults = []
    otherResults = []
    for index in range(startIndex, endIndex):
        output, error = feedEntity(neuralNet, index, True)
        blastomResults.append(output)
        output, error = feedEntity(neuralNet, index, False)
        otherResults.append(output)
    return NeuralNetwork.calculateClassificationLimit(blastomResults, otherResults, forPlotting = forPlotting)


def test(neuralNet, startIndex = 90, endIndex = 130):
    counts = [0,0,0] #correct, false positives, false negatives
    for index in range(startIndex, endIndex + 1):
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
    return [round(x/sum(counts)*100, 2) for x in counts]

def testingProcedure():
    neuralNet = NeuralNetwork.load(networkPath + 'epoch8/')
    test(neuralNet)
    print('RESULTS ON TRAINING SET:')
    test(neuralNet, 1, 10)


def trainingProcedure(new = True):
    if new :
        avgOutput = 0
        while avgOutput > 0.51 or avgOutput < 0.49:
            neuralNet = initializeNN()
            outputs =[]
            for i in range(1,3):
                outputs.append(neuralNet.output(getEntity(i, True)[0]))
                print(outputs)
            avgOutput = sum(outputs)/len(outputs)
    else :
        neuralNet = NeuralNetwork.load(networkPath + 'epoch6/')
    log = train(neuralNet)
    #neuralNet.save('network_data/new_network/')
    #datautil.writeLog('network_data/new_network/', log)

def findClassificationLimitsForConfiguration(configuration):
    networkPath = 'network_data/' + configuration + '/'
    epochs = 9
    valStart = 1
    valEnd = int(testSetFactor*datasetSize/2)
    for i in range(4, epochs + 1):
        path = networkPath + 'epoch' + str(i) + '/'
        neuralNet = NeuralNetwork.load(path, False)
        limit, results = findClassificationLimit(neuralNet, valStart, valEnd)
        print(results)
        data = [results, round(limit[0], 15)]
        datautil.saveData(path, data)

def testAllEpochs(configuration, validation = True):
    networkPath = 'network_data/' + configuration + '/'
    epochs = 11
    results = []
    validationStartIndex = int(trainingSetFactor * blastomCount)
    trainingStartIndex = int((trainingSetFactor + validationSetFactor) * blastomCount)
    start = validationStartIndex if validation else trainingStartIndex
    end = trainingStartIndex if validation else int(datasetSize/2)

    for i in range(4, epochs + 1):
        path = networkPath + 'epoch' + str(i) + '/'
        neuralNet = NeuralNetwork.load(path)
        results.append(test(neuralNet, start, end))

    if not validation:
        datautil.writeResults(networkPath, results)
    else:
        print(results)

def printNetwork(path):
    NeuralNetwork.load(path).printNetwork()



def main():
    #testingProcedure()
    #trainingProcedure(False)
    #printNetwork('network_data/new_network/')
    #findClassificationLimit(13)
    #findClassificationLimitsForConfiguration('configuration7')
    testAllEpochs('configuration15', False)

if __name__ == '__main__':
    main()
