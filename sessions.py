

class Session:

    def __init__(self, neuralNet):
        self.neuralNet = neuralNet

    def start():
        pass

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
        return datautil.
            getInput(name, datasetPath, gray, shape, rotations, avaraged)

class TrainingSession(Session):

    def __init__(datasetSize, trainingSetFactor, validationSetFactor
            learningRate = 0.01, drop = 1, startEpoch = 0, epochs = 0
            gray = False, shape = False, rotations = False, avaraged = False):
        self.startEpoch = startEpoch
        self.epochs = epochs
        self.drop = drop
        self.learningRate = learningRate
        self.trainingLog = []
        self.trainingSetSize = int(round(datasetSize * trainingSetFactor / 2))
        self.validationSetSize = int(round(datasetSize * validationSetFactor/2))
        self.gray = gray
        self.shape = shape
        self.rotations = rotations
        self.avaraged = avaraged

    def initializeNN(neuralNet):
        self.neuralNet = neuralNet

    def loadNeuralNet(path):
        self.neuralNet = NeuralNetwork.load(path)

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

    def train():
        avgError = 0
        for index in range(1, self.trainingSetSize):
            errors = trainOnEntity(self.neuralNet, index, True, self.learningRate)
            rotationDirections = ['U', 'R', 'D', 'L'] if self.rotations else ['U']

            for r, rotation in enumerate(rotationDirections):
                avgError += abs(errors[r])
                self.trainingLog.append(formatMessage('TRAINING', i, errors[r], index))
                print(self.trainingLog[-1])

            errors = trainOnEntity(neuralNet, index, False, learningRate)
            for r, rotation in enumerate(rotationDirections):
                self.trainingLog.append(formatMessage('TRAINING', i, errors[r], index, False))
                print(self.trainingLog[-1])
                avgError += abs(errors[r])

        self.trainingLog.append(formatMessage('TRAINING', i, \
             avgError/(trainingSetSize*2), isAvarage = True))
        print(self.trainingLog[-1])

    def validate():

        avgError = 0
        blastomResults = []
        otherResults = []

        for index in range(self.trainingSetSize, self.trainingSetSize + self.validationSetSize):
            output, error = feedEntity(neuralNet, index, True)
            blastomResults.append(output[0])
            avgError += abs(error)
            self.trainingLog.append(formatMessage('VALIDATION', i, error, index, True, output))
            print(self.trainingLog[-1])
            output, error = feedEntity(neuralNet, index, False)
            otherResults.append(output[0])
            self.trainingLog.append(formatMessage('VALIDATION', i, error, index, False, output))
            print(self.trainingLog[-1])
            avgError += abs(error)

    def process(epoch):
        limit, results = \
        NeuralNetwork.calculateClassificationLimit(blastomResults, otherResults)
        currentAvgError =  avgError/(validationSetSize*2)
        self.trainingLog.append\
        ('VALIDATION RESULTS : ' + str(results) + 'LIMIT: ' + str(limit))
        print(self.trainingLog[-1])
        self.trainingLog.append(formatMessage('VALIDATION', i, \
            currentAvgError, isAvarage = True))
        print(self.trainingLog[-1])
        neuralNet.setClassificationLimit(limit)

        print('Test on training set')
        self.trainingLog.append(str(test(neuralNet, 1, 10)))
        print(self.trainingLog[-1])
        saveEpoch(self.neuralNet, 0, data = [str(self.gray), \
            str(self.avaraged), str(self.shape), \
            str(self.trainingSetFactor), str(self.validationSetFactor)])
        saveEpoch(neuralNet, epoch, self.trainingLog, \
        [str(gray), str(avaraged), str(shape), str(trainingSetFactor), \
        str(validationSetFactor)])

    def start():

        print('Training set size: ', self.trainingSetSize, 'validationSetSize: ', self.validationSetSize)

        if self.startEpoch == 1:
            saveEpoch(self.neuralNet, 0, data = [str(self.gray), \
                str(self.avaraged), str(self.shape), \
                str(self.trainingSetFactor), str(self.validationSetFactor)])

        lastAvgError = None
        for i in range(self.startEpoch, self.epochs + 1):

            train()
            validate()
            process(i)
            learningRate *= drop

        return self.trainingLog
