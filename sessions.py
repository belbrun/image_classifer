

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

    def __init__(self):
        pass


    def trainOnEntity(neuralNet, index, isBlastom, learningRate):
        results = blastomResults if isBlastom else othersResults
        index += 0 if isBlastom else blastomCount
        errors = []
        for rotation in getEntity(index, isBlastom):
            errors.append(neuralNet.train(rotation, results, learningRate))
        return errors

    def start():
        
