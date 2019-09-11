import datautil
import parser
import trainer
from network import NeuralNetwork
import matplotlib.pyplot as plt


#TODO: get log results from configuration 10 and plot validation set results and training set results



def plot(title, xValues, data, xLabel = 'epohe', yLabel = 'postotci'):
    for yValues, label in data:
        plt.plot(xValues, yValues, label = label)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.show()

def plotResultsByEpoch(path):
    title = 'Rezultati klasifikacije na skupu za testiranje po epohama'
    epochResults, epochFP, epochFN = getTestResults(path)
    xValues = list(range(1, len(epochResults) + 1))
    data = [
        (epochResults, 'Postotak točno klasificiranih primjera')
        #(epochFP, 'Postotak lažno pozitivno klasificiranih primjera'),
        #(epochFN, 'Postotak lažno negativno klasificiranih primjera')
    ]
    plot(title, xValues, data)



def plotTrainingAndValidation(path):
    title = 'Postotak točne klasifikacije na skupu za validaciju i treniranje'
    validationResults = datautil.getValidationResults(path + 'epoch9/')[0]
    trainingResults = datautil.getTrainingResults(path + 'epoch9/')[0]
    xValues = list(range(1, len(validationResults) + 1))
    data = [
        (validationResults, 'Set za validaciju'),
        (trainingResults, 'Set za treniranje')
    ]
    plot(title, xValues, data)

def plotClassificationLimits(path, epoch):
    epochPath = path + 'epoch' + str(epoch) + '/'
    neuralNet = NeuralNetwork.load(epochPath)
    blastomResults, otherResults = \
        datautil.getEpochValidationOutputs(epochPath, epoch)
    results, limits = \
        NeuralNetwork.calculateClassificationLimit(blastomResults, otherResults, forPlotting = True)
    epochResults, epochFP, epochFN = parser.splitResults(results)
    data = [
        (epochResults, 'Postotak točno klasificiranih primjera'),
        (epochFP, 'Postotak lažno pozitivno klasificiranih primjera'),
        (epochFN, 'Postotak lažno negativno klasificiranih primjera')
    ]
    title = 'Postotci točno, lažno pozitivno i lažno negativno klasificiranih primjera tijekom traženja klasifikacijske granice'
    xLabel = 'Klasifikacijske granice'
    plot(title, limits, data, xLabel)

def main():
    #plotResultsByEpoch('network_data/configuration13/')
    plotTrainingAndValidation('network_data/configuration10/')
    #plotClassificationLimits('network_data/configuration10/', 2)

if __name__ == '__main__':
    main()
