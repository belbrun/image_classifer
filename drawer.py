import datautil
import matplotlib.pyplot as plt

def getResults(path):
    results = datautil.readResults(path)
    epochResults = []
    epochFP = []

    for result in results[0:-1]:
        splitResults = result[1:-1].split(',')
        epochResults.append(float(splitResults[0]))
        epochFP.append(float(splitResults[1]))

    return (epochResults, epochFP)

def plotResultsByEpoch(path):
    epochResults, epochFP = getResults(path)
    xValues = list(range(1, len(epochResults) + 1))

    plt.plot(xValues, epochResults, label = 'Postotak točno klasificiranih primjera')
    plt.plot(xValues, epochFP, label = 'Postotak lažno pozitivno klasificiranih primjera')
    plt.title('Rezultati klasifikacije na skupu za testiranje po epohama')
    plt.legend()

    plt.show()
    #plt.savefig(path + 'results_plot.png')

def main():
    plotResultsByEpoch('network_data/configuration10/')

if __name__ == '__main__':
    main()
