from network import *

def calculateClassificationLimit(blastomResults, otherResults, forceFP = False, forPlotting = False):
    #find median without outliers?
    blastomAvarage = sum(blastomResults)/len(blastomResults)
    otherAvarage = sum(otherResults)/len(blastomResults)
    average = (blastomAvarage + otherAvarage)/ 2
    difference = abs(blastomAvarage - otherAvarage)/1000
    top = max([max(otherResults), max(blastomResults)]) - difference
    limit = min([min(otherResults), min(blastomResults)]) + difference
    bestResults = [0,0,0]
    bestLimit = 0
    equalLimits = []

    #for the purpuse of logging
    resultsList = []
    limitsList = []

    while limit < top:

        results = getResultsForLimit(blastomResults, otherResults, limit)
        resultsList.append([round(x/sum(results)*100, 2) for x in results])
        limitsList.append(limit)

        isBetterResult = results[0] > bestResults[0]
        isEqual = results[0] == bestResults[0]
        hasLessFP = isEqual and \
            results[1] < bestResults[1] and forceFP
        #isLessExtreme = abs(limit - average) < abs(bestLimit - average)


        if isBetterResult or hasLessFP:
            #print('BETTER :', bestLimit, bestResults, limit, results)
            bestLimit = limit
            bestResults = results
            #equalLimits = [limit]

        #elif isEqual:
            #print('EQUAL :', bestLimit, bestResults, limit, results)
            #equalLimits.append(limit)

        limit += difference

    if forPlotting:
        return (resultsList, limitsList)


    #if len(equalLimits) > 1:
        #print('LIMITS:' , equalLimits)
    #    limit = sum(equalLimits)/len(equalLimits)
    #    result = NeuralNetwork.getResultsForLimit\
    #        (blastomResults, otherResults, bestLimit)
    #    if results[0] > bestResults[0]:
    #        bestLimit = limit
            #print(limit, bestLimit)


    return (bestLimit, [round(x/sum(bestResults)*100, 2) for x in bestResults])


def getResultsForLimit(blastomResults, otherResults, limit):
    correctBlastoms = len([i for i in blastomResults if i > limit])
    correctOthers = len([i for i in otherResults if i < limit])
    return [correctBlastoms + correctOthers, \
                len(otherResults) - correctOthers, \
                len(blastomResults) - correctBlastoms]

if __name__ == '__main__':
    calculateClassificationLimit([0.6], [0.4])
