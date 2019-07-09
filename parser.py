import re

def splitResults(results):
    epochResults = []
    epochFP = []
    epochFN = []

    for result in results:
        epochResults.append(result[0])
        epochFP.append(result[1])
        epochFN.append(result[2])

    return (epochResults, epochFP, epochFN)

def parseLinesForResults(lines, startString = None):

    results = []
    for line in lines:
        if line or startString and line.startswith(startString):
            results.append(getResults(line))

    return splitResults(results)

def getEpochValidationOutputs(lines, epoch):
    blastomResults = []
    otherResults = []
    for line in lines:
        if line.startswith('VALIDATION-- Epoch:' + str(epoch) + ' '):
            output = float(re.findall('\d+\.\d+', line)[1])
            if '_1' in line:
                blastomResults.append(output)
            elif '_0' in line:
                otherResults.append(output)

    return (blastomResults, otherResults)

def getResults(line):
    print(line)
    resultString = re.findall('\d+\.\d+\, \d+\.\d+\, \d+\.\d+', line)[0]
    splitResults = resultString.split(',')
    return (float(splitResults[0]), float(splitResults[1]), float(splitResults[2]))
