import numpy as np

class Function:

    def activate():
        pass

    def derived():
        pass

class Identity(Function):

    def activate(self, data):
        return data

    def derived(self, data):
        return 1

def simpleCost(outputValue, correctValue):
    print(outputValue)
    return abs(correctValue - outputValue)
