import numpy as np

class Function:

    def activate():
        pass

    def derived():
        pass

class Identity(Function):

    def activate(self, data):
        return data

    def derived():
        return 1
