import numpy as np

def mae(predictions,actual):
    return np.average(np.abs(predictions-actual))

def mape(predictions,actual):
    return np.average(np.abs(predictions-actual)/np.abs(actual))

