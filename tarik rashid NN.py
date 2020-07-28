import numpy as np
from scipy.special import expit, logit


class NN:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes

        self.lr = learningRate

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation = lambda x: expit(x)

    def train(self, input_list, target_list):
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndim=2).T

        hInputs = np.dot(self.wih, inputs)
        hOutputs = self.activation(hInputs)

        oInputs = np.dot(self.who, hOutputs)
        oOutputs = self.activation(oInputs)

        outputErrors = targets - oOutputs
        hiddenErrors = np.dot(self.who.T, outputErrors)

        self.who = self.lr * np.dot(outputErrors * oOutputs * (1.0 - oOutputs), hOutputs.T)
        self.wih = self.lr * np.dot(hiddenErrors * hOutputs * (1.0 - hOutputs), inputs.T)

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hInputs = np.dot(self.wih, inputs)
        hOutputs = self.activation(hInputs)

        oInputs = np.dot(self.who, hOutputs)
        oOutputs = self.activation(oInputs)

        return oOutputs
