import tensorflow as tf
import numpy as np
import sys
from collections import deque
import os
import gym
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools as it
import pathos.multiprocessing as mp
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from src.buildModelSoft import *
from chasing.exec.example import composeFowardOneTimeStepWithRandomSubtlety

def saveVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.save(model, path)
    print("Model saved in {}".format(path))


def restoreVariables(model, path):
    graph = model.graph
    saver = graph.get_collection_ref("saver")[0]
    saver.restore(model, path)
    print("Model restored from {}".format(path))
    return model

def underEvaluate(parameters):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    learningRate = parameters['learningRate']
    netDepth = parameters['layerDepth']
    netWidth = parameters['layerWidths']
    tau = parameters['tau']
    episodeRange = parameters['numOfEpisode']
    batchSize = parameters['batchSize']
    numOfAgent = 2
    stateDimPerAgent = 4
    stateDim = stateDimPerAgent * numOfAgent
    actionDim = 8
    seed = 128
    bufferSize = 500000
    maxReplaySize = 64

    gamma = 0.85
    epsilon = 0.5
    # learningRate = 0.0001
    # layerWidths = [30, 30]
    layerWidths = []
    for i in range(netDepth):
        layerWidths.append(netWidth)
    scoreList = []
    # episodeRange = numOfEpisode
    actionDelay = 1
    # updateFrequency = 1

    forwardOneStep, reset, isTerminal = composeFowardOneTimeStepWithRandomSubtlety(numOfAgent)
    sampleAction = SampleAction(actionDim)
    initializeReplayBuffer = InitializeReplayBuffer(reset, forwardOneStep, isTerminal, actionDim)

    buildModel = BuildModel(stateDim, actionDim, gamma, tau)

    model = buildModel(layerWidths)
    trainOneStep = TrainOneStep(batchSize, tau, learningRate, gamma)
    replayBuffer = deque(maxlen=bufferSize)
    replayBuffer = initializeReplayBuffer(replayBuffer, maxReplaySize)

    runTimeStep = RunTimeStep(forwardOneStep, isTerminal, sampleAction, trainOneStep, batchSize, epsilon, actionDelay,
                              actionDim)
    runEpisode = RunEpisode(reset, runTimeStep)
    runAlgorithm = RunAlgorithm(episodeRange, runEpisode)

    model, scoreList, trajectory = runAlgorithm(model, replayBuffer)
    fileName = "learningRate{}netDepth{}netWidth{}tau{}".format(learningRate, netDepth, netWidth, tau)
    dirFolder = os.path.dirname(__file__)
    folder = os.path.join(dirFolder, 'models')
    modelPath = os.path.join(folder, fileName)
    saveVariables(model, modelPath)

    # np.save('scoreUF{}.npy'.format(updateFrequency), dataFrame)
    return scoreList

def main():
    manipulateLearningRate = [0.001, 0.0001, 0.00001]
    manipulateLayerDepth = [1, 2]
    manipulateLayerWidth = [64, 128, 256]
    numOfEpisode = 10000
    batchSize = 64
    tau = 0.05
    pointGap = 10

    # manipulated variables
    manipulatedVariables = OrderedDict()

    manipulatedVariables['learningRate'] = manipulateLearningRate
    manipulatedVariables['layerDepth'] = manipulateLayerDepth
    manipulatedVariables['layerWidths'] = manipulateLayerWidth
    manipulatedVariables['tau'] = [tau]
    manipulatedVariables['numOfEpisode'] = [numOfEpisode]
    manipulatedVariables['batchSize'] = [batchSize]

    productedValues = it.product(*[[(key, value) for value in values] for key, values in manipulatedVariables.items()])
    parametersAllCondtion = [dict(list(specificValueParameter)) for specificValueParameter in productedValues]

    numCpuCores = os.cpu_count()
    print(numCpuCores)
    numCpuToUse = 18
    trainPool = mp.Pool(numCpuToUse)
    scores = trainPool.map(underEvaluate, parametersAllCondtion)
    # print(scores)
    np.save('scoretau{}batchSize{}.npy'.format(tau, batchSize), scores)
    m = 0
    pointGap = 10
    for i in range(len(manipulateLayerDepth)):
        for j in range(len(manipulateLayerWidth)):
            plt.subplot(len(manipulateLayerDepth), len(manipulateLayerWidth), len(manipulateLayerWidth)*i + j + 1)
            for k in range(len(manipulateLearningRate)):
                scoreList = scores[m]
                modifiedScoreList = [np.mean(scoreList[l * pointGap:(l + 1) * pointGap]) for l in range(int(len(scoreList) / pointGap))]
                plt.plot(modifiedScoreList, label='learningRate={}'.format(manipulateLearningRate[k]))
                plt.ylim([-1, 140])
                plt.legend()
                plt.title('netDepth={},netWidth={}'.format(manipulateLayerDepth[i], manipulateLayerWidth[j]))
                m += 1
    plt.show()

    m=0
    pointGap = 100
    for i in range(len(manipulateLayerDepth)):
        for j in range(len(manipulateLayerWidth)):
            plt.subplot(len(manipulateLayerDepth), len(manipulateLayerWidth), len(manipulateLayerWidth)*i + j + 1)
            for k in range(len(manipulateLearningRate)):
                scoreList = scores[m]
                modifiedScoreList = [np.mean(scoreList[l * pointGap:(l + 1) * pointGap]) for l in range(int(len(scoreList) / pointGap))]
                plt.plot(modifiedScoreList, label='learningRate={}'.format(manipulateLearningRate[k]))
                plt.ylim([-1, 140])
                plt.legend()
                plt.title('netDepth={},netWidth={}'.format(manipulateLayerDepth[i], manipulateLayerWidth[j]))
                m += 1
    plt.show()

if __name__ == '__main__':
    main()