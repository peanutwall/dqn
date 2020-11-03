import tensorflow as tf
import numpy as np
import sys
from collections import deque
import os
import gym
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))


from src.buildModelRL import *
from env.discreteMountainCarEnv import *
from env.discreteCartPole import *


def underEvaluate(gamma, learningRate, updateFrequency):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    envName = 'MountainCar-v0'
    # envName = 'CartPole-v0'
    env = gym.make(envName)
    stateDim = env.observation_space.shape[0]
    actionDim = env.action_space.n
    seed = 128
    bufferSize = 100000
    maxReplaySize = 32
    batchSize = 32
    # gamma = 0.95
    epsilon = 0.1
    # learningRate = 0.01
    layerWidths = [30]
    scoreList = []
    episodeRange = 1000
    actionDelay = 1
    # updateFrequency = 1

    transit = TransitMountCarDiscrete()
    reset = ResetMountCarDiscrete(seed)
    rewardFunction = rewardMountCarDiscrete
    isTerminal = IsTerminalMountCarDiscrete()
    #
    # transit = TransitCartPole()
    # reset = ResetCartPole(seed)
    # rewardFunction = RewardCartPole()
    # isTerminal = IsTerminalCartPole()

    sampleAction = SampleAction(actionDim)
    initializeReplayBuffer = InitializeReplayBuffer(reset, transit, rewardFunction, isTerminal, actionDim)
    buildModel = BuildModel(stateDim, actionDim, gamma)

    model = buildModel(layerWidths)
    calculateY = CalculateY(model, updateFrequency)
    trainOneStep = TrainOneStep(batchSize, updateFrequency, learningRate, gamma, calculateY)
    replayBuffer = deque(maxlen=bufferSize)
    replayBuffer = initializeReplayBuffer(replayBuffer, maxReplaySize)
    miniBatch = sampleData(replayBuffer, batchSize)
    forwardOneStep = ForwardOneStep(transit, rewardFunction)
    runTimeStep = RunTimeStep(forwardOneStep, isTerminal, sampleAction, trainOneStep, batchSize, epsilon, actionDelay, actionDim)
    runEpisode = RunEpisode(reset, runTimeStep)
    runAlgorithm = RunAlgorithm(episodeRange, runEpisode)
    model, scoreList, trajectory = runAlgorithm(model, replayBuffer)
    env.close()
    return scoreList

import matplotlib.pyplot as plt
manipulateGamma = [0.85, 0.9, 0.95]
manipulateLearningRate = [0.01, 0.001, 0.0001]
manipulateUpdateFrequency = [1, 3, 6]
for i in range(len(manipulateUpdateFrequency)):
    for j in range(len(manipulateLearningRate)):
        plt.subplot(len(manipulateUpdateFrequency), len(manipulateLearningRate), j + 1)
        for k in range(len(manipulateGamma)):
            scoreList = underEvaluate(manipulateGamma[k], manipulateLearningRate[j], manipulateUpdateFrequency[i])
            plt.plot(scoreList, label='gamma='+str(manipulateGamma[k]))
            plt.legend()
plt.show()

for i in range(len(manipulateLearningRate)):
    scoreList = underEvaluate(0.95, manipulateLearningRate[i], 1)
    plt.plot(scoreList, label='learningRate='+str(manipulateLearningRate[i]))
    plt.legend()
plt.show()