import tensorflow as tf
import numpy as np
import sys
from collections import deque
import os
import gym
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

from src.buildModel import *
# from src.buildModelHuber import *
# from env.discreteMountainCarEnv import *
# from env.discreteCartPole import *
from chasing.exec.example import composeFowardOneTimeStepWithRandomSubtlety

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# envName = 'MountainCar-v0'
# envName = 'CartPole-v0'
# env = gym.make(envName)
# stateDim = env.observation_space.shape[0]
# actionDim = env.action_space.n
numOfAgent = 2
stateDimPerAgent = 4
stateDim = stateDimPerAgent*numOfAgent
actionDim = 8
seed = 128
bufferSize = 100000
maxReplaySize = 256
batchSize = 256
gamma = 0.95
epsilon = 0.5
learningRate = 0.00001
layerWidths = [30, 30]
scoreList = []
episodeRange = 5000
actionDelay = 1
updateFrequency = 6

# transit = TransitMountCarDiscrete()
# reset = ResetMountCarDiscrete(seed)
# rewardFunction = rewardMountCarDiscrete
# isTerminal = IsTerminalMountCarDiscrete()
#
# transit = TransitCartPole()
# reset = ResetCartPole(seed)
# rewardFunction = RewardCartPole()
# isTerminal = IsTerminalCartPole()
#
# forwardOneStep = ForwardOneStep(transit, rewardFunction)
forwardOneStep, reset, isTerminal = composeFowardOneTimeStepWithRandomSubtlety(numOfAgent)
sampleAction = SampleAction(actionDim)
initializeReplayBuffer = InitializeReplayBuffer(reset, forwardOneStep, isTerminal, actionDim)

buildModel = BuildModel(stateDim, actionDim, gamma)

model = buildModel(layerWidths)
calculateY = CalculateY(model, updateFrequency)
trainOneStep = TrainOneStep(batchSize, updateFrequency, learningRate, gamma, calculateY)
replayBuffer = deque(maxlen=bufferSize)
replayBuffer = initializeReplayBuffer(replayBuffer, maxReplaySize)
miniBatch = sampleData(replayBuffer, batchSize)

runTimeStep = RunTimeStep(forwardOneStep, isTerminal, sampleAction, trainOneStep, batchSize, epsilon, actionDelay, actionDim)
runEpisode = RunEpisode(reset, runTimeStep)
runAlgorithm = RunAlgorithm(episodeRange, runEpisode)
model, scoreList, trajectory = runAlgorithm(model, replayBuffer)

import matplotlib.pyplot as plt

plt.plot(scoreList, color='green')
plt.show()

lastScore = scoreList[-300:]
threshold = 100
times = 0

showDemo = False
if showDemo:
    visualize = VisualizeCartPole()
    visualize(trajectory)

