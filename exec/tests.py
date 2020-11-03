
import numpy as np
import sys
from collections import deque
import os
import gym
import random
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..'))
sys.path.append(os.path.join(dirName, '..'))

dataFrame = np.load('scoreUF30.npy')
manipulateLearningRate = [0.0005, 0.0001, 0.00002]
manipulateLayerDepth = [2, 3]
manipulateLayerWidth = [20, 30, 40]
numOfEpisode = 1000
updateFrequency = 30
m = 0
pointGap = 100
for i in range(len(manipulateLayerDepth)):
    for j in range(len(manipulateLayerWidth)):
        plt.subplot(len(manipulateLayerDepth), len(manipulateLayerWidth), len(manipulateLayerWidth)*i + j + 1)
        for k in range(len(manipulateLearningRate)):
            scoreList = dataFrame[m]
            modifiedScoreList = [np.mean(scoreList[l * pointGap:(l + 1) * pointGap]) for l in range(int(len(scoreList) / pointGap))]
            plt.plot(modifiedScoreList, label='learningRate={}'.format(manipulateLearningRate[k]))
            plt.ylim([-1, 140])
            plt.legend()
            plt.title('netDepth={},netWidth={}'.format(manipulateLayerDepth[i], manipulateLayerWidth[j]))
            m += 1
plt.show()

