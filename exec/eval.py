import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName, '..', '..','..'))

import random
import numpy as np
import pickle
from collections import OrderedDict
import pandas as pd
from matplotlib import pyplot as plt

from src.constrainedChasingEscapingEnv.envMujoco import IsTerminal, TransitionFunction, ResetUniform
from src.constrainedChasingEscapingEnv.reward import RewardFunctionCompete,IsCollided
from exec.trajectoriesSaveLoad import GetSavePath, readParametersFromDf, LoadTrajectories, SaveAllTrajectories, \
    GenerateAllSampleIndexSavePaths, saveToPickle, loadFromPickle
from src.neuralNetwork.policyValueNet import GenerateModel, Train, saveVariables, sampleData, ApproximateValue, \
    ApproximatePolicy, restoreVariables
from src.constrainedChasingEscapingEnv.state import GetAgentPosFromState
from src.neuralNetwork.trainTools import CoefficientCotroller, TrainTerminalController, TrainReporter, LearningRateModifier
from src.replayBuffer import SampleBatchFromBuffer, SaveToBuffer
from exec.preProcessing import AccumulateRewards, AddValuesToTrajectory, RemoveTerminalTupleFromTrajectory, ActionToOneHot, ProcessTrajectoryForPolicyValueNet, PreProcessTrajectories
from src.algorithms.mcts import ScoreChild, SelectChild, InitializeChildren, Expand, MCTS, backup, establishPlainActionDist
from exec.trainMCTSNNIteratively.valueFromNode import EstimateValueFromNode
from src.constrainedChasingEscapingEnv.policies import stationaryAgentPolicy
from src.episode import SampleTrajectory, chooseGreedyAction
from exec.parallelComputing import GenerateTrajectoriesParallel
from exec.evaluationFunctions import ComputeStatistics

def drawPerformanceLine(dataDf, axForDraw, agentId):
    for key, grp in dataDf.groupby('miniBatchSize'):
        grp.index = grp.index.droplevel('miniBatchSize')
        # grp['agentMean'] = np.array([value for value in grp['mean'].values])
        # grp['agentMean'] =  grp['mean'].values
        # grp['agentStd'] = np.array([value for value in grp['std'].values])
        # grp['agentStd'] = grp['std'].values
        grp.plot(ax=axForDraw, y='mean', yerr='std', marker='o', label='miniBatchSize={}'.format(key))

def main():
    # important parameters

    # manipulated variables

    manipulatedVariables = OrderedDict()

    manipulatedVariables['miniBatchSize'] = [64, 256]
    manipulatedVariables['learningRate'] =  [ 1e-3,1e-4,1e-5]
    manipulatedVariables['depth'] =[5,9,17] #[4,8,16]#
    manipulatedVariables['trainSteps']=[0,5000,10000,20000,50000]
    levelNames = list(manipulatedVariables.keys())
    levelValues = list(manipulatedVariables.values())
    modelIndex = pd.MultiIndex.from_product(levelValues, names=levelNames)
    toSplitFrame = pd.DataFrame(index=modelIndex)

    # accumulate rewards for trajectories
    sheepId = 0
    wolfId = 1

    xPosIndex = [0, 1]
    getSheepPos = GetAgentPosFromState(sheepId, xPosIndex)
    getWolfPos = GetAgentPosFromState(wolfId, xPosIndex)



    killzoneRadius = 2
    numSimulations = 150
    maxRunningSteps = 30
    agentId=1


    playAliveBonus = -1/maxRunningSteps
    playDeathPenalty = 1
    playKillzoneRadius = killzoneRadius
    playIsTerminal = IsTerminal(playKillzoneRadius, getSheepPos, getWolfPos)
    playReward = RewardFunctionCompete(playAliveBonus, playDeathPenalty, playIsTerminal)

    decay = 1
    accumulateRewards = AccumulateRewards(decay, playReward)
    addValuesToTrajectory = AddValuesToTrajectory(accumulateRewards)

# generate trajectory parallel
    # generateTrajectoriesCodeName = 'generateWolfResNNEvaluationTrajectoryFixObstacle.py'
    # generateTrajectoriesCodeName = 'generateWolfNNEvaluationTrajectoryFixObstacle.py'
    # generateTrajectoriesCodeName = 'generateWolfResNNEvaluationTrajectoryMovedObstacle.py'
    generateTrajectoriesCodeName = 'generateWolfResNNEvaluationTrajectoryRandomObstacle.py'
    # generateTrajectoriesCodeName = 'generateWolfNNEvaluationTrajectoryRandomObstacle.py'
    evalNumTrials = 100
    numCpuCores = os.cpu_count()
    numCpuToUse = int(0.75 * numCpuCores)
    numCmdList = min(evalNumTrials, numCpuToUse)
    generateTrajectoriesParallel = GenerateTrajectoriesParallel(generateTrajectoriesCodeName,evalNumTrials, numCmdList)

    # run all trials and save trajectories
    generateTrajectoriesParallelFromDf = lambda df: generateTrajectoriesParallel(readParametersFromDf(df))
    toSplitFrame.groupby(levelNames).apply(generateTrajectoriesParallelFromDf)

    # save evaluation trajectories
    dirName = os.path.dirname(__file__)
    dataFolderName=os.path.join(dirName,'..','..', '..', 'data', 'multiAgentTrain', 'MCTSRandomObstacle')
    trajectoryDirectory = os.path.join(dataFolderName,'evaluationTrajectoriesResNNWithObstacle')




    if not os.path.exists(trajectoryDirectory):
        os.makedirs(trajectoryDirectory)
    trajectoryExtension = '.pickle'


    trajectoryFixedParameters = {'maxRunningSteps': maxRunningSteps, 'numSimulations': numSimulations, 'killzoneRadius': killzoneRadius,'agentId':agentId}

    getTrajectorySavePath = GetSavePath(trajectoryDirectory, trajectoryExtension, trajectoryFixedParameters)
    getTrajectorySavePathFromDf = lambda df: getTrajectorySavePath(readParametersFromDf(df))

    # compute statistics on the trajectories


    fuzzySearchParameterNames = ['sampleIndex']
    loadTrajectories = LoadTrajectories(getTrajectorySavePath, loadFromPickle, fuzzySearchParameterNames)
    loadTrajectoriesFromDf = lambda df: loadTrajectories(readParametersFromDf(df))
    measurementFunction = lambda trajectory: accumulateRewards(trajectory)[0]



    computeStatistics = ComputeStatistics(loadTrajectoriesFromDf, measurementFunction)
    statisticsDf = toSplitFrame.groupby(levelNames).apply(computeStatistics)
    print(statisticsDf)

    # manipulatedVariables['miniBatchSize'] = [64, 128]
    # manipulatedVariables['learningRate'] =  [ 1e-3,1e-4,1e-5]
    # manipulatedVariables['depth'] = [4,8,16]
    # manipulatedVariables['trainSteps']=[0,20000,40000,60000,100000,180000]

    # plot the results
    fig = plt.figure()
    numRows = len(manipulatedVariables['depth'])
    numColumns = len(manipulatedVariables['learningRate'])
    plotCounter = 1
    selfId=0
    for depth, grp in statisticsDf.groupby('depth'):
        grp.index = grp.index.droplevel('depth')

        for learningRate, group in grp.groupby('learningRate'):
            group.index = group.index.droplevel('learningRate')

            axForDraw = fig.add_subplot(numRows, numColumns, plotCounter)
            if (plotCounter % numColumns == 1) or numColumns==1:
                axForDraw.set_ylabel('depth: {}'.format(depth))
            if plotCounter <= numColumns:
                axForDraw.set_title('learningRate: {}'.format(learningRate))

            axForDraw.set_ylim(-1, 1)
            drawPerformanceLine(group, axForDraw, selfId)
            plotCounter += 1


    plt.suptitle('SupervisedNNWolfwithRandomWallState')
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    main()