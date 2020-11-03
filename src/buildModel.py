import tensorflow as tf
import numpy as np
import random
from collections import deque
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def flat(state):
    state = np.concatenate(state, axis=0)
    state = np.concatenate(state, axis=0)
    return state


class CalculateY:

    def __init__(self, model, updateFrequency):
        self.model = model
        self.step = 0
        self.updateFrequency = updateFrequency

    def __call__(self, nextStatesBatch, rewardBatch, doneBatch, gamma, model):
        if self.step % self.updateFrequency == 0:
            graph = model.graph
            self.model = model
        else:
            graph = self.model.graph
        self.step += 1
        # graph = model.graph
        evalNetOutput_ = graph.get_collection_ref('evalNetOutput')[0]
        states_ = graph.get_collection_ref("states")[0]
        evalNetOutputBatch = self.model.run(evalNetOutput_, feed_dict={states_: nextStatesBatch})
        yBatch = []
        for i in range(0, len(nextStatesBatch)):
            done = doneBatch[i][0]
            if done:
                yBatch.append(rewardBatch[i])
            else:
                yBatch.append(rewardBatch[i] + gamma * np.max(evalNetOutputBatch[i]))
        yBatch = np.asarray(yBatch).reshape(len(nextStatesBatch), -1)
        # print("reward:{}".format(rewardBatch))
        # print("eval:{}".format(evalNetOutputBatch))
        # print("yBatch:{}".format(yBatch))
        return yBatch


class BuildModel:
    def __init__(self, numStateSpace, numActionSpace, gamma, seed=1):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.gamma = gamma
        self.seed = seed

    def __call__(self, layersWidths, summaryPath="./tbdata"):
        print("Generating DQN Model with layers: {}".format(layersWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope('inputs'):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="states")
                act_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="act")
                yi_ = tf.placeholder(tf.float32, [None, 1], name="yi")
                tf.add_to_collection("states", states_)
                tf.add_to_collection("act", act_)
                tf.add_to_collection("yi", yi_)

            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.01)

            with tf.variable_scope("evalNet"):
                with tf.variable_scope("trainEvalHiddenLayers"):
                    activation_ = states_
                    for i in range(len(layersWidths)):
                        fcLayer = tf.layers.Dense(units=layersWidths[i], activation=tf.nn.relu,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="fcEvalHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = fcLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    evalHiddenOutput_ = tf.identity(activation_, name="outputHiddenEval")
                    outputEvalFCLayer = tf.layers.Dense(units=self.numActionSpace, activation=tf.nn.relu,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="fcEvalOut{}".format(len(layersWidths) + 1),
                                                        trainable=True)
                    evalNetOutput_ = outputEvalFCLayer(evalHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{outputEvalFCLayer.kernel.name}"], outputEvalFCLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{outputEvalFCLayer.bias.name}"], outputEvalFCLayer.bias)
                    tf.add_to_collections("evalNetOutput", evalNetOutput_)

            with tf.variable_scope("trainingParams"):
                learningRate_ = tf.constant(0.001, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)

            with tf.variable_scope("QTable"):
                QEval_ = tf.reduce_sum(tf.multiply(evalNetOutput_, act_), reduction_indices=1)
                tf.add_to_collections("QEval", QEval_)
                QEval_ = tf.reshape(QEval_, [-1, 1])
                # loss_ = tf.reduce_mean(tf.square(yi_ - QEval_))
                loss_ = tf.reduce_mean(tf.square(yi_ - QEval_))
                # loss_ = tf.losses.mean_squared_error(labels=yi_, predictions=QEval_)
                tf.add_to_collection("loss", loss_)

            with tf.variable_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer').minimize(loss_)
                tf.add_to_collection("trainOp", trainOpt_)

                saver = tf.train.Saver(max_to_keep=None)
                tf.add_to_collection("saver", saver)

            fullSummary = tf.summary.merge_all()
            tf.add_to_collection("summaryOps", fullSummary)
            if summaryPath is not None:
                trainWriter = tf.summary.FileWriter(summaryPath + "/train", graph=tf.get_default_graph())
                testWriter = tf.summary.FileWriter(summaryPath + "/test", graph=tf.get_default_graph())
                tf.add_to_collection("writers", trainWriter)
                tf.add_to_collection("writers", testWriter)
            saver = tf.train.Saver(max_to_keep=None)
            tf.add_to_collection("saver", saver)

            # self.soft_replace = [tf.assign(t, (1 - self.TAU) * t + self.TAU * e)
            #         for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model


class TrainOneStep:

    def __init__(self, batchSize, updateFrequency, learningRate, gamma, calculateY):
        self.batchSize = batchSize
        self.updateFrequency = updateFrequency
        self.learningRate = learningRate
        self.gamma = gamma
        self.step = 0
        self.calculateY = calculateY

    def __call__(self, model, miniBatch, batchSize):

        # print("ENTER TRAIN")
        graph = model.graph
        states_ = graph.get_collection_ref("states")[0]
        yi_ = graph.get_collection_ref("yi")[0]
        act_ = graph.get_collection_ref("act")[0]
        learningRate_ = graph.get_collection_ref("learningRate")[0]
        loss_ = graph.get_collection_ref("loss")[0]
        trainOp_ = graph.get_collection_ref("trainOp")[0]
        fetches = [loss_, trainOp_]

        states, actions, nextStates, rewards, done = miniBatch
        statesBatch = np.asarray(states).reshape(batchSize, -1)
        actBatch = np.asarray(actions).reshape(batchSize, -1)
        # print("actBatch:{}".format(actBatch))
        nextStatesBatch = np.asarray(nextStates).reshape(batchSize, -1)
        rewardBatch = np.asarray(rewards).reshape(batchSize, -1)
        doneBatch = np.asarray(done).reshape(batchSize, -1)
        yBatch = self.calculateY(nextStatesBatch, rewardBatch, doneBatch, self.gamma, model)
        feedDict = {states_: statesBatch, act_: actBatch, learningRate_: self.learningRate, yi_: yBatch}
        lossDict, trainOp = model.run(fetches, feed_dict=feedDict)

        return model, lossDict


class SampleAction:

    def __init__(self, actionDim):
        self.actionDim = actionDim

    def __call__(self, model, states, epsilon):
        if random.random() < epsilon:
            graph = model.graph
            evalNetOutput_ = graph.get_collection_ref('evalNetOutput')[0]
            states_ = graph.get_collection_ref("states")[0]
            states = flat(states)
            evalNetOutput = model.run(evalNetOutput_, feed_dict={states_: [states]})
            # print("evalNetOutput:{}".format(evalNetOutput))
            # print(np.argmax(QEval[0]))
            # print(evalNetOutput)
            return np.argmax(evalNetOutput)
        else:
            return np.random.randint(0, self.actionDim)


def memorize(replayBuffer, states, act, nextStates, reward, done, actionDim):
    onehotAction = np.zeros(actionDim)
    onehotAction[act] = 1
    replayBuffer.append((states, onehotAction, nextStates, reward, done))
    return replayBuffer


class InitializeReplayBuffer:

    def __init__(self, reset, forwardOneStep, isTerminal, actionDim):
        self.reset = reset
        self.isTerminal = isTerminal
        self.forwardOneStep = forwardOneStep
        self.actionDim = actionDim

    def __call__(self, replayBuffer, maxReplaySize):
        for i in range(maxReplaySize):
            states = self.reset()
            action = np.random.randint(0, self.actionDim)
            nextStates, reward = self.forwardOneStep(states, action)
            done = self.isTerminal(nextStates)
            replayBuffer = memorize(replayBuffer, states, action, nextStates, reward, done, self.actionDim)
        return replayBuffer


def sampleData(data, batchSize):
    batch = [list(varBatch) for varBatch in zip(*random.sample(data, batchSize))]
    return batch


def upgradeEpsilon(epsilon):
    epsilon = epsilon + 0.0001*(1-0.5)
    return epsilon

#
# class ForwardOneStep:
#
#     def __init__(self, transit, getReward):
#         self.transit = transit
#         self.getReward = getReward
#         # self.isTerminal = isTerminal
#
#     def __call__(self, states, action):
#         nextStates = self.transit(states, action)
#         reward = self.getReward(states, action, nextStates)
#         # done = self.isTerminal(states)
#         return nextStates, reward


class RunTimeStep:

    def __init__(self, forwardOneStep, isTerminal, sampleAction, trainOneStep, batchSize, epsilon, actionDelay, actionDim):
        self.forwardOneStep = forwardOneStep
        self.sampleAction = sampleAction
        self.trainOneStep = trainOneStep
        self.batchSize = batchSize
        self.actionDelay = actionDelay
        self.epsilon = epsilon
        self.actionDim = actionDim
        self.isTerminal = isTerminal

    def __call__(self, states, trajectory, model, replayBuffer, score):
        action = self.sampleAction(model, states, self.epsilon)
        # print(action)
        for i in range(self.actionDelay):
            self.epsilon = upgradeEpsilon(self.epsilon)
            nextStates, reward = self.forwardOneStep(states, action)
            done = self.isTerminal(nextStates)
            replayBuffer = memorize(replayBuffer, states, action, nextStates, reward, done, self.actionDim)
            miniBatch = sampleData(replayBuffer, self.batchSize)
            model, loss = self.trainOneStep(model, miniBatch, self.batchSize)
            # print("loss:{}".format(loss))
            score += reward
            trajectory.append(states)
            states = nextStates
        return states, done, trajectory, score, replayBuffer, model


class RunEpisode:

    def __init__(self, reset, runTimeStep):
        self.runTimeStep = runTimeStep
        self.reset = reset

    def __call__(self, model, scoreList, replayBuffer, episode):
        states = self.reset()
        score = 0
        trajectory = []
        while True:
            states, done, trajectory, score, replayBuffer, model = self.runTimeStep(states, trajectory, model, replayBuffer, score)
            if done or score >= 125:
                scoreList.append(score)
                print('episode:', episode, 'score:', score, 'max:', max(scoreList))
                break
        return model, scoreList, trajectory, replayBuffer


class RunAlgorithm:

    def __init__(self, episodeRange, runEpisode):
        self.episodeRange = episodeRange
        self.runEpisode = runEpisode

    def __call__(self, model, replayBuffer):
        scoreList = []
        for i in range(self.episodeRange):
            model, scoreList, trajectory, replayBuffer = self.runEpisode(model, scoreList, replayBuffer, i)
        return model, scoreList, trajectory


























