import tensorflow as tf
import numpy as np
import random
from collections import deque
import os
tf.compat.v1.disable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def flat(state):
    state = np.concatenate(state, axis=0)
    state = np.concatenate(state, axis=0)
    return state


class BuildModel:
    def __init__(self, numStateSpace, numActionSpace, gamma, tau, seed=1):
        self.numStateSpace = numStateSpace
        self.numActionSpace = numActionSpace
        self.gamma = gamma
        self.seed = seed
        self.tau = tau

    def __call__(self, layersWidths, summaryPath="./tbdata"):
        print("Generating DQN Model with layers: {}".format(layersWidths))
        graph = tf.Graph()
        with graph.as_default():
            if self.seed is not None:
                tf.set_random_seed(self.seed)

            with tf.name_scope('inputs'):
                states_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="states")
                nextStates_ = tf.placeholder(tf.float32, [None, self.numStateSpace], name="nextStates")
                act_ = tf.placeholder(tf.float32, [None, self.numActionSpace], name="act")
                reward_ = tf.placeholder(tf.float32, [None, 1], name="reward")
                # done_ = tf.placeholder(tf.float32, [None, 1], name="done")
                tf.add_to_collection("states", states_)
                tf.add_to_collection("nextStates", nextStates_)
                tf.add_to_collection("act", act_)
                tf.add_to_collection("reward", reward_)
                # tf.add_to_collection("done", done_)

            initWeight = tf.random_uniform_initializer(-0.03, 0.03)
            initBias = tf.constant_initializer(0.01)

            with tf.variable_scope("evalNet"):
                with tf.variable_scope("trainEvalHiddenLayers"):
                    activation_ = states_
                    for i in range(len(layersWidths)):
                        fcLayer = tf.layers.Dense(units=layersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="fcEvalHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = fcLayer(activation_)

                        tf.add_to_collections(["weights", f"weight/{fcLayer.kernel.name}"], fcLayer.kernel)
                        tf.add_to_collections(["biases", f"bias/{fcLayer.bias.name}"], fcLayer.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    evalHiddenOutput_ = tf.identity(activation_, name="outputHiddenEval")
                    outputEvalFCLayer = tf.layers.Dense(units=self.numActionSpace, activation=None,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="fcEvalOut{}".format(len(layersWidths) + 1),
                                                        trainable=True)
                    evalNetOutput_ = outputEvalFCLayer(evalHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{outputEvalFCLayer.kernel.name}"], outputEvalFCLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{outputEvalFCLayer.bias.name}"], outputEvalFCLayer.bias)
                    tf.add_to_collections("evalNetOutput", evalNetOutput_)

            with tf.variable_scope("targetNet"):
                with tf.variable_scope("trainTargetHiddenLayers"):
                    activation_ = nextStates_
                    for i in range(len(layersWidths)):
                        fcLayerTarget = tf.layers.Dense(units=layersWidths[i], activation=None,
                                                  kernel_initializer=initWeight,
                                                  bias_initializer=initBias, name="fcTargetHidden{}".format(i + 1),
                                                  trainable=True)
                        activation_ = fcLayerTarget(activation_)

                        tf.add_to_collections(["weights", f"weight/{fcLayerTarget.kernel.name}"], fcLayerTarget.kernel)
                        tf.add_to_collections(["biases", f"bias/{fcLayerTarget.bias.name}"], fcLayerTarget.bias)
                        tf.add_to_collections(["activations", f"activation/{activation_.name}"], activation_)
                    targetHiddenOutput_ = tf.identity(activation_, name="outputHiddenTarget")
                    outputTargetFCLayer = tf.layers.Dense(units=self.numActionSpace, activation=None,
                                                        kernel_initializer=initWeight,
                                                        bias_initializer=initBias,
                                                        name="fcTargetOut{}".format(len(layersWidths) + 1),
                                                        trainable=True)
                    targetNetOutput_ = outputTargetFCLayer(targetHiddenOutput_)
                    tf.add_to_collections(["weights", f"weight/{outputTargetFCLayer.kernel.name}"], outputTargetFCLayer.kernel)
                    tf.add_to_collections(["biases", f"bias/{outputTargetFCLayer.bias.name}"], outputTargetFCLayer.bias)
                    tf.add_to_collections("TargetNetOutput", targetNetOutput_)

            with tf.variable_scope("trainingParams"):
                learningRate_ = tf.constant(0.001, dtype=tf.float32)
                tf.add_to_collection("learningRate", learningRate_)

            with tf.variable_scope("QTable"):
                QEval_ = tf.reduce_sum(tf.multiply(evalNetOutput_, act_), reduction_indices=1)
                tf.add_to_collections("QEval", QEval_)
                QEval_ = tf.reshape(QEval_, [-1, 1])
                Qtarget_ = tf.reduce_max(targetNetOutput_)
                Qtarget_ = tf.reshape(Qtarget_, [-1, 1])
                yi_ = reward_ + self.gamma*Qtarget_
                # yi_ = reward_ + self.gamma * Qtarget_ * (1 - done_)
                yi_ = tf.reshape(yi_, [-1, 1])
                loss_ = tf.reduce_mean(tf.square(yi_ - QEval_))
                # loss_ = tf.losses.mean_squared_error(labels=yi_, predictions=QEval_)
                tf.add_to_collection("loss", loss_)

            evalParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evalNet')
            targetParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetNet')
            softReplace_ = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(targetParams, evalParams)]
            tf.add_to_collection("softReplace", softReplace_)

            with tf.variable_scope("train"):
                trainOpt_ = tf.train.AdamOptimizer(learningRate_, name='adamOptimizer').minimize(loss_, var_list=evalParams)
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

            model = tf.Session(graph=graph)
            model.run(tf.global_variables_initializer())

        return model


class TrainOneStep:

    def __init__(self, batchSize, tau, learningRate, gamma):
        self.batchSize = batchSize
        self.tau = tau
        self.learningRate = learningRate
        self.gamma = gamma
        self.step = 0
        # self.softReplace = softReplace

    def __call__(self, model, miniBatch, batchSize):

        # print("ENTER TRAIN")
        graph = model.graph
        states_ = graph.get_collection_ref("states")[0]
        nextStates_ = graph.get_collection_ref("nextStates")[0]
        reward_ = graph.get_collection_ref("reward")[0]
        act_ = graph.get_collection_ref("act")[0]
        learningRate_ = graph.get_collection_ref("learningRate")[0]
        loss_ = graph.get_collection_ref("loss")[0]
        # done_ = graph.get_collection_ref("done")[0]
        trainOp_ = graph.get_collection_ref("trainOp")[0]
        softReplace_ = graph.get_collection_ref("softReplace")
        fetches = [loss_, trainOp_]

        # config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        # sess = tf.Session(graph=graph)
        evalParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='evalNet')
        targetParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='targetNet')
        softReplace_ = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(targetParams, evalParams)]
        model.run(softReplace_)

        states, actions, nextStates, rewards, done = miniBatch
        statesBatch = np.asarray(states).reshape(batchSize, -1)
        actBatch = np.asarray(actions).reshape(batchSize, -1)
        # print("actBatch:{}".format(actBatch))
        nextStatesBatch = np.asarray(nextStates).reshape(batchSize, -1)
        rewardBatch = np.asarray(rewards).reshape(batchSize, -1)
        doneBatch = np.asarray(done).reshape(batchSize, -1)
        feedDict = {states_: statesBatch, nextStates_: nextStatesBatch, act_: actBatch, learningRate_: self.learningRate, reward_: rewardBatch}
        # feedDict = {states_: statesBatch, nextStates_: nextStatesBatch, act_: actBatch,
        #             learningRate_: self.learningRate, reward_: rewardBatch, done_: doneBatch}
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
            if done or score < -800:
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
