""" Developing tflearn model for cartpole Evaluations on openAI
    Here I am working  with CartPol-v0 wherin I have to balance a pole on the cart
    Every frame it is balanced, 1 score is added

"""

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')  # defines the environment to be CartPole environment
env.reset()
goal_steps = 500
score_requirement = 60
initial_games = 20000


def some_random_games():
    for episode in range(5):  # creating 5 episodes to work on
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()  # takes a random action in our environment
            observation, reward, done, info = env.step(action)  # observation: an environment-specific object
            # representing your observation of the environment. reward : amount of reward achieved by the previous
            # action. done: whether it's time to reset the environment again. info:diagnostic information useful for
            # debugging. To get the actual actions, print(ev.action_space)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break


def initial_population():
    training_data = []  # add those moves which gave score > score requirement
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []  # store the moves of every game in memory as we don't know if the score > 50
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward  # reward will be 0 or 1 for each frame
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                output = []
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print("Average accepted score: ", mean(accepted_scores))
    print("Median accepted score:", median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='Input')

    network = fully_connected(network, 128, activation='relu')  # on which input, no of nodes, activation
    network = dropout(network, 0.8)  # on which network , keep rate

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')  # no of output, activation function
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='Targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(x[0]))

    model.fit(X_inputs=x, Y_targets=y, n_epoch=3, snapshot_epoch=1, run_id='openAIStuff', show_metric=True)

    return model


training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obser = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obser) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obser.reshape(-1, len(prev_obser), 1))[0])
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obser = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
        scores.append(score)
print("Average scores", sum(scores)/len(scores))
print("Choice 1: {}, Choice 2: {}".format(choices.count(1)/len(choices), choices.count(0)/len(choices)))


