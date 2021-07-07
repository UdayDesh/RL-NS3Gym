#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ns3gym import ns3env

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
total_episodes = 200
max_steps_per_episode = 100#00
env = gym.make("ns3-v0")  # Create the environment
env.seed(seed)
env._max_episode_steps = max_steps_per_episode
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
num_inputs = 4
num_actions = 2
num_hidden = 128

epsilon = 1.0               # exploration rate
epsilon_min = 0.01
epsilon_decay = 0.999

ob_space = env.observation_space
ac_space = env.action_space
print("Observation space: ", ob_space,  ob_space.dtype)
print("Action space: ", ac_space, ac_space.n)

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation=tf.nn.relu)(inputs)
action = layers.Dense(num_actions, activation=tf.nn.softmax)(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=eps, amsgrad=False,name='ChannelPredict')
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
cum_rew_history = [0]
running_reward = 0
episode_count = 0

time_history = []
rew_history = []

for e in range(total_episodes):
    state = env.reset()
    episode_reward = []
    rewardsum = 0
    with tf.GradientTape() as tape:#persistent=True) as tape:
        for timestep in range(1, max_steps_per_episode): 
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)
            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            # Apply the sampled action in NS3 environment
            state, reward, done, _ = env.step(action)
            #print("ACTION: {}, state:{}, reward: {}, done: {}".format(action, state, reward, done))
            episode_reward.append(reward)
            #episode_reward += reward

            if done:
                print("episode: {}/{}, time: {}, rew: {}, eps: {:.2}"
                      .format(e, total_episodes, timestep, sum(episode_reward), epsilon))
                break

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for the critic
        returns = []
        discounted_sum = 0
        for r in episode_reward[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Update running reward to check condition for solving
        running_reward = 0.05 * sum(episode_reward) + (1 - 0.05) * running_reward

        # Clear the loss and reward history
        time_history.append(timestep)
        print ("Running reward:",running_reward)
        cum_rew_history.append(running_reward)
        rew_history.append(episode_reward)
        action_probs_history.clear()
        critic_value_history.clear()
        if epsilon > epsilon_min: epsilon *= epsilon_decay
        
print("Plot Learning Performance")
#print('----'*10,'\n',rew_history,'----'*10)
from excelChannel import ExcelChannel

ec = ExcelChannel()
ec.inputIterations(rew_history)
ec.inputChart(rew_history)
ec.inputCumulative(cum_rew_history)
