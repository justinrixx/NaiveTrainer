import gym
import numpy as np
import random
import math

"""
This is a Q learning algorithm that learns how to balance a pole
on a cart
"""

# Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v0')

# things about the environment
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
NUM_ACTIONS = env.action_space.n  # (left, right)
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
# to make things easier with putting things into buckets
STATE_BOUNDS[1] = [-0.5, 0.5]
STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]

# things about the simulation
MAX_T = 350
NUM_EPISODES = 1000

# algorithm parameters
gamma = .99
MIN_LEARNING_RATE = .1
MIN_EPSILON = .01


# make an array that of the size num_theta_buckets x num_theta'_buckets x NUM_ACTIONS
q_table = np.zeros( (NUM_BUCKETS[2], NUM_BUCKETS[3], NUM_ACTIONS) )


def run():

    learning_rate = get_learning_rate(0)
    epsilon = get_epsilon(0)

    for i in range(0, NUM_EPISODES):
        observation = env.reset()
        state_prev = state_to_bucket(observation)

        for t in range(MAX_T):
            env.render()

            # pick action
            action = get_action(state_prev, epsilon)

            # execute the action
            observation, reward, done, info = env.step(action)

            # get the new state and update the q_table
            state = state_to_bucket(observation)
            best_q = np.amax(q_table[state])
            q_table[state_prev + (action,)] += learning_rate * (reward + (gamma * best_q)
                                                                - q_table[state_prev + (action,)])

            # now this state is old news
            state_prev = state

            if done:
                print('finished episode after ' + str(t) + ' steps')
                break

            # Update parameters
            epsilon = get_epsilon(i)
            learning_rate = get_learning_rate(i)


def get_action(state, epsilon):
    # Select a random action
    if random.random() < epsilon:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def get_epsilon(t):
    return max(MIN_EPSILON, min(1, 1.0 - math.log10((t + 1) / 25)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(2, len(state)):  # start at 2 because we want to skip x and x'
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == '__main__':
    run()
