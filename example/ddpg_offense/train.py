from __future__ import print_function
import os
import numpy as np
import math
import tensorflow as tf
import random
from collections import deque
import sys, os



# from train.actor import Actor
# from train.critic import Critic

from tensorflow.contrib.layers import batch_norm

try:
  import hfo
except ImportError:
  print('Failed to import hfo. To install hfo, in the HFO directory'\
    ' run: \"pip install .\"')
  exit()
# def build_ddpg_graph(actor, critic):
#     actor = Actor()
#     critic = Critic()

GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
TAU = 0.01      # soft replacement

##############################     DQN class begin     ##################################


class DQN():
  def __init__(self):
    self.replay_buffer = deque()
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = 12
    self.action_dim = 10
    self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + self.action_dim + 1), dtype=np.float32)


    self.session = tf.Session()

    self.state = tf.placeholder(tf.float32, [None, self.state_dim], 'state')
    self.state_next = tf.placeholder(tf.float32, [None, self.state_dim], 'state_next')
    self.R = tf.placeholder(tf.float32, [None, 1], 'r')

    with tf.variable_scope('Actor'):
      self.action = self._build_a(self.state, scope='eval', trainable=True)
      action_next = self._build_a(self.state_next, scope='target', trainable=False)
    with tf.variable_scope('Critic'):
      q = self._build_c(self.state, self.action, scope='eval', trainable=True)
      q_next = self._build_c(self.state_next, action_next, scope='target', trainable=False)

    self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
    self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
    self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
    self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

    self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

    q_target = self.R + GAMMA * q_next

    # in the feed_dic for the td_error, the self.a should change to actions in memory


    td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)


  # def create_Q_network(self):
  #   W1 = self.weight_variable([self.state_dim, 20])
  #   b1 = self.bias_variable([20])
  #   W2 = self.weight_variable([20,self.action_dim])
  #   b2 = self.bias_variable([self.action_dim])

  #   self.state_input = tf.placeholder("float32", [None, self.state_dim])

  #   h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)

  #   self.Q_value = tf.matmul(h_layer, W2) + b2
  
  # def weight_variable(self, shape):
  #   initial = tf.truncated_normal(shape)
  #   return tf.Variable(initial)

  # def bias_variable(self, shape):
  #   initial = tf.constant(0.01, shape = shape)
  #   return tf.Variable(initial)

  # def create_training_method(self):
  #   pass


  # def perceive(self,state, action, reward, next_state, done):
  #   one_hot_action = np.zeros(self.action_dim)
  #   one_hot_action[action] = 1

  # def train_Q_network(self):
  #   pass

  def egreedy_action(self, state):
    action = np.zeros(10, float)
    action[random.randint(0,3)] = 1
    action[4] = random.uniform(-100, 100)#DASH POWER
    action[5] = random.uniform(-180, 180)#DASH DEGREES
    action[6] = random.uniform(-180, 180)#TURN DEGREES
    action[7] = random.uniform(-180, 180)#TACKLE DEGREES
    action[8] = random.uniform(0, 100)#KICK POWER
    action[9] = random.uniform(-180, 180)#KICK DEGREES
    return action

  

##############################     DQN class done     ##################################


##############################     global functions begin     ##################################

def getReward(s):
  reward=0
  #--------------------------- 
  if s==hfo.GOAL:
    reward=1
  #--------------------------- 
  elif s==hfo.CAPTURED_BY_DEFENSE:
    reward=-1
  #--------------------------- 
  elif s==hfo.OUT_OF_BOUNDS:
    reward=-1
  #--------------------------- 
  #Cause Unknown Do Nothing
  elif s==hfo.OUT_OF_TIME:
    reward=0
  #--------------------------- 
  elif s==hfo.IN_GAME:
    reward=0
  #--------------------------- 
  elif s==hfo.SERVER_DOWN:  
    reward=0
  #--------------------------- 
  else:
    print("Error: Unknown GameState", s)
  return reward

def distance(a, b):
  c = [0,0]
  c[0] = a[0] - b[0]
  c[1] = a[1] - b[1]
  return math.hypot(c[0], c[1])
def getReward2(last_state, state, status, has_kicked):
  reward = 0
  if status == hfo.GOAL:
    reward = reward + 5
  if int(state[5]) == 1 and has_kicked == False:
    reward = reward + 1
  reward = reward + distance([last_state[0], last_state[1]], [last_state[3], last_state[4]]) - distance([state[0], state[1]], [state[3], state[4]])
  reward = reward + 3 * distance([last_state[3], last_state[4]], [1,0])
  reward = reward - 3 * distance([state[3], state[4]], [1,0])#[1,0] is the goal position
  return reward



##############################     global functions done     ##################################

##############################     main function begin     ##################################

def main():
  hfo_env = hfo.HFOEnvironment()
  hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET)
  agent = DQN()
  for episode in range(5):  # replace with xrange(5) for Python 2.X
    status = hfo.IN_GAME
    count = 0
    action = np.zeros(10, dtype=float)
    has_kicked = False
    print("episode begin")
    while status == hfo.IN_GAME:
      count = count + 1
      print("count" + str(count))
      state = hfo_env.getState()
      if int(state[5]) == 1:
        has_kicked = True
      if bool(action[0]) or bool(action[1]) or bool(action[2]) or bool(action[3]) == True:
        reward = getReward2(last_state, state, status, has_kicked)
        print("reward")
        print(reward)
        # print(count, action)
      action = agent.egreedy_action(state)
      # print(action)
      if int(action[0]) == 1:
        hfo_env.act(hfo.DASH, action[4], action[5])
        print("DASH ")
      elif int(action[1]) == 1:
        hfo_env.act(hfo.TURN, action[6])
        print("turn")
      elif int(action[2]) == 1:
        hfo_env.act(hfo.TACKLE, action[7])
        print("tackle")
      elif int(action[3]) == 1:
        # hfo_env.act(hfo.KICK, action[8], action[9])
        hfo_env.act(hfo.SHOOT)
        print("kick")
      status=hfo_env.step()
      last_state = state
      
    
    # Quit if the server goes down
    if status == hfo.SERVER_DOWN:
      hfo_env.act(hfo.QUIT)
      exit()
##############################     main function done     ##################################


          
           


if __name__ == '__main__':
  main()
