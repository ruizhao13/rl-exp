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


try:
  import hfo
except ImportError:
  print('Failed to import hfo. To install hfo, in the HFO directory'\
    ' run: \"pip install .\"')
  exit()
# def build_ddpg_graph(actor, critic):
#     actor = Actor()
#     critic = Critic()
LR_A = 0.01    # learning rate for actor
LR_C = 0.01    # learning rate for critic
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
EPSILON = 0.9
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
MEMORY_CAPACITY = 10000
TAU = 0.01      # soft replacement

##############################     DQN class begin     ##################################


class DQN():
  def __init__(self):
    self.replay_buffer = deque()
    self.time_step = 0
    self.epsilon = EPSILON
    self.pointer = 0 #used for updating the memory
    self.state_dim = 12
    self.action_dim = 10
    self.a_bound = [1,1,1,1,100, 180, 180, 180, 50, 180]
    self.a_min = [0, 0, 0, 0, 0, 0, 0, 0, 50, 0]
    self.memory = np.zeros((MEMORY_CAPACITY, self.state_dim * 2 + self.action_dim + 1), dtype=np.float32)

    self.sess = tf.Session()

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
    
    self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

    a_loss = -tf.reduce_mean(q)
    self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
    self.sess.run(tf.global_variables_initializer())
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5,max_to_keep=0)


 

  def choose_action(self, state):
    return self.sess.run(self.action, {self.state: state[np.newaxis, :]})[0]

  def learn(self):
    self.sess.run(self.soft_replace)

    indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
    bt = self.memory[indices, :]
    bs = bt[:, :self.state_dim]
    ba = bt[:, self.state_dim: self.state_dim + self.action_dim]
    br = bt[:, -self.state_dim - 1: -self.state_dim]
    bs_ = bt[:, -self.state_dim :]
    # print(bt)
    # print(bs)
    # print(ba)
    # print(br)
    # print(bs_)

    self.sess.run(self.atrain, {self.state: bs})
    self.sess.run(self.ctrain, {self.state: bs, self.action: ba, self.R: br, self.state_next: bs_})
  
  
  def store_transition(self, state, action, reward, state_next):
    transition = np.hstack((state, action, [reward], state_next))
    index = self.pointer % MEMORY_CAPACITY
    self.memory[index, :] = transition
    self.pointer += 1


  def _build_a(self, state, scope, trainable):
    with tf.variable_scope(scope):
      net = tf.layers.dense(state, 1024, activation=tf.nn.relu, name='l1', kernel_initializer=tf.random_normal_initializer(),trainable=trainable)
      net2 = tf.layers.dense(net, 512, activation=tf.nn.relu, name='l2', kernel_initializer=tf.random_normal_initializer(), trainable=trainable)
      net3 = tf.layers.dense(net, 256, activation=tf.nn.relu, name='l3', kernel_initializer=tf.random_normal_initializer(), trainable=trainable)
      net4 = tf.layers.dense(net, 128, activation=tf.nn.relu, name='l4', kernel_initializer=tf.random_normal_initializer(), trainable=trainable)
      a = tf.layers.dense(net4, self.action_dim, activation=tf.nn.tanh, name='action', kernel_initializer=tf.random_normal_initializer(), trainable=trainable)
      return tf.multiply(a, self.a_bound, name='scaled_a') + self.a_min
      # not done

  def _build_c(self, state, action, scope, trainable):
    with tf.variable_scope(scope):
      c_input = tf.concat([state, action], 1)
      net = tf.layers.dense(c_input, 1024, activation=tf.nn.relu, name='l1', kernel_initializer=tf.random_normal_initializer(),trainable=trainable)
      net2 = tf.layers.dense(net, 512, activation=tf.nn.relu, name='l2', kernel_initializer=tf.random_normal_initializer(), trainable=trainable)
      net3 = tf.layers.dense(net, 256, activation=tf.nn.relu, name='l3', kernel_initializer=tf.random_normal_initializer(), trainable=trainable)
      net4 = tf.layers.dense(net, 128, activation=tf.nn.relu, name='l4', kernel_initializer=tf.random_normal_initializer(), trainable=trainable)
      # n_l1 = 30
      # w1_s = tf.get_variable('w1_s', [self.state_dim, n_l1], trainable=trainable)
      # w1_a = tf.get_variable('w1_a', [self.action_dim, n_l1], trainable=trainable)
      # b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
      # net = tf.nn.relu(tf.matmul(state, w1_s) + tf.matmul(action, w1_a) + b1)
      return tf.layers.dense(net4, 1, trainable=trainable)
  
  def egreedy_action(self):
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
  state[9] = 0
  if min(state) == -2 or state[0] == -1 or state[1] == -1 or state[0] ==1 or state[1] == 1:
    # print(state)
    state[9] = -2
    return -10000
  # print("flag")
  # print(state)
  if status == hfo.GOAL:
    reward = reward + 5
  if int(state[5]) == 1 and has_kicked == False:
    reward = reward + 1
  reward = reward + distance([last_state[0], last_state[1]], [last_state[3], last_state[4]]) - distance([state[0], state[1]], [state[3], state[4]])
  reward = reward + 3 * distance([last_state[3], last_state[4]], [1,0])
  reward = reward - 3 * distance([state[3], state[4]], [1,0])#[1,0] is the goal position
  state[9]=-2
  return reward



##############################     global functions done     ##################################

##############################     main function begin     ##################################

def main():
  hfo_env = hfo.HFOEnvironment()
  hfo_env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET)
  agent = DQN()
  model_file=tf.train.latest_checkpoint('ckpt/')
  agent.saver.restore(agent.sess,model_file)
  for episode in range(10000):  # replace with xrange(5) for Python 2.X
    status = hfo.IN_GAME
    count = 0
    action = np.zeros(10, dtype=float)
    has_kicked = False
    # print("episode begin")
    
    while status == hfo.IN_GAME:
      count = count + 1
      # print("count" + str(count))
      state = hfo_env.getState()
      
      if int(state[5]) == 1:
        has_kicked = True
      if bool(action[0]) or bool(action[1]) or bool(action[2]) or bool(action[3]) == True:
        reward = getReward2(last_state, state, status, has_kicked)
        # print("reward") 
        print(reward)
        if reward > 0:
          print('      +')
        else:
          print('               -')
        agent.store_transition(state=last_state, action=action, reward=reward, state_next=state)
        # print(count, action)
        if agent.pointer > MEMORY_CAPACITY:
          agent.learn()
      if np.random.uniform() < EPSILON:
        action = agent.choose_action(state)
      else:
        action = agent.egreedy_action()
      maximum = np.max(action[:4])
      a_c = np.where(action[:4] == maximum)
      # print(a_c)
      a_c = a_c[0][0]
      # print(a_c)
      # print(action)
      if a_c == 3 and int(state[5]) == 1:
        hfo_env.act(hfo.KICK, action[8], action[9])
        print("kick")
        
      elif a_c == 1:
        hfo_env.act(hfo.TURN, action[6])
        print("turn")
      elif a_c == 2:
        hfo_env.act(hfo.DASH, action[4], action[5])
        print("DASH")
      else:
        # hfo_env.act(hfo.KICK, action[8], action[9])
        hfo_env.act(hfo.DASH, action[4], action[5])
        print("DASH ")
      status=hfo_env.step()
      last_state = state
    
    # Quit if the server goes down
    if status == hfo.SERVER_DOWN:
      hfo_env.act(hfo.QUIT)
      exit()
    if episode % 100 == 0:
      agent.saver.save(agent.sess, 'ckpt/mnist.ckpt', global_step=episode)  
    
  
##############################     main function done     ##################################


          
           


if __name__ == '__main__':
  main()
