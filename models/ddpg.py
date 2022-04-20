# import sys
# import mujoco_py
# import gym
# import time
import numpy as np
# import argparse
# import importlib
import random
# import tensorflow as tf
# import os
# from os import listdir, makedirs
# from os.path import isfile, join
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import matplotlib.pyplot as plt

buffer_size=1000000
batch_size=128  #this can be 128 for more complex tasks such as Hopper
gamma=0.9
tau=0.001       #Target Network HyperParameters
lr_a=0.0001      #LEARNING RATE ACTOR
lr_c=0.001       #LEARNING RATE CRITIC
H1=400   #neurons of 1st layers
H2=300   #neurons of 2nd layers

buffer_start = 100
epsilon = 1
epsilon_decay = 1./100000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # Assuming that we are on a CUDA machine, the following should print a CUDA device
print(device)

class replayBuffer(object):
  def __init__(self, buffer_size, name_buffer=''):
      self.buffer_size=buffer_size  
      self.num_exp=0
      self.buffer=deque()

  def add(self, s, a, r, t, s2):
      experience=(s, a, r, t, s2)
      if self.num_exp < self.buffer_size:
          self.buffer.append(experience)
          self.num_exp +=1
      else:
          self.buffer.popleft()
          self.buffer.append(experience)

  def size(self):
      return self.buffer_size

  def count(self):
      return self.num_exp

  def sample(self, batch_size):
      if self.num_exp < batch_size:
          batch=random.sample(self.buffer, self.num_exp)
      else:
          batch=random.sample(self.buffer, batch_size)

      s, a, r, t, s2 = map(np.stack, zip(*batch))

      return s, a, r, t, s2

  def clear(self):
      self.buffer = deque()
      self.num_exp=0

def fanin_(size):
  fan_in = size[0]
  weight = 1./np.sqrt(fan_in)
  return torch.Tensor(size).uniform_(-weight, weight)

class Critic(nn.Module):
  def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=3e-3):
      super(Critic, self).__init__()
              
      self.linear1 = nn.Linear(state_dim, h1)
      self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
            
      self.linear2 = nn.Linear(h1+action_dim, h2)
      self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
              
      self.linear3 = nn.Linear(h2, 1)
      self.linear3.weight.data.uniform_(-init_w, init_w)

      self.relu = nn.ReLU()
      
  def forward(self, state, action):
      x = self.linear1(state)
      x = self.relu(x)
      #print(f'state shape:{state.shape}')
      #print(f'action shape:{action.shape}')
     
      x = self.linear2(torch.cat([x,action],1))
      
      x = self.relu(x)
      x = self.linear3(x)
      
      return x
      

class Actor(nn.Module): 
  def __init__(self, state_dim, action_dim, h1=H1, h2=H2, init_w=0.003):
      super(Actor, self).__init__()        
      self.linear1 = nn.Linear(state_dim, h1)
      self.linear1.weight.data = fanin_(self.linear1.weight.data.size())
      
      
      self.linear2 = nn.Linear(h1, h2)
      self.linear2.weight.data = fanin_(self.linear2.weight.data.size())
              
      self.linear3 = nn.Linear(h2, action_dim)
      self.linear3.weight.data.uniform_(-init_w, init_w)

      self.relu = nn.ReLU()
      self.tanh = nn.Tanh()
      
  def forward(self, state):
      x = self.linear1(state)
      x = self.relu(x)
      x = self.linear2(x)
      x = self.relu(x)
      x = self.linear3(x)
      x = self.tanh(x)
      return x

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

state_dim = 11
action_dim = 3

#print("State dim: {}, Action dim: {}".format(state_dim, action_dim))

# Instantiating an object of each class
noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

critic  = Critic(state_dim, action_dim).to(device)
actor = Actor(state_dim, action_dim).to(device)

target_critic  = Critic(state_dim, action_dim).to(device)
target_actor = Actor(state_dim, action_dim).to(device)

for target_param, param in zip(target_critic.parameters(), critic.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_actor.parameters(), actor.parameters()):
    target_param.data.copy_(param.data)

memory = replayBuffer(buffer_size)
#writer = SummaryWriter() #initialise tensorboard writer

# Defining optimization parameters
q_optimizer  = torch.optim.Adam(critic.parameters(),  lr=lr_c)
policy_optimizer = torch.optim.Adam(actor.parameters(), lr=lr_a)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(q_optimizer, 'min',factor=0.2, patience=10)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(policy_optimizer, 'min',factor=0.2, patience=10)
MSE = nn.MSELoss()

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env_specs):
    self.env_specs = env_specs



  def load_weights(self):
    pass



  def act(self, curr_obs, mode='eval'):
    state  = torch.FloatTensor(curr_obs).to(device)
    action = actor.forward(state)
    #return self.env_specs['action_space'].sample()
    #print(type(action))
    return action.detach().cpu().numpy()



  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    torch.manual_seed(-1)
    
    #env = NormalizedEnv(gym.make(ENV_NAME))

    memory.add(curr_obs, action, reward, done, next_obs)
    #print(f'printing memory count {memory.count()}')
    #print("hello world")


    #keep adding experiences to the memory until there are at least minibatch size samples
        
    if memory.count() > buffer_start:
        #sampling state, action, reward, timestep and next states batches from memory buffer
        #transform them to tensors
        s_batch, a_batch, r_batch, t_batch, s2_batch = memory.sample(batch_size)

        s_batch = torch.FloatTensor(s_batch).to(device)
        a_batch = torch.FloatTensor(a_batch).to(device)
        r_batch = torch.FloatTensor(r_batch).unsqueeze(1).to(device)
        t_batch = torch.FloatTensor(np.float32(t_batch)).unsqueeze(1).to(device)
        # print(t_batch)
        s2_batch = torch.FloatTensor(s2_batch).to(device)
        
        
        #compute loss for critic
        a2_batch = target_actor(s2_batch)
        target_q = target_critic(s2_batch, a2_batch)
        #print(f'a2 shape:{a2_batch.shape}')
        y = r_batch + (1.0 - t_batch) * gamma * target_q.detach() #detach to avoid backprop target
        q = critic(s_batch, a_batch)
        #print(f'a shape:{a_batch.shape}')
        q_optimizer.zero_grad()
        q_loss = MSE(q, y) 
        #print(f'q_loss={q_loss}')
        q_loss.backward()
        q_optimizer.step()
        
        #compute loss for actor
        policy_optimizer.zero_grad()
        policy_loss = -critic(s_batch, actor(s_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        policy_optimizer.step()
        
        #soft update of the frozen target networks
        for target_param, param in zip(target_critic.parameters(), critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

        for target_param, param in zip(target_actor.parameters(), actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
