import math
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

######## NETWORKS #######

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer
            
        Return:
            action output of network with tanh activation
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, action_dim)
        #self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = self.l2(x)
        #x = self.max_action * torch.tanh(self.l3(x)) 
        return x


class Critic(nn.Module):
    """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer
            
        Return:
            value output of network 
    """
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.ELU(state_dim + action_dim, 64)
        self.l2 = nn.Linear(14, 1)
        #self.l3 = nn.Linear(300, 1)

        # Q2 architecture
        self.l3 = nn.ELU(state_dim + action_dim, 64)
        self.l4 = nn.Linear(14, 1)
        #self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = self.l1(xu)
        #print(type(x1))
        #print(x1.shape)
        x1 = self.l2(x1)
        #print(x1.shape)
        #x1 = self.l3(x1)

        x2 = self.l3(xu)
        x2 = self.l4(x2)
        #x2 = self.l6(x2)
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = self.l1(xu)
        x1 = self.l2(x1)
        #x1 = self.l3(x1)
        return x1



############## REPLAY BUFFER ##############
class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""
    
    def __init__(self, max_size=1000000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """
        
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer
        
        Args:
            data (tuple): experience replay tuple
        """
        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size
        
        Args:
            batch_size (int): size of sample
        """
        
        #print(len(self.storage))
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind: 
            s, a, s_, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

        return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############# KFAC OPTIMIZER ##############

import glob
import os

import torch
import torch.nn as nn

'''from a2c_ppo_acktr.envs import VecNormalize


# Get a render function
def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None'''


# Necessary for KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

def _extract_patches(x, kernel_size, stride, padding):
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def compute_cov_a(a, classname, layer_info, fast_cnn):
    batch_size = a.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            a = _extract_patches(a, *layer_info)
            a = a.view(a.size(0), -1, a.size(-1))
            a = a.mean(1)
        else:
            a = _extract_patches(a, *layer_info)
            a = a.view(-1, a.size(-1)).div_(a.size(1)).div_(a.size(2))
    elif classname == 'AddBias':
        is_cuda = a.is_cuda
        a = torch.ones(a.size(0), 1)
        if is_cuda:
            a = a.cuda()

    return a.t() @ (a / batch_size)


def compute_cov_g(g, classname, layer_info, fast_cnn):
    batch_size = g.size(0)

    if classname == 'Conv2d':
        if fast_cnn:
            g = g.view(g.size(0), g.size(1), -1)
            g = g.sum(-1)
        else:
            g = g.transpose(1, 2).transpose(2, 3).contiguous()
            g = g.view(-1, g.size(-1)).mul_(g.size(1)).mul_(g.size(2))
    elif classname == 'AddBias':
        g = g.view(g.size(0), g.size(1), -1)
        g = g.sum(-1)

    g_ = g * batch_size
    return g_.t() @ (g_ / g.size(0))


def update_running_stat(aa, m_aa, momentum):
    # Do the trick to keep aa unchanged and not create any additional tensors
    m_aa *= momentum / (1 - momentum)
    m_aa += aa
    m_aa *= (1 - momentum)


class SplitBias(nn.Module):
    def __init__(self, module):
        super(SplitBias, self).__init__()
        self.module = module
        self.add_bias = AddBias(module.bias.data)
        self.module.bias = None

    def forward(self, input):
        x = self.module(input)
        x = self.add_bias(x)
        return x
           

class KFACOptimizer(torch.optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.003,
                 momentum=0.9,
                 stat_decay=0.99,
                 kl_clip=0.001,
                 damping=1e-2,
                 weight_decay=0,
                 fast_cnn=False,
                 Ts=1,
                 Tf=10):
        defaults = dict()

        def split_bias(module):
            for mname, child in module.named_children():
                if hasattr(child, 'bias') and child.bias is not None:
                    module._modules[mname] = SplitBias(child)
                else:
                    split_bias(child)

        split_bias(model)

        super(KFACOptimizer, self).__init__(model.parameters(), defaults)

        self.known_modules = {'Linear', 'Conv2d', 'AddBias'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}

        self.momentum = momentum
        self.stat_decay = stat_decay

        self.lr = lr
        self.kl_clip = kl_clip
        self.damping = damping
        self.weight_decay = weight_decay

        self.fast_cnn = fast_cnn

        self.Ts = Ts
        self.Tf = Tf

        self.optim = optim.SGD(
            model.parameters(),
            lr=self.lr * (1 - self.momentum),
            momentum=self.momentum)

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.Ts == 0:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            aa = compute_cov_a(input[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = aa.clone()

            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats:
            classname = module.__class__.__name__
            layer_info = None
            if classname == 'Conv2d':
                layer_info = (module.kernel_size, module.stride,
                              module.padding)

            gg = compute_cov_g(grad_output[0].data, classname, layer_info,
                               self.fast_cnn)

            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = gg.clone()

            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                assert not ((classname in ['Linear', 'Conv2d']) and module.bias is not None), \
                                    "You must have a bias as a separate layer"

                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)

    def step(self):
        print('optimiizing w KFAC')
        # Add weight decay
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(self.weight_decay, p.data)

        updates = {}
        for i, m in enumerate(self.modules):
            assert len(list(m.parameters())
                       ) == 1, "Can handle only one parameter at the moment"
            classname = m.__class__.__name__
            p = next(m.parameters())

            la = self.damping + self.weight_decay

            if self.steps % self.Tf == 0:
                # My asynchronous implementation exists, I will add it later.
                # Experimenting with different ways to this in PyTorch.
                self.d_a[m], self.Q_a[m] = torch.symeig(
                    self.m_aa[m], eigenvectors=True)
                self.d_g[m], self.Q_g[m] = torch.symeig(
                    self.m_gg[m], eigenvectors=True)

                self.d_a[m].mul_((self.d_a[m] > 1e-6).float())
                self.d_g[m].mul_((self.d_g[m] > 1e-6).float())

            if classname == 'Conv2d':
                p_grad_mat = p.grad.data.view(p.grad.data.size(0), -1)
            else:
                p_grad_mat = p.grad.data

            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (
                self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + la)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()

            v = v.view(p.grad.data.size())
            updates[p] = v

        vg_sum = 0
        for p in self.model.parameters():
            v = updates[p]
            vg_sum += (v * p.grad.data * self.lr * self.lr).sum()

        nu = min(1, math.sqrt(self.kl_clip / vg_sum))

        for p in self.model.parameters():
            v = updates[p]
            p.grad.data.copy_(v)
            p.grad.data.mul_(nu)

        self.optim.step()
        self.steps += 1

################# AGENT ############

class Agent(object):
        """Agent class that handles the training of the networks and provides outputs as actions
    
        Args:
            state_dim (int): state size
            action_dim (int): action size
            max_action (float): highest action to take
            device (device): cuda or cpu to process tensors
            env (env): gym environment to use"""

        def __init__(self, env_specs, max_action=1, pretrained=False):
            self.env_specs = env_specs
            state_dim = self.env_specs['observation_space'].shape[0]
            action_dim = self.env_specs['action_space'].shape[0]

            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = KFACOptimizer(self.actor)

            self.critic = Critic(state_dim, action_dim).to(device)
            self.critic_target = Critic(state_dim, action_dim).to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = KFACOptimizer(self.critic)

            self.replay_buffer = ReplayBuffer()
            self.max_action = max_action
            self.buffer_start = 1000
            self.it = 0

        def load_weights(self, root_path):
            directory = root_path#+'weights'
            filename = 'ACKTR'
            self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
            self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

        def save(self, filename='', directory=''):
            torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, 'ACKTR_'+filename))
            torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, 'ACKTR_'+filename))

        def act(self, curr_obs, mode='eval', noise=0.1):
            """Select an appropriate action from the agent policy
            
                Args:
                    curr_obs (array): current state of environment
                    noise (float): how much noise to add to acitons
                    
                Returns:
                    action (float): action clipped within action range
            
            """
            
            state = torch.FloatTensor(curr_obs.reshape(1, -1)).to(device)
            action = self.actor(state).cpu().data.numpy().flatten()

            #add noise
            if noise != 0 and mode == 'train': 
                action = (action + np.random.normal(0, noise, size=self.env_specs['action_space'].shape[0]))

            #exploratory start
            if mode == 'train' and len(self.replay_buffer.storage) < self.buffer_start:
              action = self.env_specs['action_space'].sample()
                
            return action.clip(-1, 1)

        def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=2500, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
            #iteration
            self.it += 1
            
            self.replay_buffer.add((curr_obs, next_obs, action, reward, done))

            if len(self.replay_buffer.storage) > self.buffer_start:
              # Sample replay buffer storage
              x, y, u, r, d = self.replay_buffer.sample(batch_size)
              state = torch.FloatTensor(x).to(device)
              action = torch.FloatTensor(u).to(device)
              next_state = torch.FloatTensor(y).to(device)
              done = torch.FloatTensor(1 - d).to(device)
              reward = torch.FloatTensor(r).to(device)

              # Select action according to policy and add clipped noise 
              noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
              noise = noise.clamp(-noise_clip, noise_clip)
              next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

              # Compute the target Q value
              target_Q1, target_Q2 = self.critic_target(next_state, next_action)
              target_Q = torch.min(target_Q1, target_Q2)
              target_Q = reward + (done * discount * target_Q).detach()

              # Get current Q estimates
              current_Q1, current_Q2 = self.critic(state, action)

              # Compute critic loss
              critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

              # Optimize the critic
              self.critic_optimizer.zero_grad()
              self.optimizer.acc_stats = True
              critic_loss.backward()
              self.optimizer.acc_stats = False
              self.critic_optimizer.step()

              # Delayed policy updates
              if self.it % policy_freq == 0:

                  # Compute actor loss
                  actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                  # Optimize the actor 
                  self.actor_optimizer.zero_grad()
                  self.optimizer.acc_stats = True
                  actor_loss.backward()
                  self.optimizer.acc_stats = False
                  self.actor_optimizer.step()

                  # Update the frozen target models
                  for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                  for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                      target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            #if self.optimizer.steps % self.optimizer.Ts == 0:
              

    
    
