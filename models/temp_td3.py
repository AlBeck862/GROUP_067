import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor
    """
    def __init__(self, state_dim, action_dim, max_action, dims_inner = [400, 300], activation=F.relu):
        super(Actor, self).__init__()

        dims =  [state_dim]+dims_inner+[action_dim]
        self.layers = nn.ModuleList([])
        for i in range(1,len(dims)):
            self.layers.append(nn.Linear(dims[i-1], dims[i]))

        self.max_action = max_action
        self.activation = activation

    def forward(self, x):
        for i,layer in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = self.activation(layer(x))
            else:
                x = self.max_action * torch.tanh(layer(x)) 
       
        return x


class Critic(nn.Module):
    """Critic
    """
    
    def __init__(self, state_dim, action_dim, dims_inner = [400, 300], activation=F.relu):
        super(Critic, self).__init__()

        dims =  [state_dim+action_dim]+dims_inner+[1]
        self.activation = activation

        #Q1 architecture
        self.layers_Q1 = nn.ModuleList([])
        for i in range(1,len(dims)):
            self.layers_Q1.append(nn.Linear(dims[i-1], dims[i]))


        #Q2 architecture
        self.layers_Q2 = nn.ModuleList([])
        for i in range(1,len(dims)):
            self.layers_Q2.append(nn.Linear(dims[i-1], dims[i]))

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = xu
        for i,layer in enumerate(self.layers_Q1):
            if i < len(self.layers_Q1) - 1:
                x1 = self.activation(layer(x1))
            else:
                x1 = layer(x1)

        x2 = xu
        for i,layer in enumerate(self.layers_Q2):
            if i < len(self.layers_Q2) - 1:
                x2 = self.activation(layer(x2))
            else:
                x2 = layer(x2)
        
        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = xu
        for i,layer in enumerate(self.layers_Q1):
            if i < len(self.layers_Q1) - 1:
                x1 = self.activation(layer(x1))
            else:
                x1 = layer(x1)
        return x1




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






class Agent(object):
    """Agent class that handles the training of the networks and provides outputs as actions
        lr = 1e-3
        batch_size = 100
        weight_decay = 0
        tdg_error_weight = 0
        td_error_weight = 1
    """

    def __init__(self, env_specs, max_action=1, pretrained=False, num_grad_updates = 1,tdg_error_weight = 0, td_error_weight = 1, lr=1e-3, weight_decay=0, dims_inner = [400, 300], activation_critic=F.relu, activation_actor=F.relu, batch_size=100):
        self.env_specs = env_specs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dim = self.env_specs['observation_space'].shape[0]
        action_dim = self.env_specs['action_space'].shape[0]
        self.update_num = 5

        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = 5
        self.batch_size = batch_size

        #mage
        self.tdg_error_weight = tdg_error_weight
        self.td_error_weight = td_error_weight
        self.update_ = num_grad_updates

        self.actor = Actor(state_dim, action_dim, max_action, dims_inner=dims_inner, activation=activation_actor).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, dims_inner=dims_inner, activation=activation_actor).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay)#torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim, action_dim, dims_inner=dims_inner, activation=activation_critic).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, dims_inner=dims_inner, activation=activation_critic).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=self.weight_decay)#torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        print('------------------------- Training --------------------------')
        print('Number of batch gradient descent per updates: ', num_grad_updates)
        print('batch size: ', batch_size)
        print('optimizer: ', self.actor_optimizer)
        print('\n-------------------------- Actor ---------------------------')
        print(activation_actor)
        print(self.actor)
        print('\n------------------------- Critic ---------------------------')
        print(activation_critic)
        print(self.critic)
        print('tdg_error_weight: ', self.tdg_error_weight)

        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action
        self.buffer_start = 1000
        self.it = 0
        self.pretrained = pretrained
        self.create_graph = False

    def load_weights(self, root_path):
        directory = root_path+'weights'
        filename = 'TD3'
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

    def save(self, filename='', directory=''):
            torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, 'TD3_'+filename))
            torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, 'TD3_'+filename))

    def act(self, curr_obs, mode='eval', noise=0.1):
        """Select an appropriate action from the agent policy
        
            Args:
                curr_obs (array): current state of environment
                noise (float): how much noise to add to acitons
                
            Returns:
                action (float): action clipped within action range
        
        """
        
        state = torch.FloatTensor(curr_obs.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()

        #add noise
        if noise != 0 and mode == 'train': 
            action = (action + np.random.normal(0, noise, size=self.env_specs['action_space'].shape[0]))

        #exploratory start
        if mode == 'train' and not self.pretrained and len(self.replay_buffer.storage) < self.buffer_start:
            action = self.env_specs['action_space'].sample()
            
        return action.clip(-1, 1)

    def update(self, curr_obs, action, reward, next_obs, done, timestep, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        #iteration
        self.it += 1

        batch_size=self.batch_size

        self.replay_buffer.add((curr_obs, next_obs, action, reward, done))

        if len(self.replay_buffer.storage) > self.buffer_start:

          # Sample replay buffer storage
          for _ in range(self.update_):
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            action.requires_grad_()
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            #zero tensor the same size as target Q
            zero_targets = torch.zeros_like(target_Q, device=self.device)

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            #compute TD errors
            #print('Q1:',current_Q1.retains_grad)
            q1_td_error, q2_td_error = target_Q - current_Q1, target_Q - current_Q2

            # Compute critic loss
            #critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 
            critic_loss, standard_loss, gradient_loss = torch.tensor(0, device=self.device), torch.tensor(0, device=self.device), torch.tensor(0, device=self.device)
            if self.td_error_weight > 0:
              standard_loss = 0.5 * (F.smooth_l1_loss(q1_td_error, zero_targets) + F.smooth_l1_loss(q2_td_error, zero_targets))
              critic_loss = critic_loss + self.td_error_weight * standard_loss

            if self.tdg_error_weight > 0:
              # Compute gradient critic loss
              #print('hello')
              gradients_error_norms1 = torch.autograd.grad(outputs=q1_td_error, inputs=action,
                                                          grad_outputs=torch.ones(q1_td_error.size(), device=self.device),
                                                          retain_graph=True, create_graph=True,
                                                          only_inputs=True)[0].flatten(start_dim=1).norm(dim=1, keepdim=True)
              gradients_error_norms2 = torch.autograd.grad(outputs=q2_td_error, inputs=action,
                                                          grad_outputs=torch.ones(q2_td_error.size(), device=self.device),
                                                          retain_graph=True, create_graph=True,
                                                          only_inputs=True)[0].flatten(start_dim=1).norm(dim=1, keepdim=True)

              gradient_loss = 0.5 * (F.smooth_l1_loss(gradients_error_norms1, zero_targets) + F.smooth_l1_loss(gradients_error_norms2, zero_targets))
              critic_loss = critic_loss + self.tdg_error_weight * gradient_loss                                      

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(create_graph = self.create_graph)
            torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward(create_graph = self.create_graph)
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

              






# import numpy as np
# import torch
# import torch.nn.functional as F
# import torch.nn as nn



# def hidden_init(layer):
#     fan_in = layer.weight.data.size()[0]
#     lim = 1. / np.sqrt(fan_in)
#     return (-lim, lim)

# class Actor(nn.Module):
#     """Initialize parameters and build model.
#         Args:
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             max_action (float): highest action to take
#             seed (int): Random seed
#             h1_units (int): Number of nodes in first hidden layer
#             h2_units (int): Number of nodes in second hidden layer
            
#         Return:
#             action output of network with tanh activation
#     """
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()

#         self.l1 = nn.Linear(state_dim, 400)
#         self.l2 = nn.Linear(400, 300)
#         self.l3 = nn.Linear(300, action_dim)

#         self.max_action = max_action

#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = self.max_action * torch.tanh(self.l3(x)) 
#         return x


# class Critic(nn.Module):
#     """Initialize parameters and build model.
#         Args:
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             max_action (float): highest action to take
#             seed (int): Random seed
#             h1_units (int): Number of nodes in first hidden layer
#             h2_units (int): Number of nodes in second hidden layer
            
#         Return:
#             value output of network 
#     """
    
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()

#         # Q1 architecture
#         self.l1 = nn.Linear(state_dim + action_dim, 400)
#         self.l2 = nn.Linear(400, 300)
#         self.l3 = nn.Linear(300, 1)

#         # Q2 architecture
#         self.l4 = nn.Linear(state_dim + action_dim, 400)
#         self.l5 = nn.Linear(400, 300)
#         self.l6 = nn.Linear(300, 1)

#     def forward(self, x, u):
#         xu = torch.cat([x, u], 1)

#         x1 = F.relu(self.l1(xu))
#         x1 = F.relu(self.l2(x1))
#         x1 = self.l3(x1)

#         x2 = F.relu(self.l4(xu))
#         x2 = F.relu(self.l5(x2))
#         x2 = self.l6(x2)
#         return x1, x2

#     def Q1(self, x, u):
#         xu = torch.cat([x, u], 1)

#         x1 = F.relu(self.l1(xu))
#         x1 = F.relu(self.l2(x1))
#         x1 = self.l3(x1)
#         return x1




# class ReplayBuffer(object):
#     """Buffer to store tuples of experience replay"""
    
#     def __init__(self, max_size=1000000):
#         """
#         Args:
#             max_size (int): total amount of tuples to store
#         """
        
#         self.storage = []
#         self.max_size = max_size
#         self.ptr = 0

#     def add(self, data):
#         """Add experience tuples to buffer
        
#         Args:
#             data (tuple): experience replay tuple
#         """
        
#         if len(self.storage) == self.max_size:
#             self.storage[int(self.ptr)] = data
#             self.ptr = (self.ptr + 1) % self.max_size
#         else:
#             self.storage.append(data)

#     def sample(self, batch_size):
#         """Samples a random amount of experiences from buffer of batch size
        
#         Args:
#             batch_size (int): size of sample
#         """
        
#         #print(len(self.storage))
#         ind = np.random.randint(0, len(self.storage), size=batch_size)
#         states, actions, next_states, rewards, dones = [], [], [], [], []

#         for i in ind: 
#             s, a, s_, r, d = self.storage[i]
#             states.append(np.array(s, copy=False))
#             actions.append(np.array(a, copy=False))
#             next_states.append(np.array(s_, copy=False))
#             rewards.append(np.array(r, copy=False))
#             dones.append(np.array(d, copy=False))

#         return np.array(states), np.array(actions), np.array(next_states), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)
















# class Agent(object):
#     """Agent class that handles the training of the networks and provides outputs as actions
#     """

#     def __init__(self, env_specs, max_action=1, pretrained=False, lr=1e-3):
#         self.env_specs = env_specs
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         state_dim = self.env_specs['observation_space'].shape[0]
#         action_dim = self.env_specs['action_space'].shape[0]

#         self.lr = lr

#         self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
#         self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
#         self.actor_target.load_state_dict(self.actor.state_dict())
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

#         self.critic = Critic(state_dim, action_dim).to(self.device)
#         self.critic_target = Critic(state_dim, action_dim).to(self.device)
#         self.critic_target.load_state_dict(self.critic.state_dict())
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

#         self.replay_buffer = ReplayBuffer()
#         self.max_action = max_action
#         self.buffer_start = 1000
#         self.it = 0
#         self.pretrained = pretrained

#     def load_weights(self, root_path):
#         directory = root_path+'weights'
#         filename = 'TD3'
#         self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
#         self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))

#     def save(self, filename='', directory=''):
#             torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, 'TD3_'+filename))
#             torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, 'TD3_'+filename))

#     def act(self, curr_obs, mode='eval', noise=0.1):
#         """Select an appropriate action from the agent policy
        
#             Args:
#                 curr_obs (array): current state of environment
#                 noise (float): how much noise to add to acitons
                
#             Returns:
#                 action (float): action clipped within action range
        
#         """
        
#         state = torch.FloatTensor(curr_obs.reshape(1, -1)).to(self.device)
#         action = self.actor(state).cpu().data.numpy().flatten()

#         #add noise
#         if noise != 0 and mode == 'train': 
#             action = (action + np.random.normal(0, noise, size=self.env_specs['action_space'].shape[0]))

#         #exploratory start
#         if mode == 'train' and not self.pretrained and len(self.replay_buffer.storage) < self.buffer_start:
#             action = self.env_specs['action_space'].sample()
            
#         return action.clip(-1, 1)

#     def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
#         #iteration
#         self.it += 1
        
#         self.replay_buffer.add((curr_obs, next_obs, action, reward, done))

#         if len(self.replay_buffer.storage) > self.buffer_start:
#           # Sample replay buffer storage
#           x, y, u, r, d = self.replay_buffer.sample(batch_size)
#           state = torch.FloatTensor(x).to(self.device)
#           action = torch.FloatTensor(u).to(self.device)
#           next_state = torch.FloatTensor(y).to(self.device)
#           done = torch.FloatTensor(1 - d).to(self.device)
#           reward = torch.FloatTensor(r).to(self.device)

#           # Select action according to policy and add clipped noise 
#           noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
#           noise = noise.clamp(-noise_clip, noise_clip)
#           next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

#           # Compute the target Q value
#           target_Q1, target_Q2 = self.critic_target(next_state, next_action)
#           target_Q = torch.min(target_Q1, target_Q2)
#           target_Q = reward + (done * discount * target_Q).detach()

#           # Get current Q estimates
#           current_Q1, current_Q2 = self.critic(state, action)

#           # Compute critic loss
#           critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

#           # Optimize the critic
#           self.critic_optimizer.zero_grad()
#           critic_loss.backward()
#           self.critic_optimizer.step()

#           # Delayed policy updates
#           if self.it % policy_freq == 0:

#               # Compute actor loss
#               actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

#               # Optimize the actor 
#               self.actor_optimizer.zero_grad()
#               actor_loss.backward()
#               self.actor_optimizer.step()

#               # Update the frozen target models
#               for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#                   target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

#               for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#                   target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
