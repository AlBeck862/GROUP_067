import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

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


class ReplayBufferContext(object):
    """Buffer to store tuples of experience replay"""
    
    def __init__(self, max_size=1000000, sentence_size=10):
        """
        Args:
            max_size (int): total amount of tuples to store
        """
        
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.gru_sentences = []
        self.current_episode = []
        self.sentence_size = sentence_size

    def _add_episode(self, data):
      (h1, curr_obs, h2, next_obs, action, reward, done) = data
      # print('current')
      # print(curr_obs.shape)
      # print('action')
      # print(action.shape)
      # print('next obs')
      # print(next_obs.shape)
      # print('reward')
      # print(reward.shape)
      self.current_episode.append((curr_obs, action, next_obs, reward))
      if done:
        self.append_gru_data(self.current_episode)
        self.current_episode = []

    def append_gru_data(self, episode):
      if len(episode) >= self.sentence_size:
        self.gru_sentences += (self._create_sentences(episode))

    def _create_sentences(self, episode):
      return [[episode[i+j] for j in range(self.sentence_size)] for i in range(len(episode)-self.sentence_size+1)]

    def sample_gru_data(self, batch_size):
      #(L,N,Hstate+Haction), #(L,N,Hstate+1)

      # current
      # (11,)
      # action
      # (3,)
      # next obs
      # (11,)
      # reward
      # ()

      ind = np.random.randint(0, len(self.gru_sentences), size=batch_size)
      l = [self.gru_sentences[i] for i in ind]
      states = np.asarray([[word[0] for word in sentence] for sentence in l])
      actions = np.asarray([[word[1] for word in sentence] for sentence in l]) 
      next_states = np.asarray([[word[2] for word in sentence] for sentence in l])
      rewards = np.expand_dims(np.asarray([[word[3] for word in sentence] for sentence in l]), axis=2)
      #hiddens = np.asarray([[word[4] for word in sentence] for sentence in l])
      # print(states.shape)
      # print(actions.shape)
      # print(next_states.shape)
      # print(rewards.shape)
      inputs = np.concatenate((states, actions), axis=2)#.swapaxes(0,1)
      ouputs = np.concatenate((next_states, rewards), axis=2)#.swapaxes(0,1)
      # (100, 10, 11)
      # (100, 10, 3)
      # (100, 10, 11)
      # (100, 10, 1)
      # np.swapaxes
      #print(hiddens.shape)
      # print(inputs.shape)
      # print(ouputs.shape)
      # (100, 10, 2, 1, 100)
      # (10, 100, 14)
      # (10, 100, 12)
      return inputs, ouputs
      

    def add(self, data):
        """Add experience tuples to buffer
        
        Args:
            data (tuple): experience replay tuple
        """
        self._add_episode(data)
        
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
        hidden1, hidden2, states, actions, next_states, rewards, dones = [], [], [], [], [], [], []

        for i in ind: 
            h1, s, h2, s_, a, r, d = self.storage[i]
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            next_states.append(np.array(s_, copy=False))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False))

            #hidden states
            hidden1.append(np.array(h1, copy=False))
            hidden2.append(np.array(h2, copy=False))

        states = np.array(states)
        #print(states.shape)
        #print(hidden1[0].shape)
        hidden1 = np.concatenate(hidden1, axis=1)
        hidden1 = np.swapaxes(hidden1, 0, 1).reshape(batch_size, -1)
        next_states = np.array(next_states)
        hidden2 = np.concatenate(hidden2, axis=1)
        hidden2 = np.swapaxes(hidden2, 0, 1).reshape(batch_size, -1)
        actions = np.array(actions)
        rewards = np.array(rewards).reshape(-1, 1)
        dones = np.array(dones).reshape(-1, 1)

        return hidden1, states, hidden2, next_states, actions, rewards, dones


class ContextGRU(nn.Module):
    """Initialize parameters and build model.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=100, drop_prob=0, n_layers=1):
        super(ContextGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        #inputs is a vector containing action taken + current state
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)

        #first ouput is the next state 
        self.fc = nn.Linear(hidden_dim, state_dim+1)
        
        
    def forward(self, x, h):
        '''
        Arguments:
          x: (N,L,Hin)
          h: (num_layers,N, Hhidden)
        '''
        # print(h.get_device())
        # print(x.get_device())

        #pass input current state and action through gru
        out, h = self.gru(x, h)

        #next state/reward is computed with linear activation since it is regression
        # print(out.shape)
        # print(out[:,-1].shape)
        next_state_reward = self.fc(out)

        return next_state_reward, h


class TD3Context(object):
    """Agent class that handles the training of the networks and provides outputs as actions
    """

    def __init__(self, env_specs, max_action=1, pretrained=False, lr=1e-3, lrc=1e-3, dims_inner = [400, 300], activation_critic=F.relu, activation_actor=F.relu, n_layers=1, hidden_dim=100):
        self.env_specs = env_specs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state_dim = self.env_specs['observation_space'].shape[0]
        action_dim = self.env_specs['action_space'].shape[0]

        self.lr = lr
        self.batch_size = 100
        self.GRU_loss = nn.MSELoss()
        self.context = ContextGRU(state_dim, action_dim, hidden_dim=hidden_dim, drop_prob=0, n_layers=n_layers).to(self.device)
        self.context_optimizer = torch.optim.Adam(self.context.parameters(), lr=lrc)

        self.context_hidden = np.zeros((n_layers, 1,hidden_dim))
        self.context_hidden_next = np.zeros((n_layers, 1,hidden_dim))
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.actor = Actor(state_dim+hidden_dim*n_layers, action_dim, max_action, dims_inner=dims_inner, activation=activation_actor).to(self.device)
        self.actor_target = Actor(state_dim+hidden_dim*n_layers, action_dim, max_action, dims_inner=dims_inner, activation=activation_actor).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic = Critic(state_dim+hidden_dim*n_layers, action_dim, dims_inner=dims_inner, activation=activation_critic).to(self.device)
        self.critic_target = Critic(state_dim+hidden_dim*n_layers, action_dim, dims_inner=dims_inner, activation=activation_critic).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        print('-------------------------- Context ---------------------------')
        print(self.context)
        print('\n-------------------------- Actor ---------------------------')
        print(activation_actor)
        print(self.actor)
        print('\n------------------------- Critic ---------------------------')
        print(activation_critic)
        print(self.critic)

        self.replay_buffer = ReplayBufferContext()
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
        batch_size = self.batch_size

        #update current hidden context
        self.context_hidden = self.context_hidden_next

        #state tensor input to the actor
        state = torch.FloatTensor(curr_obs.reshape(1, -1)).to(self.device)
        hidden = torch.FloatTensor(np.swapaxes(self.context_hidden, 0, 1).reshape(1, -1)).to(self.device)

        #predict an  action with the actor
        # print(state.shape)
        # print(hidden.shape)
        state_hidden_context = torch.cat([state, hidden], 1)
        action = self.actor(state_hidden_context).cpu().data.numpy().flatten()

        #update the next hidden context
        #convert current state and hidden context to torch tensor
        state = torch.FloatTensor(curr_obs.reshape((1,1,len(curr_obs)))).to(self.device)  #(L,N,Hstate)
        hidden = torch.FloatTensor(self.context_hidden).to(self.device)                   #(num_layers,N, Hhidden)
        act = torch.FloatTensor(action.reshape((1,1,len(action)))).to(self.device)        #(L,N,Haction)

        #concatenate the state and action
        x = torch.cat([state, act], 2)
        x = x.to(self.device)

        # print(x.get_device())
        # print(hidden.get_device())

        #get next hidden context from context network
        _, self.context_hidden_next = self.context(x, hidden)

        #convert next hidden context back to numpy cpu
        self.context_hidden_next = self.context_hidden_next.cpu().data.numpy()

        #add noise
        if noise != 0 and mode == 'train': 
            action = (action + np.random.normal(0, noise, size=self.env_specs['action_space'].shape[0]))

        #exploratory start
        if mode == 'train' and not self.pretrained and len(self.replay_buffer.storage) < self.buffer_start:
            action = self.env_specs['action_space'].sample()
            
        return action.clip(-1, 1)

    def context_update(self, batch_size):
        #retrieve data for gru training
        input_sequences, output_sequences = self.replay_buffer.sample_gru_data(batch_size) #(L,N,Hstate+Haction), #(L,N,Hstate+1)

        input_sequences = torch.Tensor(input_sequences).to(self.device)
        output_sequences = torch.Tensor(output_sequences).to(self.device)
        h = torch.Tensor(np.zeros((self.n_layers, batch_size, self.hidden_dim))).to(self.device)

        #get predictions
        predictions,_ = self.context(input_sequences, h)

        # print(predictions.shape)
        # print(output_sequences.shape)

        #compute the loss for the GRU context
        context_loss = self.GRU_loss(predictions, output_sequences)

        if self.it % 10000 == 0:
          print('\ncontext - loss')
          print(context_loss)

        #optimization of context network
        self.context_optimizer.zero_grad()
        context_loss.backward()
        self.context_optimizer.step()
        

    def update(self, curr_obs, action, reward, next_obs, done, timestep, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        batch_size = self.batch_size
        
        #iteration
        self.it += 1

        #store in buffer
        self.replay_buffer.add((self.context_hidden, curr_obs, self.context_hidden_next, next_obs, action, reward, done))

        #update hidden context for next iteration
        if done:
          #if done the reset the hidden states
          self.context_hidden = np.zeros((self.n_layers, 1,self.hidden_dim))
          self.context_hidden_next = np.zeros((self.n_layers, 1,self.hidden_dim))

        if len(self.replay_buffer.gru_sentences) >= batch_size:
          self.context_update(batch_size)

        if len(self.replay_buffer.storage) > self.buffer_start:
          # Sample replay buffer storage
          # hidden1, states, hidden2, next_states, actions, rewards, dones
          for _ in range(self.updates_):
            h1, x, h2, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(1 - d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            #hidden states for actor
            hidden1 = torch.FloatTensor(h1).to(self.device) #(N ,Hhidden*numl)
            hidden2 = torch.FloatTensor(h2).to(self.device) #(N ,Hhidden*numl)

            #append hidden context to states
            # print(state.shape)
            # print(hidden1.shape)

            state = torch.cat([state, hidden1], 1).to(self.device)
            next_state = torch.cat([next_state, hidden2], 1).to(self.device)

            # print(state.shape)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(self.device)
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

            # if self.it % 1000 == 0:
            #   print('\ncritic- loss')
            #   print(critic_loss)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(create_graph = self.create_graph)
            self.critic_optimizer.step()

            # Delayed policy updates
            if self.it % policy_freq == 0:

                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

                # if self.it % 1000 == 0:
                #   print('\nactor - loss')
                #   print(actor_loss)

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward(create_graph = self.create_graph)
                self.actor_optimizer.step()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)