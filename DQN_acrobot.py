import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np  

class DQN(nn.Module):
    def __init__(self, lr, n_obs, n_actions):
        super(DQN, self).__init__()
        self.lr = lr
        self.n_obs = n_obs
        self.n_actions = n_actions

        # Layers
        self.layer1 = nn.Linear(self.n_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        actions = self.layer3(x)
        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, n_obs, batch_size, n_actions, max_memsize=10000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_obs = n_obs
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.max_memsize = max_memsize
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.mem_cntr = 0
        self.action_space = [i for i in range(n_actions)]
        self.Q_policy = DQN(lr, n_obs, n_actions)
        self.Q_target = DQN(lr, n_obs, n_actions)
        self.Q_policy.load_state_dict(T.load('acrobot_model.pth'))
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.state_memory = np.zeros((self.max_memsize, n_obs), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_memsize, n_obs), dtype=np.float32)
        self.action_memory = np.zeros(self.max_memsize, dtype=np.int64)  # Use np.int64 for action memory
        self.reward_memory = np.zeros(self.max_memsize, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_memsize, dtype=np.bool_)
    
    # Store memory
    def store_experience(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.max_memsize
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            obs = np.array(obs, dtype=np.float32)
            state = T.from_numpy(obs).to(self.Q_policy.device)
            actions = self.Q_policy.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        
        self.Q_policy.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.max_memsize)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_policy.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_policy.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_policy.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_policy.device)
        action_batch = T.tensor(self.action_memory[batch], dtype=T.int64).to(self.Q_policy.device)

        # Q value
        q_eval = self.Q_policy.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_policy.loss(q_eval, q_target).to(self.Q_policy.device)
        loss.backward()
        self.Q_policy.optimizer.step()

        # Epsilon decay
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        
        # Update target network
        q_target_state_dict = self.Q_target.state_dict()
        q_policy_state_dict = self.Q_policy.state_dict()
        for key in q_policy_state_dict:
            q_target_state_dict[key] = q_policy_state_dict[key] * 0.005 + q_target_state_dict[key] * 0.995
        self.Q_target.load_state_dict(q_target_state_dict)
