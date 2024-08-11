import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

# Define the DQN model class
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions): 
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Create the LunarLander environment
env = gym.make('Acrobot-v1', render_mode='human')
state, info = env.reset()

# Get the number of observations and actions from the environment
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

# Set up device
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Create the model and load the saved state dictionary
policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load('acrobot_model2.pth'))
policy_net.eval()  # Set the model to evaluation mode

# Define the action selection function
def select_action(state):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy_net(state)
        print(action)
        print(action.shape)
        action = action.max(1).indices.item()
        print(action)
        # Add noise to the action based on the noise probability
        
        return int(action)

# Define the function to add random perturbation to the state


# Run the environment with the loaded model and perturbed observations
state, info = env.reset()

for _ in range(10000):
    # Perturb the state observation
    
    # Select action using the policy network
    action = select_action(state)
    
    # Step the environment and get the next state and reward
    state, reward, terminated, truncated, info = env.step(action)
    
    # Render the environment
    env.render()
    
    # Check if the episode is finished
    if terminated or truncated:
        break

# Close the environment
env.close()
