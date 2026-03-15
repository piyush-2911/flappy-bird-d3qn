import gymnasium as gym

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn

from datetime import datetime
import flappy_bird_gymnasium

from replay_buffer import ReplayMemory
from dqn_architecture import DQN
from optimize_fn import optimize_batch

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# environment
env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]

# training settings
rewards_per_episode = []
epsilon = 1.0
decay = 0.9999
min_epsilon = 1e-5
batch_size = 32
sync_steps = 10
epsilon_history = []
replay = ReplayMemory(maxlen=100000)

# networks
policy_net = DQN(num_states, num_actions, hidden_dim=256, enable_dueling=True).to(device)
target_net = DQN(num_states, num_actions, hidden_dim=256, enable_dueling=True).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
step_count = 0
best_reward = -float('inf')


episode = 0
while episode < 2500:
    s, _ = env.reset()
    s = torch.tensor(s, dtype=torch.float, device=device)
    done = False
    ep_reward = 0.0

    while not done and ep_reward < 999:
        # epsilon-greedy action
        if random.random() < epsilon:
            action = torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)
        else:
            with torch.no_grad():
                action = policy_net(s.unsqueeze(0)).squeeze().argmax()

        nxt_np, r, done, truncated, info = env.step(action.item())
        ep_reward += r

        nxt = torch.tensor(nxt_np, dtype=torch.float, device=device)
        r_t = torch.tensor(r, dtype=torch.float, device=device)

        replay.append((s, action, nxt, r_t, done))
        s = nxt
        step_count += 1

        if len(replay) > batch_size:
            batch = replay.sample(batch_size)
            optimize_batch(batch, policy_net, target_net, device, loss_fn, optimizer)
            epsilon = max(epsilon * decay, min_epsilon)
            epsilon_history.append(epsilon)
            if step_count >= sync_steps:
                target_net.load_state_dict(policy_net.state_dict())
                step_count = 0

    rewards_per_episode.append(ep_reward)
    if ep_reward > best_reward:
        diff = 0 if best_reward == -float('inf') else ((ep_reward - best_reward) / abs(best_reward) * 100)
        print(f"{datetime.now().strftime('%m-%d %H:%M:%S')}: New best reward {ep_reward:.1f} ({diff:+.1f}%) at episode {episode}")
        torch.save(policy_net.state_dict(), "best_model.pt")
        best_reward = ep_reward
    episode += 1


def save_graph(rewards, eps_history):
    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    # rewards vs episodes
    axes[0].plot(rewards)
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Rewards')

    # epsilon decay
    axes[1].plot(eps_history)
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Epsilon')

    plt.tight_layout()
    fig.savefig("graph_rewards.png")

save_graph(rewards_per_episode, epsilon_history)
