import gymnasium as gym
import torch

import flappy_bird_gymnasium

from dqn_architecture import DQN


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

env = gym.make("FlappyBird-v0", render_mode='human', use_lidar=False)
act_count = env.action_space.n
state_dim = env.observation_space.shape[0]

policy_net = DQN(state_dim, act_count, hidden_dim=256, enable_dueling=True).to(device)
try:
    policy_net.load_state_dict(torch.load("best_model.pt", map_location=device))
    print("model loaded")
except FileNotFoundError:
    print("best_model.pt missing")
    exit()
policy_net.eval()

rewards = []
episode = 0

try:
    while True:  # run until killed
        s, _ = env.reset()
        s = torch.tensor(s, dtype=torch.float, device=device)
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                a = policy_net(s.unsqueeze(0)).squeeze().argmax()
            nxt, r, done, truncated, info = env.step(a.item())
            ep_reward += r
            s = torch.tensor(nxt, dtype=torch.float, device=device)
        rewards.append(ep_reward)
        episode += 1
        print(f"episode {episode} reward {ep_reward}")
except KeyboardInterrupt:
    pass

# simple plot
import matplotlib.pyplot as plt
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('test_rewards.png')
print('rewards saved', rewards)
