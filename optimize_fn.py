import torch
from torch import nn

def optimize_batch(mini_batch, policy_net, target_net, device, loss_fn, optimizer):
        states, actions, next_states, rewards, dones = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)
        dones = torch.tensor(dones).float().to(device)

        gamma = 0.99
        with torch.no_grad():
            best_actions = policy_net(next_states).argmax(dim=1)
            target_q = rewards + (1 - dones) * gamma * \
                target_net(next_states).gather(dim=1, index=best_actions.unsqueeze(1)).squeeze()

        # current Q estimates
        current_q = policy_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        loss = loss_fn(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
