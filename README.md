# Flappy Bird AI: Dueling Double DQN

This repository contains a Reinforcement Learning agent trained to play Flappy Bird using a **Dueling Double Deep Q-Network (D3QN)** architecture. The agent achieves high-tier performance, clearing over **100 pipes within 2,500 training episodes**.

---

## Performance Demo

<p align="center">
<video src="demo.mp4" width="20%" autoplay loop muted playsinline></video>

<i>Agent performance from pipe 30 to 120. Peak Reward: 726.9</i>
</p>

---

## Technical Implementation

The implementation utilizes a **Dueling Double DQN** to improve learning stability and efficiency in a high-variance environment.

### Dueling Architecture
Separates the estimation of the state-value \(V(s)\) and the action-advantage \(A(s, a)\). This allows the model to identify valuable states (e.g., being centered in a gap) independently of specific actions.

### Double DQN
Addresses the overestimation bias of Q-values by using the **policy network for action selection** and the **target network for action evaluation**.

### Network Configuration
The model utilizes a **hidden dimension of 256 units**. This capacity is sufficient to approximate the physics and timing required for long-duration survival in the environment.

### Experience Replay
A replay buffer of size \(10^5\) is used to store and sample transitions, reducing temporal correlation during training updates.

---

## Requirements and Usage

### Installation

```bash
pip install gymnasium flappy-bird-gymnasium torch matplotlib
```

---

## Inference

The pre-trained weights are stored in `best_model.pt`.

To run the agent, initialize the DQN with `hidden_dim=256` and `enable_dueling=True`, then load the state dictionary.

```python
model = DQN(num_states, num_actions, hidden_dim=256, enable_dueling=True)
model.load_state_dict(torch.load("best_model.pt", map_location='cpu'))
model.eval()
```

---

## Project Structure

```
dqn_architecture.py : Model definition for the D3QN  
best_model.pt       : Saved weights of the highest-performing training iteration  
train_main.py       : Training script featuring epsilon-greedy exploration and target network synchronization  
replay_buffer.py    : Implementation of the experience replay mechanism
```