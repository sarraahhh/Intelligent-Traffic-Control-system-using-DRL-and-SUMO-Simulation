import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from model.dqn_model import DQN
from model.replay_buffer import ReplayBuffer
from training.env import TrafficEnv


EPISODES = 100
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.05
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 5
STATE_DIM = 16
ACTION_DIM = 2


os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = TrafficEnv()
model = DQN(STATE_DIM, ACTION_DIM)
target_model = DQN(STATE_DIM, ACTION_DIM)
target_model.load_state_dict(model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
buffer = ReplayBuffer()

episode_rewards = []
episode_losses = []
epsilon_history = []
vip_encounter_history = []
phase_switch_history = []
avg_rewards = []


plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line1, = ax1.plot([], [], linewidth=2)
line2, = ax1.plot([], [], linewidth=2)
line3, = ax2.plot([], [], linewidth=2)
plt.tight_layout()


epsilon = EPSILON_START

try:
    for ep in range(EPISODES):
        try:
            env.close()
        except Exception:
            pass

        state = env.reset()
        done = False
        total_reward = 0
        episode_loss_sum = 0
        loss_count = 0

        while not done:
            if not env.vip_override:
                if np.random.rand() < epsilon:
                    action = np.random.randint(ACTION_DIM)
                else:
                    with torch.no_grad():
                        q_values = model(torch.FloatTensor(state))
                        action = torch.argmax(q_values).item()
            else:
                action = None

            next_state, reward, done, info = env.step(action)

            if action is not None:
                buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if not env.vip_override and len(buffer) > BATCH_SIZE:
                samples = buffer.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*samples)

                states_tensor = torch.FloatTensor(np.array(states))
                actions_tensor = torch.LongTensor(actions).unsqueeze(1)
                rewards_tensor = torch.FloatTensor(rewards)
                next_states_tensor = torch.FloatTensor(np.array(next_states))
                dones_tensor = torch.FloatTensor(dones)

                q_vals = model(states_tensor).gather(1, actions_tensor).squeeze()

                with torch.no_grad():
                    next_q = target_model(next_states_tensor).max(1)[0]
                    expected_q = rewards_tensor + GAMMA * next_q * (1 - dones_tensor)

                loss = (q_vals - expected_q).pow(2).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                episode_loss_sum += loss.item()
                loss_count += 1

        stats = env.get_episode_stats()

        episode_rewards.append(total_reward)
        avg_reward = np.mean(episode_rewards[-10:])
        avg_rewards.append(avg_reward)
        epsilon_history.append(epsilon)
        vip_encounter_history.append(stats["vip_encounters"])
        phase_switch_history.append(stats["total_phase_switches"])

        avg_loss = episode_loss_sum / loss_count if loss_count > 0 else 0
        episode_losses.append(avg_loss)

        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            target_model.load_state_dict(model.state_dict())

        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

        if (ep + 1) % 10 == 0:
            torch.save(
                {
                    "episode": ep + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epsilon": epsilon,
                    "episode_rewards": episode_rewards,
                },
                f"models/dqn_model_ep{ep + 1}.pth"
            )

        line1.set_data(range(len(episode_rewards)), episode_rewards)
        line2.set_data(range(len(avg_rewards)), avg_rewards)
        line3.set_data(range(len(epsilon_history)), epsilon_history)

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()
        plt.pause(0.01)

finally:
    env.close()
    plt.ioff()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode_rewards": episode_rewards,
            "epsilon": epsilon,
        },
        "models/dqn_model_final.pth"
    )

    history = {
        "episode_rewards": episode_rewards,
        "avg_rewards": avg_rewards,
        "episode_losses": episode_losses,
        "epsilon_history": epsilon_history,
        "vip_encounters": vip_encounter_history,
        "phase_switches": phase_switch_history,
        "hyperparameters": {
            "episodes": EPISODES,
            "gamma": GAMMA,
            "epsilon_start": EPSILON_START,
            "epsilon_decay": EPSILON_DECAY,
            "epsilon_min": EPSILON_MIN,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        }
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/training_history_{timestamp}.json", "w") as f:
        json.dump(history, f, indent=2)

    plt.savefig(f"logs/training_plot_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.show()
