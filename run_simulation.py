import torch
import numpy as np
from model.dqn_model import DQN
from training.env import TrafficEnv


STATE_DIM = 16
ACTION_DIM = 2
MODEL_PATH = "models/dqn_model_final.pth"


model = DQN(STATE_DIM, ACTION_DIM)

checkpoint = torch.load(MODEL_PATH)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()


env = TrafficEnv()
state = env.reset()
done = False

total_reward = 0
step_count = 0
vip_actions = 0
agent_actions = 0
action_counts = {0: 0, 1: 0}

try:
    while not done:
        with torch.no_grad():
            if not env.vip_override:
                state_tensor = torch.FloatTensor(state)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
                action = np.clip(action, 0, ACTION_DIM - 1)
                agent_actions += 1
                action_counts[action] += 1
            else:
                action = None
                vip_actions += 1

        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        step_count += 1

except KeyboardInterrupt:
    pass
except Exception:
    pass
finally:
    env.close()


stats = env.get_episode_stats()

avg_reward = total_reward / step_count if step_count > 0 else 0
phase_balance = stats["phase_stats"].get("balance", None)
