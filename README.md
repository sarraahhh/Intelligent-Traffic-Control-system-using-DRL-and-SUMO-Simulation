# 🚦 Intelligent Traffic Signal Control using Deep Reinforcement Learning

This project implements a **Deep Q-Network (DQN)**–based traffic signal controller using **SUMO** and **TraCI**.  
The agent learns to dynamically switch traffic signal phases to minimize vehicle waiting time while enforcing **VIP vehicle prioritization**.

---

##  Key Features
- Deep Q-Network (DQN) with target network
- SUMO-based traffic simulation
- Real-time control using TraCI
- VIP vehicle detection and priority override
- Lane-area detectors for state representation
- Reward based on total waiting time minimization
- Deterministic evaluation of trained policy

---

##  State & Action Space
- **State**: Normalized vehicle counts from 16 lane-area detectors  
- **Actions**:
  - `0` → East–West green
  - `1` → North–South green

---

##  Results
- Stable learning with decreasing average waiting time
- Balanced phase usage (~95%)
- Consistent deterministic behavior during evaluation
- VIP priority enforced without destabilizing learning

---

## ▶️ How to Run

### Train the agent
```bash
python train.py
```
### Run the simulation 
```bash 
python simulation.py
```