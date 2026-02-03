import traci
import numpy as np
import os
from pathlib import Path
from utils.helpers import (
    get_phase_map,
    check_vip_presence,
    calculate_total_waiting_time,
    normalize_detector_counts
)
from typing import Optional


class TrafficEnv:
    def __init__(self):
        self.signal_id = "center"
        self.action_to_green_phase = {0: 0, 1: 2}
        self.phase_map = get_phase_map()
        self.incoming_edges = ["E2C", "W2C", "N2C", "S2C"]
        self.outgoing_edges = ["C2E", "C2W", "C2N", "C2S", "C2C"]

        self.detectors = [
            *(f"det_E2C_{i}" for i in range(4)),
            *(f"det_W2C_{i}" for i in range(4)),
            *(f"det_N2C_{i}" for i in range(4)),
            *(f"det_S2C_{i}" for i in range(4)),
        ]

        self.vip_type = "vip_veh"
        self.vip_check_edges = ["W2C"]

        self.min_green_time = 15
        self.yellow_time = 3

        self.current_phase = 0
        self.phase_time = 0
        self.vip_override = False
        self.step_count = 0
        self.max_steps = 700

        self.phase_counts = {0: 0, 2: 0}
        self.vip_encounters = 0
        self.total_phase_switches = 0

        self.sumo_config = self._find_sumo_config()

    def _find_sumo_config(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent

        possible_paths = [
            project_root / "simulation" / "project.sumocfg",
            Path.cwd() / "simulation" / "project.sumocfg",
            Path("simulation") / "project.sumocfg",
        ]

        for config_path in possible_paths:
            if config_path.exists():
                print(f"✅ Found SUMO config: {config_path}")
                return str(config_path)

        print("❌ ERROR: Could not find project.sumocfg!")
        for path in possible_paths:
            print(f"   - {path}")

        default_path = project_root / "simulation" / "project.sumocfg"
        print(f"⚠️  Using default path: {default_path}")
        return str(default_path)

    def reset(self):
        if not os.path.exists(self.sumo_config):
            raise FileNotFoundError(f"SUMO config not found: {self.sumo_config}")

        traci.start([
            "sumo-gui",
            "-c", self.sumo_config,
            "--max-num-teleports", "10",
            "--max-depart-delay", "1000",
            "--no-warnings", "true",
            "--time-to-teleport", "300",
            "--start", "true",
            "--quit-on-end", "false"
        ])

        self.current_phase = 0
        self.phase_time = 0
        self.vip_override = False
        self.step_count = 0
        self.phase_counts = {0: 0, 2: 0}
        self.vip_encounters = 0
        self.total_phase_switches = 0

        traci.trafficlight.setPhase(self.signal_id, 0)
        return self._get_state()

    def close(self):
        try:
            traci.close()
        except Exception:
            pass

    def _get_state(self):
        try:
            counts = np.array(
                [traci.lanearea.getLastStepVehicleNumber(det) for det in self.detectors],
                dtype=np.float32
            )
            return normalize_detector_counts(counts, max_vehicles=20)
        except Exception:
            return np.zeros(len(self.detectors), dtype=np.float32)

    def _yellow_between(self, from_green):
        return 1 if from_green == 0 else 3

    def _switch_to_phase(self, target_green):
        if self.current_phase == target_green:
            return

        yellow = self._yellow_between(self.current_phase)
        traci.trafficlight.setPhase(self.signal_id, yellow)

        for _ in range(self.yellow_time):
            traci.simulationStep()

        traci.trafficlight.setPhase(self.signal_id, target_green)
        self.current_phase = target_green
        self.phase_time = 0
        self.phase_counts[target_green] += 1
        self.total_phase_switches += 1

    def _detect_vip_and_get_required_phase(self):
        vip_present, _ = check_vip_presence(
            traci,
            self.vip_check_edges,
            self.vip_type
        )
        if vip_present:
            return True, 0
        return False, None

    def step(self, action: Optional[int]):
        self.step_count += 1
        done = self.step_count >= self.max_steps

        vip_present, vip_phase = self._detect_vip_and_get_required_phase()

        if vip_present and vip_phase is not None:
            if not self.vip_override:
                self.vip_encounters += 1
            if vip_phase != self.current_phase:
                self._switch_to_phase(vip_phase)
            self.vip_override = True
            self.phase_time = 0
        else:
            self.vip_override = False

        if not self.vip_override and action is not None:
            target_green = self.action_to_green_phase[action]
            if self.phase_time >= self.min_green_time:
                if target_green != self.current_phase:
                    self._switch_to_phase(target_green)

        try:
            traci.simulationStep()
            self.phase_time += 1
        except Exception:
            done = True

        next_state = self._get_state()
        reward = -calculate_total_waiting_time(traci, self.incoming_edges) / 100

        info = {
            "vip_override": self.vip_override,
            "current_phase": self.current_phase,
            "phase_time": self.phase_time,
            "vip_encounters": self.vip_encounters,
            "phase_switches": self.total_phase_switches
        }

        return next_state, reward, done, info

    def get_episode_stats(self):
        from utils.helpers import get_phase_stats
        phase_stats = get_phase_stats(self.phase_counts)
        return {
            "total_steps": self.step_count,
            "vip_encounters": self.vip_encounters,
            "total_phase_switches": self.total_phase_switches,
            "phase_usage": self.phase_counts,
            "phase_stats": phase_stats
        }
