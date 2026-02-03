import numpy as np


def get_phase_map():
    return {
        0: ["E2C", "W2C"],
        2: ["N2C", "S2C"]
    }


def check_vip_presence(traci, edges, vip_type="vip_veh"):
    for edge in edges:
        try:
            vehicles = traci.edge.getLastStepVehicleIDs(edge)
            for veh_id in vehicles:
                if traci.vehicle.getTypeID(veh_id) == vip_type:
                    return True, edge
        except Exception:
            pass
    return False, None


def calculate_total_waiting_time(traci, edges):
    total_wait = 0
    for edge in edges:
        try:
            total_wait += traci.edge.getWaitingTime(edge)
        except Exception:
            pass
    return total_wait


def normalize_detector_counts(counts, max_vehicles=20):
    return np.clip(counts / max_vehicles, 0, 1).astype(np.float32)


def get_phase_stats(phase_counts):
    total = sum(phase_counts.values())
    if total == 0:
        return {"total": 0, "balance": 0}

    stats = {"total": total}
    for phase, count in phase_counts.items():
        stats[f"phase_{phase}_pct"] = (count / total) * 100

    counts_list = list(phase_counts.values())
    if len(counts_list) == 2:
        stats["balance"] = min(counts_list) / max(counts_list) if max(counts_list) > 0 else 0

    return stats
