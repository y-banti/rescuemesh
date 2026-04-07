"""
RescueMesh AI Environment
Simulates disaster and coal mine emergency communication restoration.
Compliant with OpenEnv spec: step(), reset(), state()
"""

from __future__ import annotations
import math
import random
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
from enum import Enum


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ActionType(str, Enum):
    PLACE_RELAY     = "place_relay"
    MOVE_DRONE      = "move_drone"
    BOOST_SIGNAL    = "boost_signal"
    REROUTE         = "reroute"
    ACTIVATE_NODE   = "activate_node"
    DEACTIVATE_NODE = "deactivate_node"


class TaskID(str, Enum):
    EASY   = "easy_open_field"
    MEDIUM = "medium_obstacle"
    HARD   = "hard_coal_mine"


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class Node:
    id: str
    x: float          # 0–100 grid
    y: float
    signal: float     # 0.0–1.0
    battery: float    # 0.0–1.0
    is_relay: bool = False
    is_active: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Obstacle:
    x: float
    y: float
    width: float
    height: float
    attenuation: float  # signal reduction factor (0.1 = blocks 90%)

    def blocks(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Check if this obstacle blocks a line between two points."""
        # Simple AABB line-segment intersection
        ox2, oy2 = self.x + self.width, self.y + self.height
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return False
        tmin, tmax = 0.0, 1.0
        for ax, bx, amin, amax in [
            (dx, x1 - self.x, 0, self.width),
            (dy, y1 - self.y, 0, self.height),
        ]:
            if ax == 0:
                if bx < -amax or bx > 0:
                    return False
            else:
                t1 = (-bx) / ax
                t2 = (amax - bx) / ax
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return False
        return True


@dataclass
class Observation:
    nodes: list[dict]
    obstacles: list[dict]
    connected_pairs: list[list[str]]
    coverage_ratio: float        # 0.0–1.0
    active_relays: int
    step_count: int
    task_id: str
    max_steps: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Action:
    action_type: ActionType
    target_x: float = 50.0      # Grid position for place/move
    target_y: float = 50.0
    node_id: Optional[str] = None   # For boost/reroute
    boost_amount: float = 0.1

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type.value,
            "target_x": self.target_x,
            "target_y": self.target_y,
            "node_id": self.node_id,
            "boost_amount": self.boost_amount,
        }

    @staticmethod
    def from_dict(d: dict) -> "Action":
        return Action(
            action_type=ActionType(d["action_type"]),
            target_x=float(d.get("target_x", 50.0)),
            target_y=float(d.get("target_y", 50.0)),
            node_id=d.get("node_id"),
            boost_amount=float(d.get("boost_amount", 0.2)),
        )


# ─────────────────────────────────────────────
# Task Configurations
# ─────────────────────────────────────────────

TASK_CONFIGS = {
    TaskID.EASY: {
        "description": "Open field scenario. Restore communication between 4 survivor nodes using relay drones. No obstacles.",
        "max_steps": 8,
        "max_relays": 5,
        "signal_decay": 0.02,
        "battery_decay": 0.015,
        "obstacles": [],
        "signal_range": 35.0,
        "target_coverage": 1.0,
        "difficulty": "easy",
    },
    TaskID.MEDIUM: {
        "description": "Urban disaster zone. Collapsed buildings block direct links. Deploy relays to route around obstacles.",
        "max_steps": 12,
        "max_relays": 6,
        "signal_decay": 0.03,
        "battery_decay": 0.02,
        "obstacles": [
            {"x": 30, "y": 20, "width": 15, "height": 40, "attenuation": 0.15},
            {"x": 55, "y": 40, "width": 20, "height": 25, "attenuation": 0.1},
            {"x": 10, "y": 60, "width": 25, "height": 20, "attenuation": 0.12},
        ],
        "signal_range": 30.0,
        "target_coverage": 0.85,
        "difficulty": "medium",
    },
    TaskID.HARD: {
        "description": "Underground coal mine. Severe signal attenuation, tunnel constraints, low battery, strict relay budget.",
        "max_steps": 16,
        "max_relays": 5,
        "signal_decay": 0.05,
        "battery_decay": 0.035,
        "obstacles": [
            {"x": 20, "y": 0,  "width": 10, "height": 35, "attenuation": 0.05},
            {"x": 40, "y": 30, "width": 8,  "height": 50, "attenuation": 0.04},
            {"x": 65, "y": 10, "width": 10, "height": 40, "attenuation": 0.05},
            {"x": 15, "y": 65, "width": 55, "height": 8,  "attenuation": 0.06},
        ],
        "signal_range": 20.0,
        "target_coverage": 0.75,
        "difficulty": "hard",
    },
}


# ─────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────

class RescueMeshEnv:
    """
    OpenEnv-compliant environment for emergency mesh network restoration.
    """

    def __init__(self, task_id: str = "easy_open_field", seed: int = 42):
        self.task_id = TaskID(task_id)
        self.seed = seed
        self._rng = random.Random(seed)
        self._config = TASK_CONFIGS[self.task_id]

        # Episode state
        self._nodes: list[Node] = []
        self._obstacles: list[Obstacle] = []
        self._relays_placed: int = 0
        self._step_count: int = 0
        self._done: bool = False
        self._prev_coverage: float = 0.0

        self.reset()

    # ── OpenEnv required methods ──────────────

    def reset(self) -> dict:
        """Reset the environment and return initial observation."""
        self._rng = random.Random(self.seed)
        self._step_count = 0
        self._done = False
        self._relays_placed = 0
        self._prev_coverage = 0.0

        self._nodes = self._spawn_nodes()
        self._obstacles = [
            Obstacle(**o) for o in self._config["obstacles"]
        ]

        self._propagate_signals()
        obs = self._build_observation()
        self._prev_coverage = obs.coverage_ratio
        return obs.to_dict()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Execute action, return (observation, reward, done, info).

        action: dict with keys:
            action_type: str  (place_relay | move_drone | boost_signal | reroute)
            target_x: float   (0–100)
            target_y: float   (0–100)
            node_id: str      (optional, for boost/reroute)
            boost_amount: float (optional)
        """
        # Removed the self._done lockout block to allow continuous control in dashboard

        act = Action.from_dict(action)
        info: dict[str, Any] = {"action": action, "step": self._step_count}

        # Apply action
        reward = self._apply_action(act, info)

        # Apply environment decay
        self._apply_decay()

        # Re-propagate signals
        self._propagate_signals()

        self._step_count += 1
        obs = self._build_observation()

        # Shaped reward: improvement in coverage
        coverage_delta = obs.coverage_ratio - self._prev_coverage
        if coverage_delta > 0.05:
            reward += 0.3 * coverage_delta
        elif coverage_delta < -0.05:
            reward -= 0.1  # degraded

        self._prev_coverage = obs.coverage_ratio

        # Termination conditions
        target = self._config["target_coverage"]
        if obs.coverage_ratio >= target:
            reward += 1.0  # full success bonus
            self._done = True
            info["outcome"] = "success"
        elif self._step_count >= self._config["max_steps"]:
            self._done = True
            info["outcome"] = "timeout"
        else:
            info["outcome"] = "ongoing"

        info["coverage"] = obs.coverage_ratio
        info["relays_placed"] = self._relays_placed

        return obs.to_dict(), round(reward, 4), self._done, info

    def state(self) -> dict:
        """Return full internal state (for debugging / checkpointing)."""
        return {
            "task_id": self.task_id.value,
            "step_count": self._step_count,
            "done": self._done,
            "relays_placed": self._relays_placed,
            "max_relays": self._config["max_relays"],
            "nodes": [n.to_dict() for n in self._nodes],
            "obstacles": [asdict(o) for o in self._obstacles],
            "config": {k: v for k, v in self._config.items() if k != "obstacles"},
        }

    # ── Action handlers ───────────────────────

    def _apply_action(self, act: Action, info: dict) -> float:
        reward = 0.0

        if act.action_type == ActionType.PLACE_RELAY:
            if self._relays_placed >= self._config["max_relays"]:
                info["warning"] = "relay_budget_exhausted"
                reward -= 0.05
            else:
                relay = Node(
                    id=f"relay_{self._relays_placed}",
                    x=max(0, min(100, act.target_x)),
                    y=max(0, min(100, act.target_y)),
                    signal=0.9,
                    battery=1.0,
                    is_relay=True,
                )
                self._nodes.append(relay)
                self._relays_placed += 1
                reward += 0.1  # placement reward; coverage delta adds more

        elif act.action_type == ActionType.MOVE_DRONE:
            relay_nodes = [n for n in self._nodes if n.is_relay and n.is_active]
            if not relay_nodes:
                info["warning"] = "no_relay_to_move"
                reward -= 0.05
            else:
                # Move closest relay toward target
                target = (act.target_x, act.target_y)
                closest = min(relay_nodes, key=lambda n: _dist(n, target))
                old_x, old_y = closest.x, closest.y
                closest.x = max(0, min(100, act.target_x))
                closest.y = max(0, min(100, act.target_y))
                closest.battery -= 0.05  # movement cost
                if closest.battery <= 0:
                    closest.is_active = False
                    info["warning"] = "relay_battery_dead"
                    reward -= 0.1
                else:
                    reward += 0.05

        elif act.action_type == ActionType.BOOST_SIGNAL:
            target_node = self._get_node(act.node_id)
            if target_node is None or not target_node.is_active:
                info["warning"] = "invalid_boost_target"
                reward -= 0.05
            else:
                boost = min(act.boost_amount, 1.0 - target_node.signal)
                target_node.signal = min(1.0, target_node.signal + boost)
                target_node.battery -= boost * 0.3  # battery cost
                if target_node.battery <= 0:
                    target_node.is_active = False
                    reward -= 0.1
                else:
                    reward += 0.2 * boost

        elif act.action_type == ActionType.REROUTE:
            # Rerouting re-activates a degraded node and rebalances signal
            target_node = self._get_node(act.node_id)
            if target_node is None:
                info["warning"] = "invalid_reroute_target"
                reward -= 0.05
            elif not target_node.is_active:
                target_node.is_active = True
                target_node.signal = 0.6
                target_node.battery = max(target_node.battery, 0.5)
                reward += 0.15
            else:
                # Already active — small efficiency penalty
                reward -= 0.05

        elif act.action_type == ActionType.ACTIVATE_NODE:
            target_node = self._get_node(act.node_id)
            if target_node:
                target_node.is_active = True
                target_node.signal = max(target_node.signal, 0.5)
                target_node.battery = max(target_node.battery, 0.6)
                reward += 0.1
            else:
                reward -= 0.05

        elif act.action_type == ActionType.DEACTIVATE_NODE:
            target_node = self._get_node(act.node_id)
            if target_node:
                target_node.is_active = False
                target_node.signal = 0.0
                reward -= 0.1
            else:
                reward -= 0.05

        return reward

    # ── Simulation helpers ────────────────────

    def _spawn_nodes(self) -> list[Node]:
        """Spawn survivor/base nodes deterministically per task."""
        configs = {
            TaskID.EASY: [
                ("base",     5,  50, 1.0, 1.0),
                ("survivor_a", 95, 20, 0.3, 0.8),
                ("survivor_b", 95, 50, 0.3, 0.7),
                ("survivor_c", 95, 80, 0.3, 0.9),
            ],
            TaskID.MEDIUM: [
                ("base",      5,  50, 1.0, 1.0),
                ("rescue_hq", 50, 5,  0.8, 1.0),
                ("survivor_a", 80, 20, 0.2, 0.6),
                ("survivor_b", 80, 80, 0.2, 0.5),
                ("survivor_c", 20, 85, 0.3, 0.7),
            ],
            TaskID.HARD: [
                ("surface_base", 5,  50, 1.0, 1.0),
                ("tunnel_entry", 25, 50, 0.6, 0.9),
                ("miner_a",      60, 30, 0.1, 0.5),
                ("miner_b",      60, 70, 0.1, 0.4),
                ("miner_c",      85, 50, 0.05, 0.3),
            ],
        }
        return [
            Node(id=cfg[0], x=cfg[1], y=cfg[2], signal=cfg[3], battery=cfg[4])
            for cfg in configs[self.task_id]
        ]

    def _propagate_signals(self):
        """Update signal values based on proximity and obstacles."""
        signal_range = self._config["signal_range"]

        for node in self._nodes:
            if not node.is_active or node.id.startswith("base") or node.id.startswith("surface"):
                continue

            best_signal = 0.0
            for other in self._nodes:
                if other.id == node.id or not other.is_active:
                    continue
                d = _dist_nodes(node, other)
                if d > signal_range:
                    continue
                # Base signal from distance
                raw = other.signal * (1.0 - d / signal_range)
                # Obstacle attenuation
                for obs in self._obstacles:
                    if obs.blocks(node.x, node.y, other.x, other.y):
                        raw *= obs.attenuation
                best_signal = max(best_signal, raw)

            # Blend: don't erase a node's own signal entirely
            node.signal = round(max(node.signal * 0.6, best_signal * 0.4 + node.signal * 0.6), 4)
            node.signal = min(node.signal, 1.0)

    def _apply_decay(self):
        """Each step, signal and battery degrade slightly."""
        sd = self._config["signal_decay"]
        bd = self._config["battery_decay"]
        for node in self._nodes:
            if not node.is_active:
                continue
            if not (node.id.startswith("base") or node.id.startswith("surface")):
                node.signal = max(0.0, round(node.signal - sd, 4))
                node.battery = max(0.0, round(node.battery - bd, 4))
                if node.battery <= 0:
                    node.is_active = False
                    node.signal = 0.0

    def _build_observation(self) -> Observation:
        connected = self._compute_connected_pairs()
        non_base = [n for n in self._nodes if not (n.id.startswith("base") or n.id.startswith("surface"))]
        active_non_base = [n for n in non_base if n.is_active and n.signal > 0.1]
        coverage = round(len(active_non_base) / max(1, len(non_base)), 4)
        return Observation(
            nodes=[n.to_dict() for n in self._nodes],
            obstacles=[asdict(o) for o in self._obstacles],
            connected_pairs=connected,
            coverage_ratio=coverage,
            active_relays=self._relays_placed,
            step_count=self._step_count,
            task_id=self.task_id.value,
            max_steps=self._config["max_steps"],
        )

    def _compute_connected_pairs(self) -> list[list[str]]:
        """Return pairs of nodes with viable signal links."""
        pairs = []
        signal_range = self._config["signal_range"]
        active = [n for n in self._nodes if n.is_active and n.signal > 0.1]
        for i, a in enumerate(active):
            for b in active[i+1:]:
                if _dist_nodes(a, b) <= signal_range:
                    blocked = any(
                        obs.blocks(a.x, a.y, b.x, b.y) and obs.attenuation < 0.2
                        for obs in self._obstacles
                    )
                    if not blocked:
                        pairs.append([a.id, b.id])
        return pairs

    def _get_node(self, node_id: Optional[str]) -> Optional[Node]:
        if node_id is None:
            return None
        for n in self._nodes:
            if n.id == node_id:
                return n
        return None


# ─────────────────────────────────────────────
# Grader
# ─────────────────────────────────────────────

class RescueMeshGrader:
    """
    Scores a completed episode 0.0–1.0.
    Score is never constant — it depends on coverage, efficiency, battery state.
    """

    def grade(self, episode_history: list[dict]) -> dict:
        if not episode_history:
            return {"score": 0.0, "breakdown": {}}

        final_state = episode_history[-1]
        task_id = TaskID(final_state.get("task_id", "easy_open_field"))
        config = TASK_CONFIGS[task_id]
        target = config["target_coverage"]
        max_steps = config["max_steps"]
        max_relays = config["max_relays"]

        # Coverage score (0–0.5)
        coverage = final_state.get("coverage_ratio", 0.0)
        coverage_score = min(1.0, coverage / target) * 0.5

        # Efficiency score: fewer steps = higher (0–0.25)
        steps = final_state.get("step_count", max_steps)
        efficiency = max(0.0, 1.0 - steps / max_steps)
        efficiency_score = efficiency * 0.25

        # Relay economy: unused relay budget = higher (0–0.15)
        relays_used = final_state.get("relays_placed", max_relays)
        relay_economy = max(0.0, 1.0 - relays_used / max_relays)
        relay_score = relay_economy * 0.15

        # Battery preservation (0–0.10)
        nodes = final_state.get("nodes", [])
        if nodes:
            survivor_batteries = [
                n["battery"] for n in nodes
                if n.get("is_active") and not any(
                    n["id"].startswith(p) for p in ("base", "surface", "relay")
                )
            ]
            avg_battery = sum(survivor_batteries) / len(survivor_batteries) if survivor_batteries else 0.0
        else:
            avg_battery = 0.0
        battery_score = avg_battery * 0.10

        total = round(coverage_score + efficiency_score + relay_score + battery_score, 4)

        return {
            "score": total,
            "breakdown": {
                "coverage_score": round(coverage_score, 4),
                "efficiency_score": round(efficiency_score, 4),
                "relay_economy_score": round(relay_score, 4),
                "battery_preservation_score": round(battery_score, 4),
                "coverage": coverage,
                "target_coverage": target,
                "steps_used": steps,
                "relays_used": relays_used,
            },
        }


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _dist(node: Node, point: tuple[float, float]) -> float:
    return math.sqrt((node.x - point[0]) ** 2 + (node.y - point[1]) ** 2)

def _dist_nodes(a: Node, b: Node) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
