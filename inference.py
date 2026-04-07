import argparse
import json
import os
import sys
import time
from typing import Any, Optional
from collections import deque, defaultdict
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from openai import OpenAI


DEFAULT_ENV_SERVER = os.environ.get("ENV_SERVER_URL", os.environ.get("LOCAL_SERVER_URL", "http://127.0.0.1:7860"))
DEFAULT_OUTPUT = "inference_results.json"
REQUEST_TIMEOUT = 20
MAX_MODEL_RETRIES = 3
ALLOWED_ACTIONS = {"place_relay", "move_drone", "boost_signal", "reroute", "activate_node", "deactivate_node"}


# ─────────────────────────────────────────────
# Advanced Adaptive Agent with Memory
# ─────────────────────────────────────────────

@dataclass
class RecentAction:
    action_type: str
    node_id: Optional[str]
    target_x: Optional[float]
    target_y: Optional[float]
    step: int
    reward: float


class MemoriedAgent:
    """
    Intelligent agent with memory, adaptation, and strategic decision-making.
    - Memory awareness: Tracks recent actions, avoids repetition
    - Adaptation: Changes strategy on negative rewards
    - Prioritization: Focuses on weak/inactive nodes
    - Strategic placement: Relays between base and weak nodes
    """
    
    def __init__(self, memory_window: int = 10):
        self.memory_window = memory_window
        self.action_history = deque(maxlen=memory_window)
        self.node_action_counts = defaultdict(lambda: defaultdict(int))
        self.last_reward = 0.0
        self.consecutive_negative_rewards = 0
        self.step_count = 0
        self.last_coverage = 0.0
        self.strategy_mode = "exploration"
    
    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Main decision function with memory and adaptation."""
        self.step_count += 1
        
        nodes = observation.get("nodes", [])
        coverage = observation.get("coverage_ratio", 0.0)
        relays_active = observation.get("active_relays", 0)
        connected_pairs = observation.get("connected_pairs", [])
        
        # ADAPTATION: Check for negative reward pattern
        if self.last_reward < 0:
            self.consecutive_negative_rewards += 1
            if self.consecutive_negative_rewards >= 2:
                self.strategy_mode = "exploration"
                self.consecutive_negative_rewards = 0
        else:
            self.consecutive_negative_rewards = 0
            if coverage > 0.8:
                self.strategy_mode = "exploitation"
            else:
                self.strategy_mode = "exploration"
        
        # IDENTIFY PROBLEM NODES
        inactive = [n for n in nodes if not n.get("is_active", False) and not self._is_base(n)]
        weak = [n for n in nodes if n.get("signal", 0) < 0.5 and n.get("is_active", False) and not self._is_base(n)]
        disconnected = self._find_disconnected(nodes, connected_pairs)
        
        # DECISION TREE with MEMORY
        
        # RULE 1: Inactive nodes are CRITICAL
        if inactive:
            action = self._handle_inactive(inactive)
            if action:
                return action
        
        # RULE 2: Low coverage + relay budget → place strategic relay
        if coverage < 0.3 and relays_active < 12 and self.strategy_mode == "exploration":
            action = self._place_strategic_relay(nodes)
            if action:
                return action
        
        # RULE 3: Weak nodes → boost or reroute
        if weak:
            action = self._handle_weak(weak)
            if action:
                return action
        
        # RULE 4: Disconnected nodes → reroute
        if disconnected:
            action = self._handle_disconnected(disconnected)
            if action:
                return action
        
        # RULE 5: Exploration mode → try diverse actions
        if self.strategy_mode == "exploration" and coverage < 0.7:
            action = self._explore_action(nodes)
            if action:
                return action
        
        # FALLBACK: Safe default
        return self._default_action(nodes, weak)
    
    def _handle_inactive(self, inactive: list[dict]) -> dict[str, Any]:
        """Handle inactive nodes - CRITICAL priority. Try different actions to avoid loops."""
        # Try reroute first (preferred for inactive)
        target = self._select_avoid_repetition(inactive, "reroute")
        if target:
            self._record_action("reroute", target["id"], None, None)
            return {"action_type": "reroute", "node_id": target["id"]}
        
        # If reroute was just used, try activate instead
        if inactive:
            target = inactive[0]
            self._record_action("activate_node", target["id"], None, None)
            return {"action_type": "activate_node", "node_id": target["id"]}
        
        return {}
    
    def _handle_weak(self, weak: list[dict]) -> dict[str, Any]:
        """Handle weak signal nodes. Try different actions to avoid repetition."""
        # Try boost_signal first (preferred for weak)
        target = self._select_avoid_repetition(weak, "boost_signal")
        if target:
            weakness = 1.0 - target.get("signal", 0.5)
            boost = min(0.6, weakness * 0.8)
            self._record_action("boost_signal", target["id"], None, None)
            return {
                "action_type": "boost_signal",
                "node_id": target["id"],
                "boost_amount": round(boost, 2)
            }
        
        # If boost_signal was just used, try reroute instead
        if weak:
            target = weak[0]
            self._record_action("reroute", target["id"], None, None)
            return {
                "action_type": "reroute",
                "node_id": target["id"]
            }
        
        return {}
    
    def _handle_disconnected(self, disconnected: list[dict]) -> dict[str, Any]:
        """Handle disconnected nodes. Try different actions to avoid repetition."""
        # Try reroute first
        target = self._select_avoid_repetition(disconnected, "reroute")
        if target:
            self._record_action("reroute", target["id"], None, None)
            return {"action_type": "reroute", "node_id": target["id"]}
        
        # If reroute was just used, try boost_signal instead  
        if disconnected:
            target = disconnected[0]
            self._record_action("boost_signal", target["id"], None, None)
            return {
                "action_type": "boost_signal",
                "node_id": target["id"],
                "boost_amount": 0.3
            }
        
        return {}
    
    def _place_strategic_relay(self, nodes: list[dict]) -> dict[str, Any]:
        """Place relay strategically between base and weak node."""
        base = self._find_base(nodes)
        survivors = [n for n in nodes if not self._is_base(n) and not n.get("is_relay", False)]
        
        if not base or not survivors:
            return {}
        
        target = min(survivors, key=lambda n: n.get("signal", 1.0))
        ratio = [0.3, 0.5, 0.7][self.step_count % 3]
        
        x = base["x"] + (target["x"] - base["x"]) * ratio
        y = base["y"] + (target["y"] - base["y"]) * ratio
        
        jitter_x = ((self.step_count * 3) % 8) - 4
        jitter_y = ((self.step_count * 7) % 8) - 4
        
        x = max(0, min(100, x + jitter_x))
        y = max(0, min(100, y + jitter_y))
        
        self._record_action("place_relay", None, x, y)
        return {
            "action_type": "place_relay",
            "target_x": round(x, 1),
            "target_y": round(y, 1)
        }
    
    def _explore_action(self, nodes: list[dict]) -> dict[str, Any]:
        """Exploration mode: try diverse actions on different nodes."""
        candidates = [n for n in nodes if not n.get("is_relay", False) and not self._is_base(n)]
        if not candidates:
            return {}
        
        target = min(candidates, key=lambda n: sum(self.node_action_counts[n["id"]].values()))
        
        if not target.get("is_active", False):
            self._record_action("activate_node", target["id"], None, None)
            return {"action_type": "activate_node", "node_id": target["id"]}
        elif target.get("signal", 0) < 0.4:
            self._record_action("boost_signal", target["id"], None, None)
            return {
                "action_type": "boost_signal",
                "node_id": target["id"],
                "boost_amount": 0.4
            }
        return {}
    
    def _default_action(self, nodes: list[dict], weak: list[dict]) -> dict[str, Any]:
        """Fallback action when no clear decision."""
        if weak:
            target = weak[0]
            self._record_action("boost_signal", target["id"], None, None)
            return {
                "action_type": "boost_signal",
                "node_id": target["id"],
                "boost_amount": 0.2
            }
        
        self._record_action("move_drone", None, 50.0, 50.0)
        return {
            "action_type": "move_drone",
            "target_x": 50.0,
            "target_y": 50.0
        }
    
    def _select_avoid_repetition(self, candidates: list[dict], action_type: str) -> Optional[dict]:
        """Select target that hasn't had this action recently."""
        if not candidates:
            return None
        
        # Check if LAST action was this action type on this node
        if self.action_history:
            last_action = self.action_history[-1]
            # If last action is same type and node is same, return None to force different action
            for candidate in candidates:
                if last_action.node_id == candidate["id"] and last_action.action_type == action_type:
                    return None  # Force fallback to different action
        
        best_target = None
        best_score = float("inf")
        
        for candidate in candidates:
            node_id = candidate["id"]
            recent_count = 0
            for action in list(self.action_history)[-3:]:
                if action.node_id == node_id and action.action_type == action_type:
                    recent_count += 1
            
            if recent_count < best_score:
                best_score = recent_count
                best_target = candidate
        
        return best_target
    
    def _record_action(self, action_type: str, node_id: Optional[str], target_x: Optional[float], target_y: Optional[float]):
        """Record action in memory."""
        action = RecentAction(
            action_type=action_type,
            node_id=node_id,
            target_x=target_x,
            target_y=target_y,
            step=self.step_count,
            reward=self.last_reward
        )
        self.action_history.append(action)
        if node_id:
            self.node_action_counts[node_id][action_type] += 1
    
    def update_reward(self, reward: float, coverage: float):
        """Update agent with feedback from environment."""
        self.last_reward = reward
        self.last_coverage = coverage
    
    def _find_base(self, nodes: list[dict]) -> Optional[dict]:
        """Find base station (signal source)."""
        for node in nodes:
            if self._is_base(node):
                return node
        return None
    
    def _is_base(self, node: dict) -> bool:
        """Check if node is a base station."""
        node_id = str(node.get("id", "")).lower()
        return any(node_id.startswith(prefix) for prefix in ("base", "surface", "rescue", "hq", "tunnel_entry"))
    
    def _find_disconnected(self, nodes: list[dict], connected_pairs: list) -> list[dict]:
        """Find nodes not in any connected pair."""
        connected_ids = set()
        for pair in connected_pairs:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                connected_ids.add(pair[0])
                connected_ids.add(pair[1])
        
        survivors = [n for n in nodes if not self._is_base(n) and not n.get("is_relay", False)]
        return [n for n in survivors if n.get("id") not in connected_ids]


SYSTEM_PROMPT = """You are an expert emergency communications engineer. 
You are controlling a mesh rescue agent to restore connectivity in a disaster zone.

Goal: Restore communication to all survivor nodes by strategic drone/relay placement and signal management.

Valid actions:
- place_relay: x, y (0-100) - Place a new static relay node.
- move_drone: x, y (0-100) - Move an existing relay node to a new location.
- boost_signal: node_id, boost_amount (0.0-1.0) - Temporarily increase a node's signal strength at a battery cost.
- reroute: node_id - Re-activate a degraded node and rebalance its signal.
- activate_node: node_id - Power on a node that is currently inactive.
- deactivate_node: node_id - Power off a node to save battery (rarely used).

Rules:
1. Grid is 100x100. (0,0) is bottom-left, (100,100) is top-right.
2. Relays have limited budget. Use them wisely near survivor clusters.
3. Obstacles (buildings, tunnel walls) block signals. Route around them.
4. Survivors have IDs like 'survivor_a', 'miner_b'. 
5. Base nodes like 'base', 'surface_base', or 'rescue_hq' are primary signal sources.

Always output EXACTLY one JSON object. No prose. No markdown fences.
Example: {"action_type": "place_relay", "target_x": 45.2, "target_y": 60.5}
"""


def env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def coerce_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(maximum, number))


def wait_for_server(base_url: str, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return
        except requests.RequestException:
            time.sleep(1)
    raise RuntimeError(f"Environment server did not become ready within {timeout_seconds} seconds: {base_url}")


def fetch_tasks(base_url: str, requested_tasks: list[str] | None = None) -> list[dict[str, Any]]:
    response = requests.get(f"{base_url}/tasks", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    task_payload = response.json().get("tasks", [])

    tasks = []
    requested = set(requested_tasks or [])
    for task in task_payload:
        task_id = task.get("task_id")
        if not task_id:
            continue
        if requested and task_id not in requested:
            continue
        tasks.append(task)

    if requested and len(tasks) != len(requested):
        found = {task["task_id"] for task in tasks}
        missing = sorted(requested - found)
        raise RuntimeError(f"Requested task ids were not found on the server: {', '.join(missing)}")

    if not tasks:
        raise RuntimeError("No tasks were returned by the environment server.")

    return tasks


def build_user_prompt(observation: dict[str, Any], step_index: int, last_reward: float) -> str:
    node_lines = []
    for node in observation.get("nodes", []):
        node_lines.append(
            f"- {node['id']}: pos=({node['x']:.1f},{node['y']:.1f}) "
            f"signal={node['signal']:.2f} battery={node['battery']:.2f} "
            f"active={node['is_active']} relay={node['is_relay']}"
        )

    obstacle_lines = []
    for obstacle in observation.get("obstacles", []):
        obstacle_lines.append(
            f"- obstacle at ({obstacle['x']},{obstacle['y']}) size=({obstacle['width']}x{obstacle['height']}) "
            f"attenuation={obstacle['attenuation']}"
        )

    lines = [
        f"Task: {observation.get('task_id', 'unknown')}",
        f"Step: {step_index}/{observation.get('max_steps', '?')}",
        f"Coverage: {observation.get('coverage_ratio', 0.0):.2%}",
        f"Active relays: {observation.get('active_relays', 0)}",
        f"Last reward: {last_reward:.4f}",
        "",
        "Nodes:",
        *node_lines,
    ]

    if obstacle_lines:
        lines.extend(["", "Obstacles:", *obstacle_lines])

    connected_pairs = observation.get("connected_pairs", [])
    if connected_pairs:
        lines.extend(["", f"Connected pairs: {connected_pairs}"])

    lines.extend(
        [
            "",
            "Return the single best next action as JSON only.",
        ]
    )

    return "\n".join(lines)


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        parts = candidate.split("```")
        candidate = next((part for part in parts if "{" in part and "}" in part), candidate).strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(candidate[start:end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model response did not contain a valid JSON object.")


def is_base_node(node: dict[str, Any]) -> bool:
    node_id = str(node.get("id", "")).lower()
    return any(
        node_id.startswith(prefix)
        for prefix in ("base", "surface", "tunnel_entry", "rescue_hq", "hq")
    )


def fallback_action(observation: dict[str, Any]) -> dict[str, Any]:
    nodes = observation.get("nodes", [])
    base_node = next((node for node in nodes if is_base_node(node)), None)
    survivors = [
        node for node in nodes
        if not is_base_node(node) and not node.get("is_relay", False)
    ]
    inactive_nodes = [node for node in survivors if not node.get("is_active", False)]
    weak_nodes = [node for node in survivors if node.get("is_active", False) and node.get("signal", 0.0) < 0.6]

    if inactive_nodes:
        return {"action_type": "reroute", "node_id": inactive_nodes[0]["id"]}

    if weak_nodes:
        weakest = min(weak_nodes, key=lambda node: node.get("signal", 0.0))
        return {"action_type": "boost_signal", "node_id": weakest["id"], "boost_amount": 0.3}

    if base_node and survivors:
        target = min(survivors, key=lambda node: node.get("signal", 1.0))
        return {
            "action_type": "place_relay",
            "target_x": round((base_node["x"] + target["x"]) / 2, 1),
            "target_y": round((base_node["y"] + target["y"]) / 2, 1),
        }

    return {"action_type": "move_drone", "target_x": 50.0, "target_y": 50.0}


def sanitize_action(action: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    nodes = observation.get("nodes", [])
    valid_node_ids = {str(node.get("id")) for node in nodes}

    action_type = action.get("action_type")
    if action_type not in ALLOWED_ACTIONS:
        print(f"  WARNING: Invalid action type '{action_type}'. Using fallback.")
        return fallback_action(observation)

    cleaned = {
        "action_type": action_type,
        "target_x": coerce_float(action.get("target_x"), 50.0, 0.0, 100.0),
        "target_y": coerce_float(action.get("target_y"), 50.0, 0.0, 100.0),
        "node_id": action.get("node_id"),
        "boost_amount": coerce_float(action.get("boost_amount"), 0.2, 0.0, 1.0),
    }

    if action_type in {"boost_signal", "reroute", "activate_node", "deactivate_node"}:
        node_id = str(cleaned.get("node_id")) if cleaned.get("node_id") is not None else None
        if not node_id or node_id not in valid_node_ids:
            return fallback_action(observation)
        cleaned["node_id"] = node_id

    return cleaned


def request_model_action(
    client: OpenAI | None,
    model_name: str,
    observation: dict[str, Any],
    step_index: int,
    last_reward: float,
    history: list[dict[str, str]] | None = None,
    agent: MemoriedAgent | None = None,
) -> dict[str, Any]:
    """
    Request next action from model or adaptive agent.
    
    If client is None, use MemoriedAgent instead of LLM.
    Otherwise, try LLM first, fallback to agent if it fails.
    """
    
    # If agent is provided, update it with reward
    if agent:
        agent.update_reward(last_reward, observation.get("coverage_ratio", 0.0))
    
    if client is None:
        # No LLM: use adaptive agent
        if agent is None:
            agent = MemoriedAgent()
        return sanitize_action(agent.decide(observation), observation)

    # Try LLM first
    user_prompt = build_user_prompt(observation, step_index, last_reward)
    last_error = None

    messages = [{
    "role": "system",
    "content": SYSTEM_PROMPT + "\n\nAvoid repeating recent actions. Be adaptive."
}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(1, MAX_MODEL_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=250,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = extract_json_object(content)

            if history is not None:
                history.append({"role": "user", "content": user_prompt})
                history.append({"role": "assistant", "content": content})
                if len(history) > 12:
                    history.pop(0)
                    history.pop(0)

            return sanitize_action(parsed, observation)
        except Exception as exc:
            last_error = exc
            print(f"  AI attempt {attempt} failed: {exc}")
            time.sleep(1)

    print(f"Model action generation failed after {MAX_MODEL_RETRIES} attempts: {last_error}")
    
    # Fallback to adaptive agent
    if agent is None:
        agent = MemoriedAgent()
    return sanitize_action(agent.decide(observation), observation)


def reset_episode(base_url: str, task_id: str, session_id: str, seed: int) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/reset",
        json={"task_id": task_id, "session_id": session_id, "seed": seed},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("observation") or payload.get("state") or {}


def step_episode(base_url: str, action: dict[str, Any], session_id: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/step",
        json={**action, "session_id": session_id},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def grade_episode(base_url: str, session_id: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url}/grader",
        json={"session_id": session_id},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def run_task(
    env_server_url: str,
    llm_client: OpenAI | None,
    model_name: str,
    task: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    task_id = task["task_id"]
    max_steps = int(task.get("max_steps", 30))
    session_id = f"inference-{task_id}-{seed}"

    print(f"\nRunning task: {task_id}")
    observation = reset_episode(env_server_url, task_id, session_id, seed)

    total_reward = 0.0
    last_reward = 0.0
    done = False
    last_info: dict[str, Any] = {}
    executed_steps = 0

    history: list[dict[str, str]] = []
    agent = MemoriedAgent(memory_window=10)  # Create adaptive agent
    
    for step_index in range(1, max_steps + 1):
        action = request_model_action(llm_client, model_name, observation, step_index, last_reward, history, agent)

        action_target = None
        if action.get("node_id"):
            action_target = action["node_id"]
        elif action.get("target_x") is not None and action.get("target_y") is not None:
            action_target = f"({action['target_x']:.1f},{action['target_y']:.1f})"
        else:
            action_target = "unknown target"

        print(f"\n🚀 Step {step_index:02d}")
        print(f"👉 Action: {action['action_type']} on {action_target}")
        if "boost_amount" in action and action["boost_amount"] is not None:
            print(f"⚡ Boost: {action['boost_amount']}")

        step_payload = step_episode(env_server_url, action, session_id)
        observation = step_payload.get("observation") or step_payload.get("state") or {}
        last_reward = float(step_payload.get("reward", 0.0))
        done = bool(step_payload.get("done", False))
        last_info = step_payload.get("info", {})
        total_reward += last_reward
        executed_steps = step_index

        coverage_value = last_info.get("coverage")
        if coverage_value is None:
            coverage_value = observation.get("coverage_ratio", 0.0)

        print(f"🏆 Reward: {last_reward:.3f}")
        print(f"📡 Coverage: {coverage_value * 100:.1f}%")
        print(f"✅ Done: {done}")

        if done:
            break

    grade = grade_episode(env_server_url, session_id)
    result = {
        "task_id": task_id,
        "score": grade.get("score"),
        "breakdown": grade.get("breakdown", {}),
        "steps_executed": executed_steps,
        "max_steps": max_steps,
        "total_reward": round(total_reward, 4),
        "final_coverage": observation.get("coverage_ratio"),
        "outcome": last_info.get("outcome"),
    }

    print(f"\n🎯 Task Completed: {task_id}")
    print(f"⭐ Final Score: {result['score']:.3f}")
    print("=" * 40)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM inference against the RescueMesh environment.")
    parser.add_argument("--env-server", default=DEFAULT_ENV_SERVER, help="Environment server URL.")
    parser.add_argument("--api-base-url", default=env_first("API_BASE_URL", "OPENAI_BASE_URL"), help="LLM API base URL.")
    parser.add_argument("--model", default=env_first("MODEL_NAME", "OPENAI_MODEL", default="gpt-4o-mini"), help="Model name to call.")
    parser.add_argument(
        "--api-key",
        default=env_first("OPENAI_API_KEY", "HF_TOKEN", "API_KEY"),
        help="API key for the model provider.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic environment seed.")
    parser.add_argument("--tasks", nargs="*", help="Optional subset of task ids to run.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write the result summary JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.model:
        print("ERROR: Missing model name. Set MODEL_NAME or pass --model.", file=sys.stderr)
        return 1

    llm_client = None
    if args.api_key:
        client_kwargs: dict[str, Any] = {"api_key": args.api_key}
        if args.api_base_url:
            client_kwargs["base_url"] = args.api_base_url
        llm_client = OpenAI(**client_kwargs)
        print(f"Model: {args.model}")
    else:
        print("NOTICE: No API key found. Running with ADAPTIVE AGENT (no LLM).")
        print("The MemoriedAgent will handle all decisions with memory and adaptation.")
        print("To use an LLM, create a .env file with OPENAI_API_KEY=sk-...")

    try:
        wait_for_server(args.env_server)
        tasks = fetch_tasks(args.env_server, args.tasks)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Environment server: {args.env_server}")
    print(f"Model: {args.model}")
    if args.api_base_url:
        print(f"Model API base URL: {args.api_base_url}")

    all_results = []
    for task in tasks:
        try:
            all_results.append(run_task(args.env_server, llm_client, args.model, task, args.seed))
        except Exception as exc:  # noqa: BLE001
            print(f"Task failed: {task.get('task_id')}: {exc}", file=sys.stderr)
            all_results.append(
                {
                    "task_id": task.get("task_id"),
                    "error": str(exc),
                }
            )

    scored_results = [result for result in all_results if isinstance(result.get("score"), (int, float))]
    average_score = round(sum(result["score"] for result in scored_results) / len(scored_results), 4) if scored_results else None

    output_payload = {
        "agent": "llm_inference_with_adaptive_fallback",
        "model": args.model,
        "env_server": args.env_server,
        "seed": args.seed,
        "results": all_results,
        "average_score": average_score,
    }

    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(output_payload, output_file, indent=2)

    print(f"\nSaved inference summary to {args.output}")
    if average_score is not None:
        print(f"Average score: {average_score}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
