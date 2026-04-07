"""
RescueMesh AI Environment — FastAPI Server
Required endpoints: /tasks, /grader, /baseline, /step, /reset, /state
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Optional
import uvicorn
import os

from environment import RescueMeshEnv, RescueMeshGrader, TASK_CONFIGS, TaskID, ActionType

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="RescueMesh AI Environment",
    description="OpenEnv-compliant environment for emergency communication restoration.",
    version="1.0.0",
)

# Allow dashboard (Vite dev server) and any local origin to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env registry: session_id → env instance
_envs: dict[str, RescueMeshEnv] = {}
_histories: dict[str, list[dict]] = {}
_grader = RescueMeshGrader()

DEFAULT_SESSION = "default"


def _get_env(session_id: str) -> RescueMeshEnv:
    if session_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _envs[session_id]


def _convert_to_dashboard_format(obs: dict) -> dict:
    """
    Convert observation format from environment to dashboard format.
    Dashboard expects field names like 'active' instead of 'is_active'.
    Dashboard expects 'connectivity' instead of 'coverage_ratio'.
    """
    if not obs:
        return obs
    
    # Convert nodes: is_active -> active, is_relay -> relay
    converted_nodes = []
    for node in obs.get("nodes", []):
        converted_nodes.append({
            "id": node.get("id"),
            "x": node.get("x"),
            "y": node.get("y"),
            "signal": node.get("signal"),
            "battery": node.get("battery"),
            "active": node.get("is_active", True),  # Convert is_active to active
            "relay": node.get("is_relay", False),  # Convert is_relay to relay
        })
    
    return {
        "nodes": converted_nodes,
        "obstacles": obs.get("obstacles", []),
        "connected_pairs": obs.get("connected_pairs", []),
        "coverage_ratio": obs.get("coverage_ratio", 0.0),
        "connectivity": obs.get("coverage_ratio", 0.0),  # Alias for dashboard
        "active_relays": obs.get("active_relays", 0),
        "step_count": obs.get("step_count", 0),
        "task_id": obs.get("task_id", ""),
        "max_steps": obs.get("max_steps", 0),
    }


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(default="easy_open_field", description="Task ID to start")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    session_id: str = Field(default=DEFAULT_SESSION)


class StepRequest(BaseModel):
    action_type: str = Field(..., description="place_relay | move_drone | boost_signal | reroute")
    target_x: float = Field(default=50.0, ge=0.0, le=100.0)
    target_y: float = Field(default=50.0, ge=0.0, le=100.0)
    node_id: Optional[str] = Field(default=None)
    boost_amount: float = Field(default=0.2, ge=0.0, le=1.0)
    session_id: str = Field(default=DEFAULT_SESSION)


class GraderRequest(BaseModel):
    session_id: str = Field(default=DEFAULT_SESSION)


class BaselineRequest(BaseModel):
    task_id: str = Field(default="easy_open_field")
    seed: int = Field(default=42)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "RescueMesh AI Environment",
        "version": "1.0.0",
        "endpoints": ["/tasks", "/reset", "/step", "/state", "/grader", "/baseline"],
    }


@app.get("/tasks")
def list_tasks():
    """Returns all available tasks and the full action schema."""
    tasks = []
    for tid, cfg in TASK_CONFIGS.items():
        tasks.append({
            "task_id": tid.value,
            "description": cfg["description"],
            "difficulty": cfg["difficulty"],
            "max_steps": cfg["max_steps"],
            "max_relays": cfg["max_relays"],
            "target_coverage": cfg["target_coverage"],
        })

    action_schema = {
        "type": "object",
        "properties": {
            "action_type": {
                "type": "string",
                "enum": [a.value for a in ActionType],
                "description": "Type of action to perform",
            },
            "target_x": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": "X coordinate on 100x100 grid",
            },
            "target_y": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 100.0,
                "description": "Y coordinate on 100x100 grid",
            },
            "node_id": {
                "type": "string",
                "description": "ID of existing node (required for boost_signal / reroute)",
                "nullable": True,
            },
            "boost_amount": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Signal boost magnitude (for boost_signal only)",
            },
        },
        "required": ["action_type"],
    }

    observation_schema = {
        "nodes": "list of {id, x, y, signal, battery, is_relay, is_active}",
        "obstacles": "list of {x, y, width, height, attenuation}",
        "connected_pairs": "list of [node_id_a, node_id_b]",
        "coverage_ratio": "float 0.0–1.0",
        "active_relays": "int",
        "step_count": "int",
        "task_id": "str",
        "max_steps": "int",
    }

    return {
        "tasks": tasks,
        "action_schema": action_schema,
        "observation_schema": observation_schema,
        "reward_info": {
            "place_relay": "+0.1 base, +0.3 × coverage_delta if positive",
            "move_drone": "+0.05 for valid move, -0.05 if no relay exists",
            "boost_signal": "+0.2 × boost_amount for effective boost",
            "reroute": "+0.15 for reviving inactive node, -0.05 if redundant",
            "coverage_milestone": "+1.0 bonus when target_coverage reached",
            "degradation": "-0.1 penalty if coverage drops > 5%",
        },
    }


@app.post("/reset")
def reset(req: ResetRequest):
    """Reset (or start) an episode. Returns initial observation."""
    try:
        env = RescueMeshEnv(task_id=req.task_id, seed=req.seed)
        _envs[req.session_id] = env
        _histories[req.session_id] = []
        obs = env.reset()
        _histories[req.session_id].append(obs)
        
        # Convert observation to dashboard format
        # Note: the converted state is kept in memory or handled by client 
        # to preserve EXACT OpenEnv response dictionary format
        _convert_to_dashboard_format(obs)
        
        return {"observation": obs}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Execute one action. Returns (observation, reward, done, info)."""
    env = _get_env(req.session_id)
    action = {
        "action_type": req.action_type,
        "target_x": req.target_x,
        "target_y": req.target_y,
        "node_id": req.node_id,
        "boost_amount": req.boost_amount,
    }
    try:
        obs, reward, done, info = env.step(action)
        _histories[req.session_id].append(obs)
        
        # Convert observation to dashboard format
        _convert_to_dashboard_format(obs)
        
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = DEFAULT_SESSION):
    """Return full internal state for debugging."""
    env = _get_env(session_id)
    return env.state()


@app.post("/grader")
def grade(req: GraderRequest):
    """
    Score the completed episode. Returns score 0.0–1.0 with breakdown.
    Must call /reset and run some /step calls first.
    """
    if req.session_id not in _histories or not _histories[req.session_id]:
        raise HTTPException(status_code=400, detail="No episode history found. Run /reset and /step first.")
    result = _grader.grade(_histories[req.session_id])
    return result


@app.post("/baseline")
def run_baseline(req: BaselineRequest):
    """
    Run a simple heuristic baseline agent (no LLM required).
    Returns scores across all 3 tasks for reproducibility.
    """
    results = {}
    tasks = [TaskID.EASY, TaskID.MEDIUM, TaskID.HARD]

    for task in tasks:
        env = RescueMeshEnv(task_id=task.value, seed=req.seed)
        history = []
        obs = env.reset()
        history.append(obs)
        total_reward = 0.0

        for step_num in range(TASK_CONFIGS[task]["max_steps"]):
            action = _heuristic_action(obs, env.state(), step_num)
            obs, reward, done, info = env.step(action)
            history.append(obs)
            total_reward += reward
            if done:
                break

        score = _grader.grade(history)
        results[task.value] = {
            "score": score["score"],
            "breakdown": score["breakdown"],
            "total_reward": round(total_reward, 4),
            "steps": obs.get("step_count", 0),
        }

    return {
        "agent": "heuristic_baseline",
        "seed": req.seed,
        "results": results,
        "average_score": round(sum(r["score"] for r in results.values()) / len(results), 4),
    }


# ─────────────────────────────────────────────
# Heuristic baseline agent
# ─────────────────────────────────────────────

def _heuristic_action(obs: dict, state: dict, step_num: int) -> dict:
    """
    Simple rule-based agent:
    - First few steps: place relays at midpoints between base and survivors
    - Then: boost nodes with low signal
    - Finally: reroute dead nodes
    """
    nodes = obs.get("nodes", [])
    max_relays = state.get("max_relays", 5)
    relays_placed = state.get("relays_placed", 0)

    # Find base and survivor positions
    base = next((n for n in nodes if n["id"].startswith("base") or n["id"].startswith("surface")), None)
    survivors = [n for n in nodes if not n["id"].startswith("base") and
                 not n["id"].startswith("surface") and
                 not n["id"].startswith("relay")]

    # Phase 1: Place relays at midpoints (use roughly half budget)
    relay_budget = max_relays // 2 + 1
    if relays_placed < relay_budget and base and survivors:
        target = survivors[step_num % len(survivors)] if survivors else None
        if target:
            mx = (base["x"] + target["x"]) / 2 + (step_num % 3 - 1) * 5
            my = (base["y"] + target["y"]) / 2 + (step_num % 2 - 0.5) * 5
            return {
                "action_type": "place_relay",
                "target_x": round(min(95, max(5, mx)), 1),
                "target_y": round(min(95, max(5, my)), 1),
            }

    # Phase 2: Boost lowest-signal active survivor
    active_survivors = [n for n in survivors if n.get("is_active") and n["signal"] < 0.6]
    if active_survivors:
        target = min(active_survivors, key=lambda n: n["signal"])
        return {
            "action_type": "boost_signal",
            "node_id": target["id"],
            "boost_amount": 0.3,
        }

    # Phase 3: Reroute dead nodes
    dead = [n for n in nodes if not n.get("is_active") and not n["id"].startswith("relay")]
    if dead:
        return {
            "action_type": "reroute",
            "node_id": dead[0]["id"],
        }

    # Fallback: move a relay closer to centroid of survivors
    if survivors and relays_placed > 0:
        cx = sum(n["x"] for n in survivors) / len(survivors)
        cy = sum(n["y"] for n in survivors) / len(survivors)
        return {
            "action_type": "move_drone",
            "target_x": round(cx, 1),
            "target_y": round(cy, 1),
        }

    # Default no-op (still a valid action)
    return {"action_type": "boost_signal", "node_id": None, "boost_amount": 0.0}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
