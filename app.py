"""
RescueMesh AI Environment — FastAPI Server (app.py)
OpenEnv-compliant endpoints: /tasks, /reset, /step, /state, /grader, /baseline
"""

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
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
    redirect_slashes=False,  # Prevents 307 redirect issues with validators
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global session registry
_envs: dict[str, RescueMeshEnv] = {}
_histories: dict[str, list[dict]] = {}
_grader = RescueMeshGrader()

DEFAULT_SESSION = "default"


def _get_env(session_id: str) -> RescueMeshEnv:
    if session_id not in _envs:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first."
        )
    return _envs[session_id]


# ─────────────────────────────────────────────
# Pydantic models — all fields optional with defaults
# so validator can POST empty body {} without 422 errors
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(default="easy_open_field")
    seed: Optional[int] = Field(default=42)
    session_id: Optional[str] = Field(default=DEFAULT_SESSION)


class StepRequest(BaseModel):
    action_type: str = Field(..., description="place_relay | move_drone | boost_signal | reroute")
    target_x: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    target_y: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    node_id: Optional[str] = Field(default=None)
    boost_amount: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    session_id: Optional[str] = Field(default=DEFAULT_SESSION)


class GraderRequest(BaseModel):
    session_id: Optional[str] = Field(default=DEFAULT_SESSION)


class BaselineRequest(BaseModel):
    task_id: Optional[str] = Field(default="easy_open_field")
    seed: Optional[int] = Field(default=42)


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RescueMesh API is running", "status": "ok"}


@app.get("/tasks")
def list_tasks():
    """Returns all available tasks and the full action/observation schema."""
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
        "relays_placed": "int",   # NOTE: was active_relays — fixed to match environment.py
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
def reset(req: ResetRequest = Body(default=None)):
    """
    Reset (or start) an episode. Returns initial observation.
    Accepts empty body — all fields have safe defaults.
    """
    if req is None:
        req = ResetRequest()

    try:
        env = RescueMeshEnv(task_id=req.task_id, seed=req.seed)
        _envs[req.session_id] = env
        _histories[req.session_id] = []
        obs = env.reset()
        _histories[req.session_id].append(obs)
        return {"observation": obs}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """Execute one action. Returns (observation, reward, done, info)."""
    env = _get_env(req.session_id)
    try:
        obs, reward, done, info = env.step(req.dict())
        _histories[req.session_id].append(obs)
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state")
def state(session_id: str = DEFAULT_SESSION):
    """Return full internal state for debugging."""
    env = _get_env(session_id)
    return env.state()


@app.post("/grader")
def grade(req: GraderRequest = Body(default=None)):
    """
    Score the completed episode. Returns score 0.0–1.0 with breakdown.
    Must call /reset and run some /step calls first.
    """
    if req is None:
        req = GraderRequest()

    if req.session_id not in _histories or not _histories[req.session_id]:
        raise HTTPException(
            status_code=400,
            detail="No episode history found. Run /reset and /step first."
        )
    return _grader.grade(_histories[req.session_id])


@app.post("/baseline")
def run_baseline(req: BaselineRequest = Body(default=None)):
    """
    Run heuristic baseline agent across all 3 tasks.
    Returns scores for reproducibility benchmarking.
    """
    if req is None:
        req = BaselineRequest()

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
        "average_score": round(
            sum(r["score"] for r in results.values()) / len(results), 4
        ),
    }


# ─────────────────────────────────────────────
# Heuristic baseline agent
# ─────────────────────────────────────────────

def _heuristic_action(obs: dict, state: dict, step_num: int) -> dict:
    """
    Simple rule-based agent:
    - Phase 1: place relays at midpoints between base and survivors
    - Phase 2: boost nodes with low signal
    - Phase 3: reroute dead nodes
    """
    nodes = obs.get("nodes", [])
    max_relays = state.get("max_relays", 5)
    relays_placed = state.get("relays_placed", 0)  # Fixed: was active_relays

    base = next(
        (n for n in nodes if n["id"].startswith("base") or n["id"].startswith("surface")),
        None
    )
    survivors = [
        n for n in nodes
        if not n["id"].startswith("base")
        and not n["id"].startswith("surface")
        and not n["id"].startswith("relay")
    ]

    # Phase 1: place relays at midpoints
    relay_budget = max_relays // 2 + 1
    if relays_placed < relay_budget and base and survivors:
        target = survivors[step_num % len(survivors)]
        mx = (base["x"] + target["x"]) / 2 + (step_num % 3 - 1) * 5
        my = (base["y"] + target["y"]) / 2 + (step_num % 2 - 0.5) * 5
        return {
            "action_type": "place_relay",
            "target_x": round(min(95.0, max(5.0, mx)), 1),
            "target_y": round(min(95.0, max(5.0, my)), 1),
        }

    # Phase 2: boost lowest-signal active survivor
    active_low = [
        n for n in survivors
        if n.get("is_active") and n["signal"] < 0.6
    ]
    if active_low:
        target = min(active_low, key=lambda n: n["signal"])
        return {
            "action_type": "boost_signal",
            "node_id": target["id"],
            "boost_amount": 0.3,
        }

    # Phase 3: reroute dead nodes
    dead = [
        n for n in nodes
        if not n.get("is_active") and not n["id"].startswith("relay")
    ]
    if dead:
        return {"action_type": "reroute", "node_id": dead[0]["id"]}

    # Phase 4: move drone toward centroid of survivors
    if survivors and relays_placed > 0:
        cx = sum(n["x"] for n in survivors) / len(survivors)
        cy = sum(n["y"] for n in survivors) / len(survivors)
        return {
            "action_type": "move_drone",
            "target_x": round(cx, 1),
            "target_y": round(cy, 1),
        }

    # Safe fallback no-op
    return {"action_type": "boost_signal", "node_id": None, "boost_amount": 0.0}


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)