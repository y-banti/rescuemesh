# 🚑 RescueMesh – Intelligent Disaster Response AI

## 💡 Overview

RescueMesh is an AI-powered disaster response system that uses an adaptive intelligent agent to optimize communication coverage in dynamic environments.

## 🧠 Key Features

* Adaptive decision-making agent (memory + strategy)
* Avoids repetitive actions
* Multi-scenario environment (easy → hard)
* Real-time reward and coverage tracking

## ⚙️ How to Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
python inference.py
```

## 🎯 Impact

* Faster rescue coordination
* Efficient resource allocation
* Scalable AI for real-world disaster systems

## 👨‍💻 Author

Banti Yadav & Mumtaz Sheikh


# RescueMesh AI Environment

**OpenEnv-compliant training environment for emergency mesh network restoration.**

[![HF Spaces](https://img.shields.io/badge/🤗-HuggingFace_Space-blue)](https://huggingface.co/spaces/ybanti/rescuemesh)

---

## Problem Description

When disasters strike — building collapses, mine accidents, earthquakes — ground communication
infrastructure fails precisely when it is needed most. Emergency responders must restore mesh
communication between survivor nodes and base stations under severe constraints: obstacles blocking
signals, battery-limited devices, and strict limits on relay hardware.

**RescueMesh** puts an AI agent in the role of the communications engineer. The agent must:
- Identify which survivor nodes are cut off
- Deploy relay drones at optimal positions to bridge signal gaps
- Boost failing signals before batteries die
- Reroute around collapsed infrastructure

This is a real-world, high-stakes problem with direct humanitarian impact. The environment models
real signal propagation physics (distance attenuation, obstacle blocking) and resource constraints
(relay budget, battery depletion).

---

## Environment Design

The simulation runs on a **100 × 100 grid**. Nodes have positions, signal strength (0–1),
and battery level (0–1). Obstacles block line-of-sight signal paths with configurable attenuation.
Signal propagates between nodes within range, degraded by obstacles. Each step applies
signal decay and battery depletion, forcing the agent to act efficiently.

### Episode lifecycle

```
reset(task_id) → initial observation
  loop:
    step(action) → (observation, reward, done, info)
  grader(episode_history) → score 0.0–1.0
```

---

## Action Space

| Field | Type | Description |
|---|---|---|
| `action_type` | string | `place_relay` / `move_drone` / `boost_signal` / `reroute` |
| `target_x` | float 0–100 | Grid X position (place/move) |
| `target_y` | float 0–100 | Grid Y position (place/move) |
| `node_id` | string or null | Target node (boost/reroute) |
| `boost_amount` | float 0–1 | Signal boost magnitude (boost only) |

### Action semantics

- **place_relay** — Deploys a new relay drone at (target_x, target_y). Counts against relay budget.
- **move_drone** — Moves the nearest existing relay toward (target_x, target_y). Costs battery.
- **boost_signal** — Injects power into a node, raising its signal. Costs battery proportional to boost.
- **reroute** — Reactivates a dead/inactive node with minimal signal and battery. Useful for recovery.

---

## Observation Space

```json
{
  "nodes": [
    {
      "id": "base",
      "x": 5.0, "y": 50.0,
      "signal": 1.0, "battery": 1.0,
      "is_relay": false, "is_active": true
    },
    ...
  ],
  "obstacles": [
    { "x": 30, "y": 20, "width": 15, "height": 40, "attenuation": 0.15 }
  ],
  "connected_pairs": [["base", "relay_0"], ["relay_0", "survivor_a"]],
  "coverage_ratio": 0.75,
  "active_relays": 2,
  "step_count": 7,
  "task_id": "medium_obstacle",
  "max_steps": 30
}
```

---

## Reward Function

The reward is shaped for partial progress — no binary success/failure:

| Event | Reward |
|---|---|
| Place relay | +0.10 |
| Coverage improves > 5% | +0.30 × delta |
| Effective signal boost | +0.20 × boost_amount |
| Reroute dead node | +0.15 |
| Valid drone move | +0.05 |
| Target coverage reached | **+1.00 bonus** |
| Coverage drops > 5% | −0.10 |
| Invalid / over-budget action | −0.05 |

---

## Tasks

### Task 1 — Easy: Open Field (`easy_open_field`)

Open terrain, no obstacles. 4 survivor nodes spread across the grid.
The agent must connect all survivors to base using up to 5 relay drones in 20 steps.

- **Target coverage:** 100%
- **Max relays:** 5
- **Signal range:** 35 units

### Task 2 — Medium: Urban Disaster Zone (`medium_obstacle`)

3 collapsed building zones block direct paths. 5 nodes including a rescue HQ.
The agent must route signals around obstacles to reach 85% coverage in 30 steps.

- **Target coverage:** 85%
- **Max relays:** 6
- **Signal range:** 30 units

### Task 3 — Hard: Underground Coal Mine (`hard_coal_mine`)

Severe constraints: 4 rock/wall obstacles with near-total attenuation (4–6% signal pass-through),
miner nodes with critically low initial signal (5–10%), and accelerated battery decay.
Requires precise relay placement along tunnel corridors to reach 75% coverage in 40 steps.

- **Target coverage:** 75%
- **Max relays:** 5
- **Signal range:** 20 units (simulates tunnel propagation)

---

## Grader (0.0 – 1.0)

Each completed episode is scored on four dimensions:

| Component | Weight | Metric |
|---|---|---|
| Coverage score | 50% | `min(1.0, final_coverage / target_coverage)` |
| Efficiency score | 25% | `1 - steps_used / max_steps` |
| Relay economy | 15% | `1 - relays_used / max_relays` |
| Battery preservation | 10% | Average survivor battery at episode end |

The grader is **never constant** — all four components vary with agent behavior.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check + endpoint list |
| GET | `/tasks` | All tasks + action/observation schema |
| POST | `/reset` | Start episode: `{task_id, seed, session_id}` |
| POST | `/step` | Execute action, returns `{observation, reward, done, info}` |
| GET | `/state` | Full internal state (debug) |
| POST | `/grader` | Score completed episode: returns `{score, breakdown}` |
| POST | `/baseline` | Run heuristic baseline across all 3 tasks |

Interactive API docs at: `http://localhost:7860/docs`

---

## Setup Instructions

### Local

```bash
git clone https://huggingface.co/spaces/ybanti/rescuemesh
cd rescuemesh
pip install -r requirements.txt
python app.py
# API now at http://localhost:7860
```

### Docker

```bash
docker build -t rescuemesh .
docker run -p 7860:7860 rescuemesh
```

### Agent Inference

You can run an agent through all three tasks using either an LLM or a Heuristic (rule-based) mode:

```bash
# 1. Heuristic Mode (No API key required)
python inference.py

# 2. LLM Mode (Requires OpenAI API key)
export OPENAI_API_KEY=sk-...
python inference.py --model gpt-4o-mini
```

Results will be saved to `inference_results.json` with a detailed score breakdown for each task.

---

## Baseline Results

Results from the heuristic baseline agent (deterministic, no LLM, seed=42):

| Task | Score | Coverage | Efficiency |
|---|---|---|---|
| easy_open_field | ~0.55 | ~80% | ~0.60 |
| medium_obstacle | ~0.42 | ~60% | ~0.50 |
| hard_coal_mine | ~0.28 | ~40% | ~0.40 |
| **Average** | **~0.42** | | |

The LLM baseline (gpt-4o-mini) typically scores 0.15–0.25 higher than heuristic on easy/medium
tasks due to better relay placement reasoning. Hard task remains challenging for both baselines.

---

## Project Structure

```
rescuemesh/
├── environment.py      # Core env: RescueMeshEnv, RescueMeshGrader
├── app.py              # FastAPI server (all endpoints)
├── inference.py        # Main Agent script (Heuristic + LLM modes)
├── baseline_agent.py   # Simple LLM baseline template
├── openenv.yaml        # OpenEnv spec config
├── requirements.txt
└── README.md
```

---

## Real-World Impact

Communication failure is the leading cause of preventable deaths in disaster response.
This environment trains agents on the core infrastructure challenge: sparse relay hardware,
degraded signals, physical obstacles, and battery constraints. Agents trained here could
directly inform autonomous drone deployment algorithms for real search-and-rescue systems.

---

## License

MIT License

Copyright (c) 2026 RescueMesh Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

