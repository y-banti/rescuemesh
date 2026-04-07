"""Rescuemesh Environment Client."""

from typing import Dict, Optional
import requests


class StepResult:
    """Wrapper for step results."""
    def __init__(self, observation=None, reward=None, done=False):
        self.observation = observation
        self.reward = reward
        self.done = done


class RescuemeshEnv:
    """
    Client for the Rescuemesh Environment.

    This client maintains connections to the environment server,
    enabling efficient multi-step interactions.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> client = RescuemeshEnv(base_url="http://localhost:8000")
        >>> result = client.reset(task_id="easy_open_field")
        >>> result = client.step({"action_type": "place_relay", "target_x": 50, "target_y": 50})
    """

    def __init__(self, base_url: str = "http://localhost:8000", session_id: str = "default"):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    def reset(self, task_id: str = "easy_open_field", seed: int = 42):
        """Reset environment and return initial observation."""
        try:
            response = requests.post(
                f"{self.base_url}/reset",
                json={"task_id": task_id, "seed": seed, "session_id": self.session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to reset environment: {e}")

    def step(self, action: Dict):
        """Execute one step and return (observation, reward, done, info)."""
        try:
            response = requests.post(
                f"{self.base_url}/step",
                json={**action, "session_id": self.session_id}
            )
            response.raise_for_status()
            data = response.json()
            return (data["observation"], data["reward"], data["done"], data.get("info", {}))
        except Exception as e:
            raise RuntimeError(f"Failed to execute step: {e}")

    def state(self):
        """Get current environment state."""
        try:
            response = requests.get(
                f"{self.base_url}/state",
                params={"session_id": self.session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to get state: {e}")

    def grade(self):
        """Grade the completed episode."""
        try:
            response = requests.post(
                f"{self.base_url}/grader",
                json={"session_id": self.session_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to grade: {e}")
