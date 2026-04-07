from typing import List, Optional, Dict, Any

try:
    from openenv.core.env_server.types import Action, Observation
    from pydantic import Field, BaseModel

    class NodeState(BaseModel):
        id: str = Field(description="Node ID")
        x: float = Field(description="X coordinate")
        y: float = Field(description="Y coordinate")
        signal: float = Field(description="Node signal strength 0.0-1.0")
        battery: float = Field(description="Node battery level 0.0-1.0")
        is_relay: bool = Field(default=False)
        is_active: bool = Field(default=True)

    class ObstacleModel(BaseModel):
        x: float
        y: float
        width: float
        height: float
        attenuation: float

    class RescuemeshAction(Action):
        """Action for the Rescuemesh environment."""
        action_type: str = Field(..., description="Action type: place_relay, move_drone, boost_signal, reroute")
        target_x: float = Field(default=50.0)
        target_y: float = Field(default=50.0)
        node_id: Optional[str] = Field(default=None)
        boost_amount: float = Field(default=0.2)

    class RescuemeshObservation(Observation):
        """Observation/State of the Rescuemesh environment."""
        nodes: List[NodeState] = Field(default_factory=list)
        obstacles: List[ObstacleModel] = Field(default_factory=list)
        connected_pairs: List[List[str]] = Field(default_factory=list)
        coverage_ratio: float = Field(default=0.0)
        active_relays: int = Field(default=0)
        step_count: int = Field(default=0)
        task_id: str = Field(default="")
        max_steps: int = Field(default=0)

except ImportError:
    # openenv is not available — fall back to plain Pydantic models
    from pydantic import BaseModel, Field

    class NodeState(BaseModel):  # type: ignore[no-redef]
        id: str = Field(description="Node ID")
        x: float = Field(description="X coordinate")
        y: float = Field(description="Y coordinate")
        signal: float = Field(description="Node signal strength 0.0-1.0")
        battery: float = Field(description="Node battery level 0.0-1.0")
        is_relay: bool = Field(default=False)
        is_active: bool = Field(default=True)

    class ObstacleModel(BaseModel):  # type: ignore[no-redef]
        x: float
        y: float
        width: float
        height: float
        attenuation: float

    class RescuemeshAction(BaseModel):  # type: ignore[no-redef]
        action_type: str = Field(..., description="Action type: place_relay, move_drone, boost_signal, reroute")
        target_x: float = Field(default=50.0)
        target_y: float = Field(default=50.0)
        node_id: Optional[str] = Field(default=None)
        boost_amount: float = Field(default=0.2)

    class RescuemeshObservation(BaseModel):  # type: ignore[no-redef]
        nodes: List[NodeState] = Field(default_factory=list)
        obstacles: List[ObstacleModel] = Field(default_factory=list)
        connected_pairs: List[List[str]] = Field(default_factory=list)
        coverage_ratio: float = Field(default=0.0)
        active_relays: int = Field(default=0)
        step_count: int = Field(default=0)
        task_id: str = Field(default="")
        max_steps: int = Field(default=0)

        # OpenEnv client standard fields
        done: bool = Field(default=False)
        reward: Optional[float] = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)
