from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DatasetConfig:
    hist_timesteps: int = 10
    agent_hist_dim: int = 128
    agent_int_dim: int = 128
    road_dim: int = 64
    traffic_light_dim: int = 32
    num_agents: int = 20
    num_near_agents: int = 5
    num_road_segments: int = 50
    num_traffic_lights: int = 10
