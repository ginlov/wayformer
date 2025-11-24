from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DatasetConfig:
    ##############################################
    # Do not change, they depend on the dataset  #
    ##############################################
    hist_timesteps: int = 10
    future_timesteps: int = 40
    agent_hist_dim: int = 8
    agent_int_dim: int = 8
    road_dim: int = 6
    traffic_light_dim: int = 3
    ##############################################
    # You can change these parameters            #
    ##############################################
    num_near_agents: int = 6
    num_road_segments: int = 5000
    num_traffic_lights: int = 10
