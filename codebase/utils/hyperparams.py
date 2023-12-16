from dataclasses import dataclass, field
from typing import Tuple, List

from injection.scheludes.ready_schedules import InjectionSchedules
from utils.enums import AdvNormMethods, InjectionTypes, AssitiveActors


@dataclass
class Hyperparams:
    """ This is a dataclass providing default values for PPO hyperparameters. Note that
    those defaults are the ones performed the best for my implementation."""
    seed: int = 1234
    rollout_len: int = 4096
    max_episode_len: int = 1000
    gamma: float = 0.99
    gae_lambda: float = 0.98
    n_updates_per_iteration: int = 10
    clip: float = 0.2

    lr: float = 5e-3  # 3e-4
    min_lr: float = 1e-4

    batchify: bool = False
    batch_size: int = 256

    clip_grad: bool = False
    max_grad: float = 0.5

    max_kl_divergence: float = 1  # 0.02
    ent_coeff: float = 0  # 0.1
    advantage_calc: str = "gae"  # "classic" | "gae"

    learn_std: bool = False
    std_start: float = 0.25
    std_end: float = 0.01

    tanh_acts: bool = False

    # Additional hyperparams
    normalize_obs: bool = True
    hidden_dim: int = 64
    num_hidden_layers_actor: int = 2
    num_hidden_layers_critic: int = 2
    adv_norm_method: AdvNormMethods = AdvNormMethods.range_scale  # AdvNormMethods.range_scale


    # Injection hyperparams
    injection_enabled: bool = True
    injection_type: InjectionTypes = InjectionTypes.scheduled
    injection_schedule: InjectionSchedules = InjectionSchedules.sch2 # Only valid for InjectionTypes.scheduled
    injection_verbose: bool = False

    assistive_actor_type: AssitiveActors = AssitiveActors.mountaincar_basic # only valid for InjectionTypes.decremental
    assist_duration_perc: float = 0.01 # only valid for InjectionTypes.decremental
    decrement_noise_std: float = 0.15 # only valid for InjectionTypes.decremental

