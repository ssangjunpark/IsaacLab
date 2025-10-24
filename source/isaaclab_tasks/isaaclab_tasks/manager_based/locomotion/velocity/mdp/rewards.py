# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_air_time_balanced_biped(
    env,
    command_name: str,
    threshold: float,
    sensor_cfg,
    *,
    tolerance: float = 0.05,
    penalty_scale: float = 1.0,
    non_single_stance_penalty: float = 0.1,
    speed_gate: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] 
    in_contact = contact_time > 0.0
   
    in_mode_time = torch.where(in_contact, contact_time, air_time) 
    single_stance = (in_contact.int().sum(dim=1) == 1)
    # but only during single-stance (else 0). This keeps steps alternating & bounded.
    in_mode_time_masked = torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time))
    base_reward, _ = torch.min(in_mode_time_masked, dim=1)   # [B]
    base_reward = torch.clamp(base_reward, max=threshold)    # Imbalance penalty (only during single stance):

    tL = in_mode_time[:, 0]
    tR = in_mode_time[:, 1]
    rel_imbalance = (tL - tR).abs() / (tL + tR + eps)


    excess = torch.clamp(rel_imbalance - tolerance, min=0.0) # [B]
    imbalance_penalty = penalty_scale * excess * single_stance.float()    # Non-single-stance (double-stance or flight) penalty
    not_single = (~single_stance).float()
    phase_penalty = non_single_stance_penalty * not_single    # Gate by motion command magnitude (no reward when agent is intended to stand still)
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    moving = (torch.norm(cmd_xy, dim=1) > speed_gate).float()    
    reward = (base_reward - imbalance_penalty - phase_penalty) * moving

    reward = torch.clamp(reward, min=0.0)
    return torch.exp(-reward)

def feet_air_time_positive_biped_pen(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return torch.exp(-reward)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def lin_xy_bad(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalty for not tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return lin_vel_error / std**2

def leg_distance_bigger(
    env, joint_name, below_threshold, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    ):
    asset = env.scene[asset_cfg.name]
    idx = asset.find_joints(joint_name)

    q = asset.data.joint_pos[:, idx]

    abs_diff = torch.abs(q[:, 0] - q[:, 1])
    
    return max(0, torch.sum(abs_diff) - below_threshold)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def _contacts_now(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # shape: (N, H, B, 3): [envs, history, bodies, xyz]
    forces_hist = contact_sensor.data.net_forces_w_history
    feet_forces = forces_hist[:, -1, sensor_cfg.body_ids, :]
    in_contact = (feet_forces.norm(dim=-1) > threshold)
    return in_contact, feet_forces


def braking_impulse_penalty(env: ManagerBasedRLEnv,
                            sensor_cfg: SceneEntityCfg):
    """
    Penalize *negative* fore-aft GRF during stance (braking).
    Implementation detail:
      - Uses world X as forward. If your task's forward is body-yaw axis,
        replace with a transform to that frame.
    Returns: (N,) penalty (positive magnitude; your RewardTerm weight should be negative)
    """
    in_contact, feet_forces = _contacts_now(env, sensor_cfg)
    fx_world = feet_forces[..., 0]                                     # forward = +X (world)
    braking = torch.clamp(-fx_world, min=0.0) * in_contact.float()
    # Integrate over one step to approximate impulse (dt is scalar)
    return braking.mean(dim=-1) * env.cfg.sim.dt


def sprint_bonus(env: ManagerBasedRLEnv,
                 sensor_cfg: SceneEntityCfg,
                 min_speed: float = 4.0,
                 min_flight: float = 0.15):
    """
    Bonus if base forward speed >= min_speed AND duty factor < min_flight.
      - Base speed from robot root lin vel (world)
      - Duty factor from per-foot contacts at the current step
    Returns: (N,) bonus in [0,1]
    """
    robot = env.scene["robot"]
    v_world = robot.data.root_lin_vel_w                                # (N, 3)
    speed_ok = (v_world[..., 0] > min_speed)

    in_contact, _ = _contacts_now(env, sensor_cfg)
    duty_factor = in_contact.float().mean(dim=-1)                      # (N,)
    flight_ok = (duty_factor < min_flight)

    return (speed_ok & flight_ok).float()


def footstrike_behind_com_bonus(env: ManagerBasedRLEnv,
                                asset_cfg: SceneEntityCfg,
                                sensor_cfg: SceneEntityCfg,
                                x_offset_target: float = -0.06,
                                tolerance: float = 0.08):
    """
    Reward feet that are (on-contact) around x_offset_target *behind* COM (world X).
    Returns: (N,) average per-env score in [0,1]
    """
    robot = env.scene[asset_cfg.name]
    in_contact, _ = _contacts_now(env, sensor_cfg)

    # Positions (world)
    foot_pos_w = robot.data.body_pos_w[:, sensor_cfg.body_ids, :]      # (N, n_feet, 3)
    root_pos_w = robot.data.root_pos_w                                 # (N, 3)
    x_offset = foot_pos_w[..., 0] - root_pos_w[..., 0:1]               # (N, n_feet)

    err = (x_offset - x_offset_target).abs() * in_contact.float()
    good = (err < tolerance).float()

    denom = in_contact.float().sum(dim=-1).clamp(min=1.0)              # (N,)
    return good.sum(dim=-1) / denom


def symmetric_stride_bonus(env: ManagerBasedRLEnv,
                           sensor_cfg: SceneEntityCfg,
                           left_ids: list[int] | None = None,
                           right_ids: list[int] | None = None,
                           max_allowable_diff: float = 0.08):
    """
    Encourage similar instantaneous contact fraction for L/R groups.
    Args:
      left_ids/right_ids: lists of *indices inside sensor_cfg.body_ids* (not global IDs).
                          Pass these from your config for precise grouping.
                          If None, we split the feet set in half.
    Returns: (N,) bonus in [0,1]
    """
    in_contact, _ = _contacts_now(env, sensor_cfg)                     # (N, n_feet)

    n_feet = in_contact.shape[1]
    if not left_ids or not right_ids:
        # Fallback: first half = left, second half = right
        half = n_feet // 2
        left_idx = slice(0, half)
        right_idx = slice(half, n_feet)
        l_contact = in_contact[:, left_idx].float().mean(dim=-1)
        r_contact = in_contact[:, right_idx].float().mean(dim=-1)
    else:
        l_contact = in_contact[:, left_ids].float().mean(dim=-1)
        r_contact = in_contact[:, right_ids].float().mean(dim=-1)

    diff = (l_contact - r_contact).abs()
    return (diff < max_allowable_diff).float()