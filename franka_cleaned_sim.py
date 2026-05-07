"""
Cleaned Franka robot dynamics demo.

Main purpose:
1) Free-space demo: robot moves with resolved-rate IK + joint-space PD.
2) Wall-contact demo: before contact uses joint-space PD; during contact uses
   Hybrid Force/Position Control (HPFC).

The script is organized as a small, readable simulation entry point with two
standard demos and a deterministic results folder.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.linalg import block_diag

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

# When this file is stored in: <repo_root>/CPBox/Franka_arm,
# add <repo_root> to sys.path so `from CPBox...` works when the script is
# launched directly from PyCharm or from the command line.
REPO_ROOT_CANDIDATE = SCRIPT_DIR.parent.parent
if SCRIPT_DIR.parent.name == "CPBox" and (REPO_ROOT_CANDIDATE / "CPBox").exists():
    repo_root_text = str(REPO_ROOT_CANDIDATE)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)

try:
    from CPBox.Solvers.CP_Judice_Child import CPJudice
except Exception:  # CPBox may be unavailable in a fresh environment.
    CPJudice = None


@dataclass
class SimulationConfig:
    """Contact solver settings."""

    solver_name: str = "judice"
    bound_init: str = "normprev"       # normprev / friczero
    mode_rescale: str = "inner"        # none / inner / outer


class FrankaRobotSimulator:
    """Franka FR3 simplified dynamics simulator used for PD + HPFC demos."""

    def __init__(self, cfg: Optional[SimulationConfig] = None):
        self.cfg = cfg or SimulationConfig()

        # -------------------------
        # Robot link inertial data
        # -------------------------
        self.robot_links = {
            "fr3_link0": {
                "com": np.array([-0.0172, 0.0004, 0.0745]),
                "inertia": np.array([[0.009, 0.0, 0.002],
                                     [0.0, 0.0115, 0.0],
                                     [0.002, 0.0, 0.0085]]),
                "mass": 2.3966,
            },
            "fr3_link1": {
                "com": np.array([4.128e-07, -0.0181251324, -0.038603597]),
                "inertia": np.array([[0.023927316485107913, 1.3317903455714081e-05, -0.00011404774918616684],
                                     [1.3317903455714081e-05, 0.0224821613275756, -0.0019950320628240115],
                                     [-0.00011404774918616684, -0.0019950320628240115, 0.006350098258530016]]),
                "mass": 2.9274653454,
            },
            "fr3_link2": {
                "com": np.array([0.0031828864, -0.0743221644, 0.0088146084]),
                "inertia": np.array([[0.041938946257609425, 0.00020257331521090626, 0.004077784227179924],
                                     [0.00020257331521090626, 0.02514514885014724, -0.0042252158006570156],
                                     [0.004077784227179924, -0.0042252158006570156, 0.06170214472888839]]),
                "mass": 2.9355370338,
            },
            "fr3_link3": {
                "com": np.array([0.0407015686, -0.0048200565, -0.0289730823]),
                "inertia": np.array([[0.02410142547240885, 0.002405265953945111, -0.001208114807676599],
                                     [0.002405265953945111, 0.01974053266708178, -0.002104212683891874],
                                     [-0.001208114807676599, -0.002104212683891874, 0.019044494482244823]]),
                "mass": 2.2449013699,
            },
            "fr3_link4": {
                "com": np.array([-0.0459100965, 0.063049296, -0.0085187868]),
                "inertia": np.array([[0.03452998321913202, 0.013226488001973316, 0.0062156735565278335],
                                     [0.013226488001973316, 0.028881621933049058, -0.0009762833870704552],
                                     [0.0062156735565278335, -0.0009762833870704552, 0.04125471171146641]]),
                "mass": 2.6155955791,
            },
            "fr3_link5": {
                "com": np.array([-0.0016039605, 0.0292536262, -0.097296599]),
                "inertia": np.array([[0.051610278463662895, -0.005715092815864025, -0.00680244902229547],
                                     [-0.005715092815864025, 0.04787729713371481, 0.010673985108535986],
                                     [-0.00680244902229547, 0.010673985108535986, 0.016423625579357254]]),
                "mass": 2.3271207594,
            },
            "fr3_link6": {
                "com": np.array([0.0597131221, -0.0410294666, -0.0101692726]),
                "inertia": np.array([[0.005412333594383447, 0.006192948624835693, 0.0014218436819489315],
                                     [0.006192948624835693, 0.014058329545509979, -0.0013140753741120031],
                                     [0.0014218436819489315, -0.0013140753741120031, 0.016080817924212554]]),
                "mass": 1.8170376524,
            },
            "fr3_link7": {
                "com": np.array([0.0045225817, 0.0086261921, -0.0161633251]),
                "inertia": np.array([[0.00021092389150104718, -2.433299114461931e-05, 4.564480393778983e-05],
                                     [-2.433299114461931e-05, 0.00017718568002411474, 8.744070223226438e-05],
                                     [4.564480393778983e-05, 8.744070223226438e-05, 5.993190599659971e-05]]),
                "mass": 0.6271432862,
            },
        }

        # -------------------------
        # Robot geometry / DH data
        # -------------------------
        self.n_joints = 7
        self.z_axis = np.array([0.0, 0.0, 1.0])

        # Last d includes tool/TCP extension from the original code.
        self.d_list = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107 + 0.3]
        self.a_list = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0]
        self.alpha_list = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2,
                           -np.pi / 2, np.pi / 2, np.pi / 2, 0.0]

        # -------------------------
        # Simulation parameters
        # -------------------------
        self.h = 0.01
        self.gravity = 9.80665
        self.steps_total = 300

        # Free-space task velocity.
        self.free_space_velocity = np.array([0.05, 0.0, 0.0])
        self.free_space_angular_velocity = np.zeros(3)
        # Optional piecewise free-space velocity schedule.
        # Format: [(end_step, velocity_xyz), ...]. If None, free_space_velocity is used.
        self.free_space_velocity_segments = None

        # Contact tangential task velocity.
        self.contact_tangent_velocity = np.array([0.0, 0.0, -0.07])
        self.contact_angular_velocity = np.zeros(3)

        # Joint PD gains.
        self.kp_joint = np.diag([800.0] * self.n_joints)
        self.kd_joint = np.diag([2.0 * np.sqrt(800.0)] * self.n_joints)

        # HPFC gains.
        self.desired_normal_force = 2.5
        self.force_kp = 0.1
        self.force_ki = 0.05
        self.task_kp = 15.0
        self.task_kd = 10.0
        self.normal_force_integral = 0.0

        # Wall/contact setup.
        self.enable_wall_contact = True
        self.use_global_contact_frame = False
        self.x_wall = 0.80
        self.wall_width_y = 1.0
        self.wall_height_z = 1.0
        self.wall_draw_x_offset = 0.01

        # Contact points are expressed in the end-effector/flange frame.
        # Shape: (N, 3).  N controls the contact-point number.
        # Default [[0, 0, 0]] means one contact point at the EE/TCP.
        self.contact_points_local = np.array([[0.0, 0.0, 0.0]], dtype=float)

        # Curved wall from the original contact demo.
        # Set wall_bump_amp = 0.0 in run_wall_hpfc_test() if a flat wall is needed.
        self.wall_bump_amp = -0.12
        self.wall_bump_center_z = 0.73
        self.wall_bump_sigma_z = 0.035

        self.friction_mu = 0.2
        self.constraint_damping = 1e4
        self.constraint_stiffness = 1e6

        # Contact state.
        self.collision_check_tol = 1e-5
        self.contact_indices: list[int] = []
        self.previous_contact_indices: list[int] = []
        self.phi_vals = np.array([np.inf])
        self.lambda_prev = np.zeros((0, 1))
        self.release_counts = np.zeros(1, dtype=int)
        self.phi_prev_contact = np.array([np.inf], dtype=float)

        # HPFC command state.
        self.xd_pos_cmd: Optional[np.ndarray] = None
        self.xdotd_pos_cmd = np.zeros(3)
        self.phi_prev_ctrl: Optional[float] = None
        self.normal_force_measured_filtered = 0.0

        # Debug values updated every step.
        self.print_debug = False
        self.last_contact_info = {
            "Fn_real": np.nan,
            "Fn_des": np.nan,
            "vt_real": np.nan,
            "vt_des_real": np.nan,
            "vn_real": np.nan,
        }

    # ============================================================
    # Basic math helpers
    # ============================================================

    @staticmethod
    def skew(v: np.ndarray) -> np.ndarray:
        """Return the skew-symmetric matrix of a 3D vector."""
        return np.array([[0.0, -v[2], v[1]],
                         [v[2], 0.0, -v[0]],
                         [-v[1], v[0], 0.0]])

    @staticmethod
    def mass_block(mass: float) -> np.ndarray:
        """3x3 translational mass block for one link."""
        return mass * np.eye(3)

    @staticmethod
    def relative_rotation_matrix(alpha: float, theta: float) -> np.ndarray:
        """Rotation between two adjacent robot frames."""
        rx = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(alpha), -np.sin(alpha)],
                       [0.0, np.sin(alpha), np.cos(alpha)]])
        rz = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                       [np.sin(theta), np.cos(theta), 0.0],
                       [0.0, 0.0, 1.0]])
        return rx @ rz

    # ============================================================
    # Forward kinematics and Jacobians
    # ============================================================

    def forward_rotation_chain(self, thetas: np.ndarray) -> list[np.ndarray]:
        """Return world-frame rotation matrices from base to each joint and flange."""
        thetas = np.asarray(thetas, dtype=float).reshape(self.n_joints)

        rotation_chain = [np.eye(3)]
        rotation_acc = np.eye(3)
        for alpha_i, theta_i in zip(self.alpha_list[: self.n_joints], thetas):
            rotation_acc = rotation_acc @ self.relative_rotation_matrix(alpha_i, theta_i)
            rotation_chain.append(rotation_acc.copy())

        # Flange has no extra rotation in this model.
        rotation_chain.append(rotation_chain[-1].copy())
        return rotation_chain

    def get_joint_positions(self, thetas: np.ndarray) -> np.ndarray:
        """Return positions of base, 7 joints, and flange/TCP. Shape: (9, 3)."""
        thetas = np.asarray(thetas, dtype=float).reshape(self.n_joints)
        rotation_chain = self.forward_rotation_chain(thetas)
        thetas_with_flange = np.concatenate([thetas, [0.0]])

        joint_positions = [np.zeros(3)]
        current_pos = np.zeros(3)

        for i in range(8):
            relative_rotation = self.relative_rotation_matrix(self.alpha_list[i], thetas_with_flange[i])
            local_offset = relative_rotation.T @ np.array([self.a_list[i], 0.0, 0.0]) + np.array([0.0, 0.0, self.d_list[i]])
            current_pos = current_pos + rotation_chain[i + 1] @ local_offset
            joint_positions.append(current_pos.copy())

        return np.vstack(joint_positions)

    def get_ee_pose(self, thetas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return end-effector position and rotation matrix in world frame."""
        ee_pos = self.get_joint_positions(thetas)[-1]
        ee_rot = self.forward_rotation_chain(thetas)[8]
        return ee_pos, ee_rot

    def get_contact_points_local(self) -> np.ndarray:
        """Return configured contact points in the end-effector frame. Shape: (N, 3)."""
        points = np.asarray(self.contact_points_local, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("contact_points_local must have shape (N, 3).")
        if points.shape[0] == 0:
            raise ValueError("At least one contact point is required.")
        return points

    def ensure_contact_state_size(self) -> None:
        """Resize contact-state arrays if contact_points_local was changed."""
        n_points = self.get_contact_points_local().shape[0]
        if self.phi_prev_contact.shape[0] != n_points:
            self.phi_prev_contact = np.full(n_points, np.inf, dtype=float)
            self.release_counts = np.zeros(n_points, dtype=int)
            self.phi_vals = np.full(n_points, np.inf, dtype=float)
            self.contact_indices = []
            self.previous_contact_indices = []
            self.lambda_prev = np.zeros((0, 1))

    def get_contact_points_world(self, thetas: np.ndarray) -> np.ndarray:
        """Convert local contact points to world coordinates. Shape: (N, 3)."""
        ee_pos, ee_rot = self.get_ee_pose(thetas)
        local_points = self.get_contact_points_local()
        return ee_pos.reshape(1, 3) + local_points @ ee_rot.T

    def get_primary_contact_point_world(self, thetas: np.ndarray) -> np.ndarray:
        """Return the first active contact point, or the first configured point."""
        points_world = self.get_contact_points_world(thetas)
        if self.contact_indices:
            first_idx = int(self.contact_indices[0])
            if 0 <= first_idx < points_world.shape[0]:
                return points_world[first_idx]
        return points_world[0]

    def contact_point_linear_jacobian(self, thetas: np.ndarray, contact_point_local: np.ndarray) -> np.ndarray:
        """Linear Jacobian for one configured contact point. Shape: (3, 7)."""
        ee_jac = self.ee_jacobian(thetas)
        ee_linear_jac = ee_jac[:3, :]
        ee_angular_jac = ee_jac[3:, :]
        _, ee_rot = self.get_ee_pose(thetas)
        offset_world = ee_rot @ np.asarray(contact_point_local, dtype=float).reshape(3)
        return ee_linear_jac - self.skew(offset_world) @ ee_angular_jac

    def angular_jacobian_single_link(
        self,
        link_index: int,
        rotation_i: np.ndarray,
        previous_angular_jacobian: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Angular Jacobian for one link COM. link_index is 0..7."""
        if link_index == 0:
            return np.zeros((3, self.n_joints))
        if previous_angular_jacobian is None:
            raise ValueError("previous_angular_jacobian is required when link_index > 0")

        jacobian = np.zeros((3, self.n_joints))
        jacobian[:, : link_index - 1] = previous_angular_jacobian[:, : link_index - 1]
        jacobian[:, link_index - 1] = rotation_i @ self.z_axis
        return jacobian

    def com_linear_jacobian_single_link(
        self,
        link_index: int,
        thetas: np.ndarray,
        rotation_i: np.ndarray,
        previous_angular_jacobian: np.ndarray,
        previous_joint_jacobian: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linear Jacobian at one link COM plus the updated joint-origin Jacobian."""
        if link_index == 0:
            return np.zeros((3, self.n_joints)), np.zeros((3, self.n_joints))

        thetas_with_flange = np.concatenate([thetas, [0.0]])
        relative_rotation = self.relative_rotation_matrix(self.alpha_list[link_index - 1], thetas_with_flange[link_index - 1])
        local_joint_offset = relative_rotation.T @ np.array([self.a_list[link_index - 1], 0.0, 0.0]) + np.array([0.0, 0.0, self.d_list[link_index - 1]])
        world_joint_offset = rotation_i @ local_joint_offset

        joint_jacobian_i = previous_joint_jacobian - self.skew(world_joint_offset) @ previous_angular_jacobian

        local_com_offset = self.robot_links[f"fr3_link{link_index}"]["com"]
        world_com_offset = rotation_i @ local_com_offset

        angular_jacobian_i = self.angular_jacobian_single_link(link_index, rotation_i, previous_angular_jacobian)
        com_jacobian_i = joint_jacobian_i - self.skew(world_com_offset) @ angular_jacobian_i
        return com_jacobian_i, joint_jacobian_i

    def com_to_joint_jacobian(self, thetas: np.ndarray) -> np.ndarray:
        """Stack all link COM Jacobians. Shape: (42, 7)."""
        thetas = np.asarray(thetas, dtype=float).reshape(self.n_joints)
        all_jacobians = np.zeros((6 * self.n_joints, self.n_joints))

        previous_joint_jacobian = np.zeros((3, self.n_joints))
        previous_angular_jacobian = np.zeros((3, self.n_joints))
        rotation_chain = self.forward_rotation_chain(thetas)

        for link_index in range(1, self.n_joints + 1):
            rotation_i = rotation_chain[link_index]
            com_jacobian_i, joint_jacobian_i = self.com_linear_jacobian_single_link(
                link_index,
                thetas,
                rotation_i,
                previous_angular_jacobian,
                previous_joint_jacobian,
            )
            angular_jacobian_i = self.angular_jacobian_single_link(link_index, rotation_i, previous_angular_jacobian)

            row0 = 6 * (link_index - 1)
            all_jacobians[row0: row0 + 3, :] = com_jacobian_i
            all_jacobians[row0 + 3: row0 + 6, :] = angular_jacobian_i

            previous_joint_jacobian = joint_jacobian_i
            previous_angular_jacobian = angular_jacobian_i

        return all_jacobians

    def com_to_joint_jacobian_dot(self, theta_dots: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Numerically estimate time derivative of the COM-to-joint Jacobian."""
        theta_dots = np.asarray(theta_dots, dtype=float).reshape(self.n_joints)
        thetas = np.asarray(thetas, dtype=float).reshape(self.n_joints)

        jac_plus = self.com_to_joint_jacobian(thetas + theta_dots * self.h)
        jac_minus = self.com_to_joint_jacobian(thetas - theta_dots * self.h)
        return (jac_plus - jac_minus) / (2.0 * self.h)

    def ee_jacobian(self, thetas: np.ndarray) -> np.ndarray:
        """End-effector Jacobian. Shape: (6, 7), ordered as [linear; angular]."""
        thetas = np.asarray(thetas, dtype=float).reshape(self.n_joints)
        rotation_chain = self.forward_rotation_chain(thetas)

        previous_joint_jacobian = np.zeros((3, self.n_joints))
        previous_angular_jacobian = np.zeros((3, self.n_joints))

        for link_index in range(1, self.n_joints + 1):
            rotation_i = rotation_chain[link_index]
            angular_jacobian_i = self.angular_jacobian_single_link(link_index, rotation_i, previous_angular_jacobian)
            _, joint_jacobian_i = self.com_linear_jacobian_single_link(
                link_index,
                thetas,
                rotation_i,
                previous_angular_jacobian,
                previous_joint_jacobian,
            )
            previous_angular_jacobian = angular_jacobian_i
            previous_joint_jacobian = joint_jacobian_i

        joint7_jacobian = previous_joint_jacobian
        angular_jacobian_7 = previous_angular_jacobian

        thetas_with_flange = np.concatenate([thetas, [0.0]])
        relative_rotation = self.relative_rotation_matrix(self.alpha_list[7], thetas_with_flange[7])
        local_7_to_ee = relative_rotation.T @ np.array([self.a_list[7], 0.0, 0.0]) + np.array([0.0, 0.0, self.d_list[7]])
        world_7_to_ee = rotation_chain[8] @ local_7_to_ee

        linear_jacobian = joint7_jacobian - self.skew(world_7_to_ee) @ angular_jacobian_7
        return np.vstack([linear_jacobian, angular_jacobian_7])

    def ee_jacobian_dot(self, thetas: np.ndarray, theta_dots: np.ndarray) -> np.ndarray:
        """Numerically estimate time derivative of the EE Jacobian."""
        theta_dots = np.asarray(theta_dots, dtype=float).reshape(self.n_joints)
        thetas = np.asarray(thetas, dtype=float).reshape(self.n_joints)

        jac_plus = self.ee_jacobian(thetas + theta_dots * self.h)
        jac_minus = self.ee_jacobian(thetas - theta_dots * self.h)
        return (jac_plus - jac_minus) / (2.0 * self.h)

    def solve_ik_velocity(self, thetas: np.ndarray, linear_velocity: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
        """Resolved-rate IK: desired EE twist -> desired joint velocity.

        A very small damped least-squares term is used only when the Jacobian is
        close to singular. Away from singularity this behaves like the standard
        pseudoinverse used in the original code.
        """
        desired_twist = np.hstack([linear_velocity, angular_velocity])
        jac = self.ee_jacobian(thetas)
        singular_values = np.linalg.svd(jac, compute_uv=False)
        min_sigma = float(np.min(singular_values)) if singular_values.size else 0.0

        if min_sigma > 1e-4:
            theta_dot_des = np.linalg.pinv(jac) @ desired_twist
        else:
            damping = 1e-3
            theta_dot_des = jac.T @ np.linalg.solve(jac @ jac.T + damping * damping * np.eye(6), desired_twist)

        # Safety limit for the demo trajectory; this prevents a sudden large joint
        # velocity if a user edits the free-space path too aggressively.
        max_joint_speed = 1.5
        speed_norm = float(np.linalg.norm(theta_dot_des, ord=np.inf))
        if speed_norm > max_joint_speed:
            theta_dot_des = theta_dot_des * (max_joint_speed / (speed_norm + 1e-12))
        return theta_dot_des

    # ============================================================
    # Robot dynamics
    # ============================================================

    def link_mass_matrix(self, thetas: np.ndarray) -> np.ndarray:
        """Build block diagonal mass/inertia matrix in world frame. Shape: (42, 42)."""
        blocks = []
        rotation_chain = self.forward_rotation_chain(thetas)

        for i in range(self.n_joints):
            link_name = f"fr3_link{i + 1}"
            mass = self.robot_links[link_name]["mass"]
            inertia_local = self.robot_links[link_name]["inertia"]
            rotation_i = rotation_chain[i + 1]
            inertia_world = rotation_i @ inertia_local @ rotation_i.T
            blocks.extend([self.mass_block(mass), inertia_world])

        return block_diag(*blocks)

    def link_angular_velocities(self, theta_dots: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Angular velocity of each link in world frame. Shape: (7, 3)."""
        theta_dots = np.asarray(theta_dots, dtype=float).reshape(self.n_joints)
        rotation_chain = self.forward_rotation_chain(thetas)

        angular_velocities = []
        current_w = np.zeros(3)
        for i in range(self.n_joints):
            current_w = current_w + theta_dots[i] * (rotation_chain[i + 1] @ self.z_axis)
            angular_velocities.append(current_w.copy())
        return np.vstack(angular_velocities)

    @staticmethod
    def link_coriolis_block(angular_velocity: np.ndarray, inertia_world: np.ndarray) -> np.ndarray:
        """One-link rotational Coriolis-like term: w x (I w)."""
        return np.cross(angular_velocity, inertia_world @ angular_velocity)

    def link_coriolis_vector(self, theta_dots: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Stack Coriolis-like wrench terms for all links. Shape: (42,)."""
        angular_velocities = self.link_angular_velocities(theta_dots, thetas)
        rotation_chain = self.forward_rotation_chain(thetas)

        coriolis_terms = []
        for i in range(self.n_joints):
            link_name = f"fr3_link{i + 1}"
            inertia_local = self.robot_links[link_name]["inertia"]
            inertia_world = rotation_chain[i + 1] @ inertia_local @ rotation_chain[i + 1].T
            coriolis_terms.extend([0.0, 0.0, 0.0])
            coriolis_terms.extend(self.link_coriolis_block(angular_velocities[i], inertia_world))

        return np.array(coriolis_terms)

    def gravity_wrench_vector(self) -> np.ndarray:
        """Stack gravity force for all links. Shape: (42, 1)."""
        gravity_wrench = np.zeros((6 * self.n_joints, 1))
        for i in range(self.n_joints):
            mass = self.robot_links[f"fr3_link{i + 1}"]["mass"]
            gravity_wrench[6 * i + 2, 0] = -mass * self.gravity
        return gravity_wrench

    def joint_dynamics_terms(self, thetas: np.ndarray, theta_dots: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return M(q), inv(M), c(q,dq), g(q) in joint space."""
        mass_link = self.link_mass_matrix(thetas)
        com_jac = self.com_to_joint_jacobian(thetas)
        com_jac_dot = self.com_to_joint_jacobian_dot(theta_dots, thetas)

        mass_joint = com_jac.T @ mass_link @ com_jac
        mass_joint_inv = np.linalg.inv(mass_joint)

        coriolis_link = self.link_coriolis_vector(theta_dots, thetas)
        coriolis_joint = com_jac.T @ coriolis_link + com_jac.T @ mass_link @ com_jac_dot @ theta_dots
        gravity_joint = com_jac.T @ self.gravity_wrench_vector().flatten()
        return mass_joint, mass_joint_inv, coriolis_joint, gravity_joint

    def pd_joint_torque(self, theta: np.ndarray, theta_dot: np.ndarray, theta_des: np.ndarray, theta_dot_des: np.ndarray) -> np.ndarray:
        """Joint-space PD torque/acceleration command."""
        return self.kp_joint @ (theta_des - theta) + self.kd_joint @ (theta_dot_des - theta_dot)

    # ============================================================
    # Wall/contact model
    # ============================================================

    def wall_height(self, y: float, z: float) -> float:
        """Wall surface height h(y,z), where x_surface = x_wall + h(y,z)."""
        amp = float(self.wall_bump_amp)
        if abs(amp) < 1e-12:
            return 0.0

        z_center = float(self.wall_bump_center_z)
        sigma = max(float(self.wall_bump_sigma_z), 1e-12)
        dz = z - z_center
        return float(amp * np.exp(-0.5 * (dz / sigma) ** 2))

    def wall_height_gradient(self, y: float, z: float) -> Tuple[float, float]:
        """Return gradient (dh/dy, dh/dz) of wall height."""
        amp = float(self.wall_bump_amp)
        if abs(amp) < 1e-12:
            return 0.0, 0.0

        z_center = float(self.wall_bump_center_z)
        sigma = max(float(self.wall_bump_sigma_z), 1e-12)
        dz = z - z_center
        exp_term = np.exp(-0.5 * (dz / sigma) ** 2)
        dh_dz = amp * exp_term * (-(dz / (sigma * sigma)))
        return 0.0, float(dh_dz)

    def wall_signed_distance(self, thetas: np.ndarray, check_wall_range: bool = True) -> np.ndarray:
        """
        Signed distance from each configured contact point to the wall.
        Positive: no contact. Zero/negative: contact or penetration.
        """
        self.ensure_contact_state_size()
        points_world = self.get_contact_points_world(thetas)
        phi_list = []

        for point_world in points_world:
            y, z = float(point_world[1]), float(point_world[2])
            x_surface = self.x_wall + self.wall_height(y, z)
            dh_dy, dh_dz = self.wall_height_gradient(y, z)
            normal_scale = np.sqrt(1.0 + dh_dy * dh_dy + dh_dz * dh_dz)
            phi = (x_surface - float(point_world[0])) / normal_scale

            if check_wall_range and (abs(y) > self.wall_width_y or abs(z) > self.wall_height_z):
                phi = max(phi, 0.0)
            phi_list.append(float(phi))

        return np.asarray(phi_list, dtype=float)

    def contact_frame_at_point(self, point_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return wall normal n and two tangent directions t1/t2 in world frame."""
        y, z = float(point_world[1]), float(point_world[2])
        _, dh_dz = self.wall_height_gradient(y, z)

        t1 = np.array([0.0, 1.0, 0.0], dtype=float)
        t2 = np.array([dh_dz, 0.0, 1.0], dtype=float)
        t2 = t2 / (np.linalg.norm(t2) + 1e-12)

        normal = np.cross(t1, t2)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        return normal, t1, t2

    def update_contact_state(self, thetas: np.ndarray) -> None:
        """Update active contact-point indices using signed distance and hysteresis."""
        self.ensure_contact_state_size()
        self.previous_contact_indices = list(self.contact_indices)

        if not self.enable_wall_contact:
            self.contact_indices = []
            self.phi_vals = np.full(self.get_contact_points_local().shape[0], np.inf, dtype=float)
            return

        phis = self.wall_signed_distance(thetas, check_wall_range=True)
        self.phi_vals = phis

        phi_dot_est = np.zeros_like(phis)
        finite_prev = np.isfinite(self.phi_prev_contact)
        phi_dot_est[finite_prev] = (phis[finite_prev] - self.phi_prev_contact[finite_prev]) / self.h
        self.phi_prev_contact = phis.copy()

        previous_set = set(self.previous_contact_indices)
        enter_tol = 1e-5
        soft_release_tol = 3e-4
        hard_release_tol = 3e-4
        release_need = 3

        new_contact_indices: list[int] = []
        for idx, phi_value in enumerate(phis):
            was_in_contact = idx in previous_set

            if was_in_contact:
                if phi_value > hard_release_tol:
                    self.release_counts[idx] = 0
                    continue

                release_candidate = (phi_value > soft_release_tol) and (phi_dot_est[idx] > 0.0)
                if release_candidate:
                    self.release_counts[idx] += 1
                    if self.release_counts[idx] < release_need:
                        new_contact_indices.append(idx)
                    else:
                        self.release_counts[idx] = 0
                else:
                    self.release_counts[idx] = 0
                    new_contact_indices.append(idx)
            else:
                if phi_value <= enter_tol:
                    self.release_counts[idx] = 0
                    new_contact_indices.append(idx)

        self.contact_indices = new_contact_indices

    def contact_jacobian_at_config(self, thetas: np.ndarray) -> np.ndarray:
        """Return stacked contact Jacobian W. Shape: (3*N_active, 7)."""
        if len(self.contact_indices) == 0:
            return np.zeros((0, self.n_joints))

        local_points = self.get_contact_points_local()
        points_world = self.get_contact_points_world(thetas)
        rows = []

        for contact_idx in self.contact_indices:
            point_linear_jac = self.contact_point_linear_jacobian(thetas, local_points[contact_idx])
            normal, t1, t2 = self.contact_frame_at_point(points_world[contact_idx])

            j_normal = (normal @ point_linear_jac).reshape(1, -1)
            j_t1 = (t1 @ point_linear_jac).reshape(1, -1)
            j_t2 = (t2 @ point_linear_jac).reshape(1, -1)
            rows.extend([-j_normal, j_t1, j_t2])

        return np.vstack(rows)

    def contact_jacobian_and_dot(self, thetas: np.ndarray, theta_dots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return contact Jacobian W and numerical Wdot."""
        w0 = self.contact_jacobian_at_config(thetas)
        w_plus = self.contact_jacobian_at_config(thetas + theta_dots * self.h)
        w_minus = self.contact_jacobian_at_config(thetas - theta_dots * self.h)
        w_dot = (w_plus - w_minus) / (2.0 * self.h)
        return w0, w_dot

    def initial_contact_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Bounds for CP solver. Tangential bounds use previous normal force."""
        n_contact = len(self.contact_indices)
        if n_contact == 0:
            return np.zeros((0, 1)), np.zeros((0, 1))

        upper_bounds = []
        lower_bounds = []
        for local_contact_id in range(n_contact):
            normal_prev = 0.0
            normal_slot = 3 * local_contact_id
            if self.cfg.bound_init == "normprev" and self.lambda_prev.size > normal_slot:
                normal_prev = max(0.0, float(self.lambda_prev[normal_slot, 0]))
            upper_bounds.extend([np.inf, self.friction_mu * normal_prev, self.friction_mu * normal_prev])
            lower_bounds.extend([0.0, -self.friction_mu * normal_prev, -self.friction_mu * normal_prev])

        return np.array(upper_bounds).reshape(-1, 1), np.array(lower_bounds).reshape(-1, 1)

    def contact_regularizer(self) -> Tuple[np.ndarray, np.ndarray]:
        """Regularization used by the contact solve."""
        if len(self.contact_indices) == 0:
            return np.zeros((0, 0)), np.zeros((0, 1))

        c_rows = []
        d_rows = []
        for idx in self.contact_indices:
            phi = float(self.phi_vals[idx])
            c_n = 1.0 / (self.h * self.constraint_damping + self.h * self.h * self.constraint_stiffness)
            d_n = phi / (self.h + self.constraint_damping / self.constraint_stiffness)
            c_rows.append([c_n, 0.0, 0.0])
            d_rows.append([d_n, 0.0, 0.0])

        return np.diag(np.ravel(c_rows)), np.vstack(d_rows).reshape(-1, 1)

    def project_contact_force_box(self, force: np.ndarray) -> np.ndarray:
        """Project one contact force to normal >= 0 and box friction limits."""
        force = np.asarray(force, dtype=float).reshape(3)
        normal_force = max(0.0, float(force[0]))
        tangential_limit = self.friction_mu * normal_force
        force[0] = normal_force
        force[1] = np.clip(force[1], -tangential_limit, tangential_limit)
        force[2] = np.clip(force[2], -tangential_limit, tangential_limit)
        return force

    def project_contact_force_vector(self, force: np.ndarray) -> np.ndarray:
        """Project stacked contact forces. Each contact uses [normal, tangent1, tangent2]."""
        force = np.asarray(force, dtype=float).reshape(-1)
        if force.size % 3 != 0:
            raise ValueError("The stacked contact force vector must have length 3*N.")
        for row0 in range(0, force.size, 3):
            force[row0: row0 + 3] = self.project_contact_force_box(force[row0: row0 + 3])
        return force

    def fallback_contact_solve(self, matrix_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
        """
        Small projected-gradient fallback when CPBox is unavailable.
        This is mainly for demo/testing, not a full replacement for CPJudice.
        """
        if matrix_a.size == 0:
            return np.zeros((0, 1))

        matrix_a = np.asarray(matrix_a, dtype=float)
        vector_b = np.asarray(vector_b, dtype=float).reshape(-1)
        n_vars = vector_b.size

        if self.lambda_prev.size == n_vars:
            force = self.lambda_prev.reshape(-1).copy()
        else:
            force = np.zeros(n_vars, dtype=float)

        step = 1.0 / (np.linalg.norm(matrix_a, ord=2) + 1e-9)
        for _ in range(80):
            grad = matrix_a @ force + vector_b
            force = force - step * grad
            for start in range(0, n_vars, 3):
                force[start: start + 3] = self.project_contact_force_box(force[start: start + 3])
        return force.reshape(-1, 1)

    def solve_contact_force(self, matrix_a: np.ndarray, vector_b: np.ndarray) -> np.ndarray:
        """Solve contact force using CPJudice when available; otherwise use fallback."""
        if len(self.contact_indices) == 0:
            return np.zeros((0, 1))

        upper_bounds, lower_bounds = self.initial_contact_bounds()
        if CPJudice is not None and self.cfg.solver_name == "judice":
            judice = CPJudice(self.friction_mu, matrix_a, vector_b, upper_bounds, lower_bounds, self.cfg.mode_rescale)
            judice.solve()
            return np.asarray(judice.cp_output["f"], dtype=float).reshape(-1, 1)

        return self.fallback_contact_solve(matrix_a, vector_b)

    # ============================================================
    # Hybrid force/position controller
    # ============================================================

    def operational_space_matrix(self, thetas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return EE Jacobian and operational-space inertia matrix."""
        mass_joint, mass_joint_inv, _, _ = self.joint_dynamics_terms(thetas, np.zeros(self.n_joints))
        del mass_joint
        ee_jac = self.ee_jacobian(thetas)
        op_mass = np.linalg.inv(ee_jac @ mass_joint_inv @ ee_jac.T + 1e-6 * np.eye(6))
        return ee_jac, op_mass

    def hpfc_projection_matrix(self, thetas: np.ndarray, reg: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Classic HPFC projector. Force-controlled direction is wall normal."""
        _, op_mass = self.operational_space_matrix(thetas)
        op_mass_inv = np.linalg.inv(op_mass + reg * np.eye(6))

        primary_contact_point = self.get_primary_contact_point_world(thetas)
        if self.use_global_contact_frame:
            normal = np.array([1.0, 0.0, 0.0])
        else:
            normal, _, _ = self.contact_frame_at_point(primary_contact_point)

        force_selector = np.hstack([normal.reshape(1, 3), np.zeros((1, 3))])
        selector_mass = force_selector @ op_mass_inv @ force_selector.T
        selector_mass = selector_mass + reg * np.eye(selector_mass.shape[0])
        projector = np.eye(6) - force_selector.T @ np.linalg.inv(selector_mass) @ force_selector @ op_mass_inv
        return projector, force_selector

    def hpfc_torque(
        self,
        thetas: np.ndarray,
        theta_dots: np.ndarray,
        desired_position: np.ndarray,
        desired_velocity: np.ndarray,
        desired_normal_force: float,
    ) -> np.ndarray:
        """Hybrid force/position controller torque command."""
        _, mass_joint_inv, coriolis_joint, gravity_joint = self.joint_dynamics_terms(thetas, theta_dots)
        bias_joint = coriolis_joint - gravity_joint

        ee_pos, _ = self.get_ee_pose(thetas)
        ee_jac = self.ee_jacobian(thetas)
        ee_jac_dot = self.ee_jacobian_dot(thetas, theta_dots)
        ee_twist = ee_jac @ theta_dots

        op_mass = np.linalg.inv(ee_jac @ mass_joint_inv @ ee_jac.T + 1e-6 * np.eye(6))
        projector, force_selector = self.hpfc_projection_matrix(thetas, reg=1e-6)
        force_projector = np.eye(6) - projector

        primary_contact_point = self.get_primary_contact_point_world(thetas)
        if self.use_global_contact_frame:
            normal = np.array([1.0, 0.0, 0.0])
        else:
            normal, _, _ = self.contact_frame_at_point(primary_contact_point)

        # Tangential position/velocity part.
        tangent_projector = np.eye(3) - np.outer(normal, normal)
        position_error_t = tangent_projector @ (desired_position - ee_pos)
        velocity_error_t = tangent_projector @ (desired_velocity - ee_twist[:3])

        motion_wrench = np.zeros(6)
        motion_wrench[:3] = self.task_kp * position_error_t + self.task_kd * velocity_error_t

        # Normal-force PI part.
        if self.lambda_prev.size >= 1:
            measured_normal_force = float(np.sum(self.lambda_prev[0::3, 0]))
        else:
            measured_normal_force = 0.0
        self.normal_force_measured_filtered = 0.8 * self.normal_force_measured_filtered + 0.2 * measured_normal_force
        force_error = desired_normal_force - self.normal_force_measured_filtered

        phi_values = self.wall_signed_distance(thetas, check_wall_range=True)
        phi_now = float(np.min(phi_values))
        if phi_now <= 0.0:
            self.normal_force_integral += self.h * force_error
        else:
            self.normal_force_integral *= 0.9
        self.normal_force_integral = float(np.clip(self.normal_force_integral, -50.0, 50.0))

        normal_force_cmd = desired_normal_force + self.force_kp * force_error + self.force_ki * self.normal_force_integral
        if phi_now > 0.0:
            normal_force_cmd = max(0.0, min(normal_force_cmd, 0.5))
        else:
            normal_force_cmd = max(0.0, normal_force_cmd)

        force_wrench = (force_selector.T * normal_force_cmd).reshape(6)

        # Operational-space bias term.
        eta_op = op_mass @ (ee_jac @ (mass_joint_inv @ bias_joint) - ee_jac_dot @ theta_dots)

        wrench_cmd = projector @ motion_wrench + force_projector @ force_wrench + eta_op
        return ee_jac.T @ wrench_cmd

    # ============================================================
    # One-step dynamics and full episode simulation
    # ============================================================

    def desired_ee_velocity(self, in_contact: bool, step_index: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Return desired EE velocity depending on free/contact mode."""
        if in_contact:
            return self.contact_tangent_velocity, self.contact_angular_velocity

        if self.free_space_velocity_segments:
            for end_step, velocity in self.free_space_velocity_segments:
                if step_index < int(end_step):
                    return np.asarray(velocity, dtype=float).reshape(3), self.free_space_angular_velocity
            return np.asarray(self.free_space_velocity_segments[-1][1], dtype=float).reshape(3), self.free_space_angular_velocity

        return self.free_space_velocity, self.free_space_angular_velocity

    def step_dynamics(
        self,
        theta_dots: np.ndarray,
        thetas: np.ndarray,
        theta_des: np.ndarray,
        theta_dot_des: np.ndarray,
        theta_ddot_des: np.ndarray,
    ) -> np.ndarray:
        """Integrate one dynamics step. Free space uses PD; contact uses HPFC + contact solver."""
        in_contact = len(self.contact_indices) > 0
        was_in_contact = len(self.previous_contact_indices) > 0
        entered_contact = in_contact and not was_in_contact
        left_contact = (not in_contact) and was_in_contact

        if entered_contact:
            self.normal_force_integral = 0.0
            ee_pos, _ = self.get_ee_pose(thetas)
            self.xd_pos_cmd = ee_pos.copy()

        if left_contact:
            self.lambda_prev = np.zeros((0, 1))
            self.normal_force_integral = 0.0

        mass_joint, mass_joint_inv, coriolis_joint, gravity_joint = self.joint_dynamics_terms(thetas, theta_dots)

        if not in_contact:
            pd_cmd = self.pd_joint_torque(thetas, theta_dots, theta_des, theta_dot_des)
            actuator_torque = mass_joint @ (pd_cmd + theta_ddot_des) + coriolis_joint - gravity_joint
            rhs = actuator_torque - coriolis_joint + gravity_joint
            theta_dot_next = theta_dots + self.h * np.linalg.solve(mass_joint, rhs)
            self.last_contact_info = {
                "Fn_real": np.nan,
                "Fn_des": np.nan,
                "vt_real": np.nan,
                "vt_des_real": np.nan,
                "vn_real": np.nan,
            }
            return theta_dot_next

        # -------------------------
        # Contact branch: HPFC
        # -------------------------
        contact_jac, contact_jac_dot = self.contact_jacobian_and_dot(thetas, theta_dots)
        ee_pos, _ = self.get_ee_pose(thetas)
        contact_points_world = self.get_contact_points_world(thetas)
        primary_contact_idx = int(self.contact_indices[0])
        primary_contact_point = contact_points_world[primary_contact_idx]
        normal_real, tangent1_real, tangent2_real = self.contact_frame_at_point(primary_contact_point)

        if self.use_global_contact_frame:
            normal_ctrl = np.array([1.0, 0.0, 0.0])
            tangent2_ctrl = np.array([0.0, 0.0, 1.0])
            desired_contact_velocity = self.contact_tangent_velocity.copy()
        else:
            normal_ctrl = normal_real
            tangent2_ctrl = tangent2_real
            speed = float(np.linalg.norm(self.contact_tangent_velocity))
            tangent_direction = tangent2_real.copy()
            if float(np.dot(tangent_direction, self.contact_tangent_velocity)) < 0.0:
                tangent_direction = -tangent_direction
            desired_contact_velocity = speed * tangent_direction

        # Fade controller near release boundary to avoid force wind-up after separation.
        active_phi_values = self.phi_vals[self.contact_indices]
        phi_now = float(np.min(active_phi_values)) if active_phi_values.size else float(np.min(self.wall_signed_distance(thetas, True)))
        if self.phi_prev_ctrl is None:
            self.phi_prev_ctrl = phi_now
        phi_dot_est = (phi_now - self.phi_prev_ctrl) / self.h
        self.phi_prev_ctrl = phi_now

        if phi_now <= 0.0:
            contact_scale = 1.0
        elif phi_dot_est >= 0.0:
            contact_scale = 0.0
        else:
            contact_scale = 0.1

        desired_normal_force = contact_scale * self.desired_normal_force
        desired_velocity_use = contact_scale * desired_contact_velocity

        if self.xd_pos_cmd is None or contact_scale == 0.0:
            self.xd_pos_cmd = ee_pos.copy()
            self.xdotd_pos_cmd = np.zeros(3)
        else:
            tangent_projector = np.eye(3) - np.outer(normal_ctrl, normal_ctrl)
            xd_pos = self.xd_pos_cmd + self.h * desired_velocity_use
            xd_pos = ee_pos + contact_scale * (tangent_projector @ (xd_pos - ee_pos))
            self.xd_pos_cmd = xd_pos.copy()
            self.xdotd_pos_cmd = desired_velocity_use.copy()

        actuator_torque = self.hpfc_torque(
            thetas=thetas,
            theta_dots=theta_dots,
            desired_position=self.xd_pos_cmd,
            desired_velocity=desired_velocity_use,
            desired_normal_force=desired_normal_force,
        )

        # Contact force solve.
        matrix_a = self.h * (contact_jac @ mass_joint_inv @ contact_jac.T)
        velocity_now = (contact_jac @ theta_dots).reshape(-1, 1)
        velocity_from_dynamics = (contact_jac @ mass_joint_inv @ (actuator_torque - coriolis_joint + gravity_joint)).reshape(-1, 1)
        jacobian_dot_velocity = (contact_jac_dot @ theta_dots).reshape(-1, 1)
        vector_b = velocity_now + self.h * (velocity_from_dynamics + jacobian_dot_velocity)

        regularizer_c, regularizer_d = self.contact_regularizer()
        if regularizer_c.shape[0] > 0:
            matrix_a = matrix_a + regularizer_c
            vector_b = vector_b + regularizer_d

        contact_force = self.solve_contact_force(matrix_a, vector_b)
        self.lambda_prev = contact_force.copy()

        rhs = actuator_torque + gravity_joint - coriolis_joint + (contact_jac.T @ contact_force).ravel()
        theta_dot_next = theta_dots + self.h * (mass_joint_inv @ rhs)

        # Debug information.
        ee_twist = self.ee_jacobian(thetas) @ theta_dots
        ee_linear_velocity = ee_twist[:3]
        contact_force_world = np.zeros(3, dtype=float)
        for local_contact_id, contact_idx in enumerate(self.contact_indices):
            force_slice = contact_force[3 * local_contact_id: 3 * local_contact_id + 3, 0]
            lambda_n, lambda_t1, lambda_t2 = [float(v) for v in force_slice]
            n_i, t1_i, t2_i = self.contact_frame_at_point(contact_points_world[contact_idx])
            contact_force_world += (-lambda_n) * n_i + lambda_t1 * t1_i + lambda_t2 * t2_i

        fn_real = -float(np.dot(normal_real, contact_force_world))
        vt_real = float(np.dot(tangent2_real, ee_linear_velocity))
        vt_des_real = float(np.dot(tangent2_real, desired_velocity_use))
        vn_real = float(np.dot(normal_real, ee_linear_velocity))
        self.last_contact_info = {
            "Fn_real": fn_real,
            "Fn_des": desired_normal_force,
            "vt_real": vt_real,
            "vt_des_real": vt_des_real,
            "vn_real": vn_real,
        }

        if self.print_debug:
            mode = "GLOBAL" if self.use_global_contact_frame else "LOCAL"
            vt_ctrl = float(np.dot(tangent2_ctrl, ee_linear_velocity))
            print(
                f"[{mode}] phi={phi_now:+.3e}, Fn={fn_real:+.3f}/{desired_normal_force:+.3f}, "
                f"vt={vt_real:+.4f}/{vt_des_real:+.4f}, vt_ctrl={vt_ctrl:+.4f}, vn={vn_real:+.3e}"
            )

        return theta_dot_next

    def simulate_episode(self, theta_dots_init: np.ndarray, thetas_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run one simulation episode and return theta_dot history and theta history."""
        theta_dots = np.asarray(theta_dots_init, dtype=float).reshape(self.n_joints).copy()
        thetas = np.asarray(thetas_init, dtype=float).reshape(self.n_joints).copy()

        theta_dots_history = [theta_dots.copy()]
        thetas_history = [thetas.copy()]

        theta_ref = thetas.copy()
        theta_dot_des_prev = np.zeros_like(theta_dots)
        ee0, _ = self.get_ee_pose(thetas)
        self.xd_pos_cmd = ee0.copy()

        for step_index in range(self.steps_total):
            self.update_contact_state(thetas)
            in_contact = len(self.contact_indices) > 0

            if not in_contact:
                if len(self.previous_contact_indices) > 0:
                    # Realign reference after leaving contact, so PD does not jump.
                    theta_ref = thetas.copy()
                    theta_dot_des_prev = np.zeros_like(theta_dots)

                linear_vel_des, angular_vel_des = self.desired_ee_velocity(in_contact=False, step_index=step_index)
                theta_dot_des = self.solve_ik_velocity(thetas, linear_vel_des, angular_vel_des)
                theta_ref = theta_ref + self.h * theta_dot_des
                theta_des = theta_ref.copy()
                theta_ddot_des = (theta_dot_des - theta_dot_des_prev) / self.h
                theta_dot_des_prev = theta_dot_des.copy()
            else:
                # In contact, tangential velocity and force target are generated inside step_dynamics().
                theta_dot_des = np.zeros_like(theta_dots)
                theta_des = thetas.copy()
                theta_ddot_des = np.zeros_like(theta_dots)
                theta_dot_des_prev = theta_dot_des.copy()

            theta_dots = self.step_dynamics(theta_dots, thetas, theta_des, theta_dot_des, theta_ddot_des)
            thetas = thetas + self.h * theta_dots

            theta_dots_history.append(theta_dots.copy())
            thetas_history.append(thetas.copy())

        return np.asarray(theta_dots_history), np.asarray(thetas_history)

    # ============================================================
    # Drawing / animation
    # ============================================================

    def draw_wall_surface(self, ax, resolution: int = 40, alpha: float = 0.35):
        """Draw wall surface x = x_wall + h(y,z) and return plotted grids."""
        y = np.linspace(-self.wall_width_y, self.wall_width_y, resolution)
        z = np.linspace(0.0, self.wall_height_z, resolution)
        y_grid, z_grid = np.meshgrid(y, z, indexing="xy")
        x_grid = self.x_wall + np.vectorize(self.wall_height)(y_grid, z_grid) + self.wall_draw_x_offset
        ax.plot_surface(x_grid, y_grid, z_grid, linewidth=0, antialiased=True, alpha=alpha)
        return x_grid, y_grid, z_grid

    def draw_wall_edges(self, ax, lw: float = 1.2, alpha: float = 0.8) -> None:
        """Draw the wall rectangle outline."""
        y1, y2 = -self.wall_width_y, self.wall_width_y
        z1, z2 = 0.0, self.wall_height_z
        outline_y = [y1, y2, y2, y1, y1]
        outline_z = [z1, z1, z2, z2, z1]
        outline_x = [self.x_wall + self.wall_height(outline_y[k], outline_z[k]) + self.wall_draw_x_offset for k in range(5)]
        ax.plot(outline_x, outline_y, outline_z, color="k", lw=lw, alpha=alpha)

    def draw_wall_centerline(self, ax, y0: float = 0.0, n: int = 200, lw: float = 1.0, alpha: float = 0.9) -> None:
        """Draw the wall profile along y = y0, useful for seeing the curved wall."""
        z_values = np.linspace(0.0, self.wall_height_z, n)
        y_values = np.full_like(z_values, y0, dtype=float)
        x_values = self.x_wall + np.array([self.wall_height(float(y0), float(z)) for z in z_values]) + self.wall_draw_x_offset
        ax.plot(x_values, y_values, z_values, color="k", lw=lw, alpha=alpha)

    def animate_robot(
        self,
        thetas_history: np.ndarray,
        draw_wall: bool = True,
        interval: int = 40,
        show: bool = True,
        save_path: Optional[str] = None,
        fps: int = 30,
        tail_len: int = 60,
        rod_lw: float = 5.0,
        wall_resolution: int = 60,
    ) -> FuncAnimation:
        """Create the same 3D-style animation used by the original demo.

        The animation uses every simulated frame. No frame skipping or GIF
        acceleration is applied.
        """
        if thetas_history is None or len(thetas_history) == 0:
            raise ValueError("thetas_history is empty.")

        thetas_history = np.asarray(thetas_history, dtype=float)
        all_joint_positions = np.array([self.get_joint_positions(theta) for theta in thetas_history])
        ee_positions = all_joint_positions[:, -1, :]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        wall_grids = None
        if draw_wall and self.enable_wall_contact:
            wall_grids = self.draw_wall_surface(ax, resolution=wall_resolution, alpha=0.35)
            self.draw_wall_edges(ax, lw=1.2, alpha=0.8)
            self.draw_wall_centerline(ax, y0=0.0, n=200, lw=1.0, alpha=0.9)

        main_segments0 = np.zeros((8, 2, 3))
        tip_segments0 = np.zeros((1, 2, 3))

        robot_links = Line3DCollection(main_segments0, linewidths=float(rod_lw), colors="navy", alpha=1.0)
        robot_tip = Line3DCollection(tip_segments0, linewidths=float(rod_lw) * 1.15, colors="limegreen", alpha=1.0)
        robot_links_highlight = Line3DCollection(main_segments0, linewidths=max(0.8, float(rod_lw) * 0.25), colors="white", alpha=0.22)
        robot_tip_highlight = Line3DCollection(tip_segments0, linewidths=max(0.8, float(rod_lw) * 0.28), colors="white", alpha=0.22)
        ax.add_collection3d(robot_links)
        ax.add_collection3d(robot_tip)
        ax.add_collection3d(robot_links_highlight)
        ax.add_collection3d(robot_tip_highlight)

        joint_marker_size = 0.90 * float(rod_lw)
        ee_marker_size = 1.15 * float(rod_lw)
        joint_dots, = ax.plot([], [], [], linestyle="None", marker="o",
                              markersize=joint_marker_size,
                              markerfacecolor="crimson", markeredgecolor="white",
                              markeredgewidth=0.8)
        ee_dot, = ax.plot([], [], [], linestyle="None", marker="o",
                          markersize=ee_marker_size,
                          markerfacecolor="gold", markeredgecolor="black",
                          markeredgewidth=1.0)
        ee_tail, = ax.plot([], [], [], "-", lw=max(1.2, 0.45 * float(rod_lw)), alpha=0.75)

        xyz = all_joint_positions.reshape(-1, 3)
        x_min, y_min, z_min = np.min(xyz, axis=0)
        x_max, y_max, z_max = np.max(xyz, axis=0)
        if wall_grids is not None:
            xg, yg, zg = wall_grids
            x_min = min(float(x_min), float(np.min(xg)))
            x_max = max(float(x_max), float(np.max(xg)))
            y_min = min(float(y_min), float(np.min(yg)))
            y_max = max(float(y_max), float(np.max(yg)))
            z_min = min(float(z_min), float(np.min(zg)))
            z_max = max(float(z_max), float(np.max(zg)))

        pad = 0.05 * max(1e-6, float(x_max - x_min), float(y_max - y_min), float(z_max - z_min))
        ax.set_xlim(float(x_min) - pad, float(x_max) + pad)
        ax.set_ylim(float(y_min) - pad, float(y_max) + pad)
        ax.set_zlim(max(0.0, float(z_min) - pad), float(z_max) + pad)
        ax.set_box_aspect([1, 1, 1.5])
        ax.view_init(elev=10, azim=-110)

        # Match the clean original animation style.
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        except Exception:
            pass
        ax.set_title("")

        def build_main_segments(points: np.ndarray) -> np.ndarray:
            segments = np.zeros((8, 2, 3))
            for i in range(8):
                segments[i, 0, :] = points[i]
                segments[i, 1, :] = points[i + 1]
            return segments

        def build_tip_segments(points: np.ndarray) -> np.ndarray:
            segment = np.zeros((1, 2, 3))
            segment[0, 0, :] = points[7]
            segment[0, 1, :] = points[8]
            return segment

        def init_animation():
            robot_links.set_segments(main_segments0)
            robot_tip.set_segments(tip_segments0)
            robot_links_highlight.set_segments(main_segments0)
            robot_tip_highlight.set_segments(tip_segments0)
            joint_dots.set_data([], [])
            joint_dots.set_3d_properties([])
            ee_dot.set_data([], [])
            ee_dot.set_3d_properties([])
            ee_tail.set_data([], [])
            ee_tail.set_3d_properties([])
            return robot_links, robot_tip, robot_links_highlight, robot_tip_highlight, joint_dots, ee_dot, ee_tail

        def update_animation(frame_idx: int):
            points = all_joint_positions[frame_idx]
            main_segments = build_main_segments(points)
            tip_segments = build_tip_segments(points)
            robot_links.set_segments(main_segments)
            robot_tip.set_segments(tip_segments)
            robot_links_highlight.set_segments(main_segments)
            robot_tip_highlight.set_segments(tip_segments)

            joints = points[:8]
            joint_dots.set_data(joints[:, 0], joints[:, 1])
            joint_dots.set_3d_properties(joints[:, 2])

            ee = points[8]
            ee_dot.set_data([ee[0]], [ee[1]])
            ee_dot.set_3d_properties([ee[2]])

            if tail_len and tail_len > 0:
                tail_start = max(0, frame_idx - int(tail_len))
                tail = ee_positions[tail_start: frame_idx + 1]
                ee_tail.set_data(tail[:, 0], tail[:, 1])
                ee_tail.set_3d_properties(tail[:, 2])
            else:
                ee_tail.set_data([], [])
                ee_tail.set_3d_properties([])

            return robot_links, robot_tip, robot_links_highlight, robot_tip_highlight, joint_dots, ee_dot, ee_tail

        animation = FuncAnimation(
            fig,
            update_animation,
            frames=all_joint_positions.shape[0],
            init_func=init_animation,
            interval=interval,
            blit=False,
            repeat=False,
            cache_frame_data=False,
        )

        if save_path:
            save_path_obj = Path(save_path)
            save_path_obj.parent.mkdir(parents=True, exist_ok=True)
            animation.save(str(save_path_obj), writer=PillowWriter(fps=fps))

        if show:
            plt.show()
        else:
            plt.close(fig)

        return animation


# ============================================================
# Demo functions used by main
# ============================================================


def default_initial_state() -> Tuple[np.ndarray, np.ndarray]:
    """Common initial state for both demos; matches the original contact demo."""
    thetas_init = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571 + np.pi / 2, 0.0])
    theta_dots_init = np.zeros(7)
    return thetas_init, theta_dots_init


def get_results_dir(results_dir: Optional[str] = None) -> Path:
    """Return the directory used for generated GIF files."""
    path = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_free_space_test(
    show: bool = False,
    save_gif: bool = True,
    results_dir: Optional[str] = None,
) -> Tuple[FrankaRobotSimulator, np.ndarray, np.ndarray]:
    """Demo 1: no wall. IK builds a joint reference; joint-space PD drives the arm."""
    model = FrankaRobotSimulator()
    model.enable_wall_contact = False
    model.steps_total = 420

    # A longer free-space trajectory with small piecewise Cartesian motions.
    # The motion stays near the original initial configuration and avoids singular regions.
    model.free_space_velocity_segments = [
        (140, np.array([0.000, 0.030, 0.000])),
        (280, np.array([0.035, 0.000, 0.000])),
        (420, np.array([0.000, 0.000, -0.020])),
    ]

    thetas_init, theta_dots_init = default_initial_state()
    theta_dots_history, thetas_history = model.simulate_episode(theta_dots_init, thetas_init)

    ee_start, _ = model.get_ee_pose(thetas_history[0])
    ee_end, _ = model.get_ee_pose(thetas_history[-1])
    print("\n[Free-space test]")
    print(f"  Initial joint config: {np.round(thetas_history[0], 4)}")
    print(f"  EE start: {np.round(ee_start, 4)}")
    print(f"  EE end:   {np.round(ee_end, 4)}")
    print(f"  EE move:  {np.round(ee_end - ee_start, 4)}")
    print("  Controller: resolved-rate IK -> joint-space PD")

    if save_gif:
        path = get_results_dir(results_dir) / "free_space_pd.gif"
        model.animate_robot(
            thetas_history,
            draw_wall=False,
            show=False,
            save_path=str(path),
            fps=30,
            tail_len=80,
        )
        print(f"  Saved GIF: {path.resolve()}")

    if show:
        model.animate_robot(thetas_history, draw_wall=False, show=True, tail_len=80)

    return model, theta_dots_history, thetas_history

def run_wall_hpfc_test(
    show: bool = False,
    save_gif: bool = True,
    results_dir: Optional[str] = None,
) -> Tuple[FrankaRobotSimulator, np.ndarray, np.ndarray]:
    """Demo 2: wall contact. Before contact uses IK + PD; during contact uses HPFC."""
    model = FrankaRobotSimulator()
    model.enable_wall_contact = True
    model.steps_total = 692

    # Contact number/positions can be changed here.
    # One row = one contact point in the end-effector/flange local frame.
    # Default is one EE/TCP contact point, matching the original demo.
    model.contact_points_local = np.array([[0.0, 0.0, 0.0]], dtype=float)

    # Free-space approach direction: move toward +X until reaching the wall.
    model.free_space_velocity = np.array([0.05, 0.0, 0.0])
    model.free_space_velocity_segments = None

    # During contact: keep desired normal force and move tangentially along the wall.
    model.contact_tangent_velocity = np.array([0.0, 0.0, -0.07])
    model.desired_normal_force = 2.5
    model.force_kp = 0.1
    model.force_ki = 0.05
    model.task_kp = 15.0
    model.task_kd = 10.0

    # Curved wall from the original contact demo. Set this to 0.0 for a flat wall.
    model.wall_bump_amp = -0.12
    model.wall_bump_center_z = 0.73
    model.wall_bump_sigma_z = 0.035

    thetas_init, theta_dots_init = default_initial_state()
    theta_dots_history, thetas_history = model.simulate_episode(theta_dots_init, thetas_init)

    ee_start, _ = model.get_ee_pose(thetas_history[0])
    ee_end, _ = model.get_ee_pose(thetas_history[-1])
    phi_start = float(np.min(model.wall_signed_distance(thetas_history[0])))
    phi_end = float(np.min(model.wall_signed_distance(thetas_history[-1])))
    x_surface_start = model.x_wall + model.wall_height(float(ee_start[1]), float(ee_start[2]))

    print("\n[Wall + HPFC test]")
    print(f"  Initial joint config: {np.round(thetas_history[0], 4)}")
    print(f"  EE start: {np.round(ee_start, 4)}")
    print(f"  EE end:   {np.round(ee_end, 4)}")
    print(f"  EE move:  {np.round(ee_end - ee_start, 4)}")
    print(f"  Wall x at initial EE y/z: {x_surface_start:.4f} m")
    print(f"  Initial wall distance phi: {phi_start:+.4e} m")
    print(f"  Final wall distance phi:   {phi_end:+.4e} m")
    print(f"  Wall bump amplitude: {model.wall_bump_amp:+.3f} m")
    print(f"  Contact points local: {model.contact_points_local.tolist()}")
    print(f"  Last contact info: {model.last_contact_info}")
    print("  Controller before contact: resolved-rate IK -> joint-space PD")
    print("  Controller during contact: Hybrid Force/Position Control")

    if save_gif:
        path = get_results_dir(results_dir) / "wall_hpfc.gif"
        model.animate_robot(
            thetas_history,
            draw_wall=True,
            show=False,
            save_path=str(path),
            fps=30,
            tail_len=80,
        )
        print(f"  Saved GIF: {path.resolve()}")

    if show:
        model.animate_robot(thetas_history, draw_wall=True, show=True, tail_len=80)

    return model, theta_dots_history, thetas_history

def main() -> None:
    parser = argparse.ArgumentParser(description="Franka free-space and wall-contact HPFC demos.")
    parser.add_argument("--demo", choices=["free", "wall", "both"], default="both",
                        help="Choose which demo to run. Default: both.")
    parser.add_argument("--show", action="store_true", help="Show matplotlib animation window.")
    parser.add_argument(
        "--no-save-gif",
        action="store_true",
        help="Run simulation without writing GIF files. By default, GIFs are saved.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="GIF output folder. Default: ./results next to this script.",
    )
    args = parser.parse_args()

    save_gif = not args.no_save_gif

    if args.demo in ("free", "both"):
        run_free_space_test(
            show=args.show,
            save_gif=save_gif,
            results_dir=args.results_dir,
        )

    if args.demo in ("wall", "both"):
        run_wall_hpfc_test(
            show=args.show,
            save_gif=save_gif,
            results_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
