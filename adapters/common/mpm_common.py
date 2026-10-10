"""Shared helpers for single-environment MPM adapters."""
import numpy as np


def particle_poses(positions):
    positions = np.asarray(positions).reshape(-1, 3)
    out = np.zeros((len(positions), 7))
    out[:, :3] = positions
    out[:, 6] = 1  # Protocol padding: MPM particles have no rigid orientation.
    return out.tolist()


def common_settings(job):
    return dict(mpm=job["scene"]["mpm"], material=job["scene"]["mpm"]["material"],
                precision="f32", sleeping=False, world_execution="single", graph_capture=False,
                comparison="common initial material parameters, different constitutive/discretization implementations")
