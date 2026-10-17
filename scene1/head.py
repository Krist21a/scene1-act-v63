"""Aim the head camera at manipulation targets; restore the ACT camera pose."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def head_pivot(description):
    root = ET.fromstring(description)
    joints = {j.find('child').attrib['link']: j for j in root.findall('joint')}
    chain, frame = [], 'zhead_1_link'
    while frame != 'base_link':
        if frame not in joints or len(chain) > 20:
            raise ValueError('Head has no URDF chain to base_link')
        joint = joints[frame]
        chain.append(joint)
        frame = joint.find('parent').attrib['link']
    matrix = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find('origin')
        attrs = origin.attrib if origin is not None else {}
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', list(map(float, attrs.get('rpy', '0 0 0').split()))).as_matrix()
        transform[:3, 3] = list(map(float, attrs.get('xyz', '0 0 0').split()))
        matrix = matrix @ transform
    return matrix


class HeadLookController:
    def __init__(self, config, description):
        self.config = config
        self.home = np.deg2rad(config['home_degrees'])
        self.limit = np.deg2rad(config['limits_degrees'])
        self.speed = np.deg2rad(config['speed_degrees_s'])
        self.pivot = head_pivot(description)
        self.camera_local = None
        self.command = None
        self.home_frames = 0

    def head_matrix(self, angles):
        rotation = np.eye(4)
        rotation[:3, :3] = Rotation.from_euler('ZY', angles).as_matrix()
        return self.pivot @ rotation

    def calibrate(self, snapshot):
        if self.camera_local is None and snapshot is not None:
            self.camera_local = np.linalg.inv(self.head_matrix(snapshot['head_angles_rad'])) @ snapshot['camera_to_base']

    def desired(self, target, snapshot):
        self.calibrate(snapshot)
        if target is None or self.camera_local is None:
            return self.home.copy()
        target = np.asarray(target, dtype=float)
        def error(angles):
            camera = self.head_matrix(angles) @ self.camera_local
            point = camera[:3, :3].T @ (target - camera[:3, 3])
            direction = point / max(np.linalg.norm(point), 1e-6)
            return np.r_[direction[:2], min(0., direction[2]) * 2]
        start = self.command if self.command is not None else self.home
        result = least_squares(error, np.clip(start, -self.limit + 1e-6, self.limit - 1e-6),
                               bounds=(-self.limit, self.limit), max_nfev=30)
        return result.x

    def update(self, target, measured, snapshot, dt):
        measured = np.asarray(measured, dtype=float)
        if self.command is None:
            self.command = np.clip(measured, -self.limit, self.limit)
        desired = self.desired(target, snapshot)
        self.command += np.clip(desired - self.command, -self.speed * dt, self.speed * dt)
        at_home = (target is None and np.max(np.abs(measured - self.home))
                   <= math.radians(self.config['home_tolerance_degrees']))
        self.home_frames = self.home_frames + 1 if at_home else 0
        ready = self.home_frames >= self.config['stable_frames']
        return self.command.copy(), {'state': 'look_at_target' if target is not None else 'return_to_act_view',
                                     'target_base_m': None if target is None else np.asarray(target).tolist(),
                                     'desired_degrees': np.rad2deg(desired).tolist(),
                                     'command_degrees': np.rad2deg(self.command).tolist(),
                                     'measured_degrees': np.rad2deg(measured).tolist(),
                                     'act_view_ready': ready}
