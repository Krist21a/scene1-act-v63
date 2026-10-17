"""FK/IK from the public robot URDF, using the simulated 2F85 pinch frame.

Only robot geometry and joint sensors are used; no scene/object ground truth.
The official arm-only service currently uses a different upper-arm length.
"""
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .motion import action_vector


class UrdfArmKinematics:
    def __init__(self, rospy=None, description=None, tool_xyz=(0, 0, -0.1488)):
        if description is None:
            description = rospy.get_param('/humanoid_description')
        root = ET.fromstring(description)
        by_child = {j.find('child').attrib['link']: j for j in root.findall('joint')}
        self.chains = []
        self.limits = []
        self.tool_xyz = np.asarray(tool_xyz, dtype=float)
        for side in ('l', 'r'):
            chain = []
            frame = f'zarm_{side}7_link'
            while frame != 'base_link':
                if frame not in by_child or len(chain) > 20:
                    raise ValueError('Arm URDF chain does not reach base_link')
                joint = by_child[frame]
                chain.append(joint)
                frame = joint.find('parent').attrib['link']
            parsed, limits = [], []
            for joint in reversed(chain):
                origin = joint.find('origin')
                attrs = origin.attrib if origin is not None else {}
                tr = np.eye(4)
                tr[:3, :3] = Rotation.from_euler('xyz', [float(x) for x in attrs.get('rpy', '0 0 0').split()]).as_matrix()
                tr[:3, 3] = [float(x) for x in attrs.get('xyz', '0 0 0').split()]
                name = joint.attrib['name']
                index = None
                axis = np.zeros(3)
                if joint.attrib['type'] != 'fixed' and name != 'waist_yaw_joint':
                    if name not in [f'zarm_{side}{i}_joint' for i in range(1, 8)]:
                        raise ValueError(f'Unmapped arm joint {name}')
                    index = int(name[len(f'zarm_{side}'):].split('_')[0]) - 1
                    axis = np.array([float(x) for x in joint.find('axis').attrib['xyz'].split()])
                    axis /= np.linalg.norm(axis)
                    limit = joint.find('limit').attrib
                    limits.append((float(limit['lower']), float(limit['upper'])))
                parsed.append((tr, index, axis))
            if len(limits) != 7:
                raise ValueError('Expected seven arm joints')
            self.chains.append(parsed)
            self.limits.append(np.asarray(limits))

    def matrix(self, q, side):
        mat = np.eye(4)
        for origin, index, axis in self.chains[side]:
            mat = mat @ origin
            if index is not None:
                rot = np.eye(4)
                rot[:3, :3] = Rotation.from_rotvec(axis * q[index]).as_matrix()
                mat = mat @ rot
        mat[:3, 3] += mat[:3, :3] @ self.tool_xyz
        return mat

    def fk(self, action):
        action = action_vector(action)
        poses = []
        for side, q in enumerate((action[:7], action[8:15])):
            mat = self.matrix(q, side)
            poses.append(SimpleNamespace(pos_xyz=mat[:3, 3].tolist(),
                         quat_xyzw=Rotation.from_matrix(mat[:3, :3]).as_quat().tolist()))
        return SimpleNamespace(left_pose=poses[0], right_pose=poses[1])

    def ik(self, poses, seed, orientation_tolerance=.10, joint_offsets=None, upright_axes=None):
        seed = action_vector(seed)
        joints = []
        for side, (pose, q0) in enumerate(((poses.left_pose, seed[:7]), (poses.right_pose, seed[8:15]))):
            target_position = np.asarray(pose.pos_xyz)
            target_rotation = Rotation.from_quat(pose.quat_xyzw)
            upright_axis = None if upright_axes is None else upright_axes[side]
            if upright_axis is not None:
                upright_axis = np.asarray(upright_axis) / np.linalg.norm(upright_axis)
                target_normal = target_rotation.apply(upright_axis)
            orientation_weight = 0.15
            def residual(q):
                mat = self.matrix(q, side)
                rotation = Rotation.from_matrix(mat[:3, :3])
                rotation_error = ((target_rotation.inv() * rotation).as_rotvec()
                                  if upright_axis is None else rotation.apply(upright_axis) - target_normal)
                return np.r_[mat[:3, 3] - target_position,
                             orientation_weight * rotation_error,
                             0.001 * (q - q0)]
            limits = self.limits[side]
            if joint_offsets is not None:
                offset = np.asarray(joint_offsets)[side * 7:side * 7 + 7]
                limits = np.column_stack((np.maximum(limits[:, 0], limits[:, 0] - offset),
                                          np.minimum(limits[:, 1], limits[:, 1] - offset)))
            lower = np.maximum(limits[:, 0], q0 - 0.45)
            upper = np.minimum(limits[:, 1], q0 + 0.45)
            starts = [q0, q0, q0 + [0, 0, 0, -.12, 0, .12, 0],
                      q0 + [0, .08, 0, -.08, 0, -.12, 0]]
            for attempt, start in enumerate(starts):
                orientation_weight = 0.15 if attempt == 0 else 0.05
                solution = least_squares(residual, np.clip(start, lower + 1e-8, upper - 1e-8),
                                         bounds=(lower, upper), max_nfev=180,
                                         ftol=1e-8, xtol=1e-8, gtol=1e-8)
                error = residual(solution.x)
                angle = np.linalg.norm(error[3:6]) / orientation_weight
                if upright_axis is not None:
                    normal = Rotation.from_matrix(self.matrix(solution.x, side)[:3, :3]).apply(upright_axis)
                    angle = float(np.arccos(np.clip(normal @ target_normal, -1, 1)))
                if np.linalg.norm(error[:3]) <= .005 and angle <= orientation_tolerance:
                    break
            else:
                raise RuntimeError(f'Local URDF IK failed for arm {side}: position error {np.linalg.norm(error[:3]):.3f} m, orientation {angle:.3f} rad')
            joints.extend(solution.x)
        return np.asarray(joints)
