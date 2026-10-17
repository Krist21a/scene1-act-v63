"""Optional bounded roof-reference compensation for the first weighing target.

The held RGB-D component is dominated by its roof. Its horizontal projection
changes as the tilted parcel rests upright. This experiment shifts the target
once after leveling; original motion, release and settlement checks still run.
"""
import numpy as np


class PlacementTarget:
    def __init__(self, max_offset_m=0., axis=None):
        if not np.isfinite(max_offset_m) or not 0 <= max_offset_m <= .008:
            raise ValueError('Roof target compensation must be between 0 and 8 mm')
        self.max_offset_m = max_offset_m
        if axis not in (None, 0, 1):
            raise ValueError('Roof target axis must be X, Y or unrestricted')
        self.axis = axis
        self.applied = None
        self.info = None

    def before_step(self, assist):
        p = assist.pending
        if (self.max_offset_m == 0 or p is None or p['kind'] != 'weighing'
                or p.get('stage') != 'correct' or not p.get('level_ready')
                or self.applied is p):
            return
        seen = p.get('last_box_observation', {})
        normal = seen.get('roof_normal_base')
        if (normal is None or not seen.get('orientation_confirmed')
                or p.get('last_box_step', -100) < p['steps']-1
                or seen.get('tilt_degrees', 90.) > assist.config['level_tolerance_degrees']):
            return
        normal = np.asarray(normal, float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or normal[2] <= 0:
            return
        normal /= max(np.linalg.norm(normal), 1e-9)
        geometry = assist.geometry[1]
        if geometry is None:
            return
        offset = np.asarray(geometry['extent'])[2]*normal[:2]
        if self.axis is not None:
            offset[1-self.axis] = 0.
        offset *= min(1., self.max_offset_m/max(np.linalg.norm(offset), 1e-9))
        original = np.asarray(assist.weighing_target, float).copy()
        assist.weighing_target = original.copy()
        assist.weighing_target[:2] += offset
        p['stable'] = 0
        self.applied = p
        self.info = dict(original_target_base_m=original.tolist(),
                         target_base_m=assist.weighing_target.tolist(),
                         roof_normal_base=normal.tolist(), offset_xy_m=offset.tolist(),
                         max_offset_m=self.max_offset_m, axis=self.axis)

    def trace(self, assist):
        return self.info if self.applied is not None and assist.pending is self.applied else None
