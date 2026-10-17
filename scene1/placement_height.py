"""Refresh only the observed support height during local weighing.

RgbdScene constructs pad heights from its measured tabletop plane. The pad
footprint can remain cached under the hand, but its old base-frame height
must not be mixed with a newly registered parcel cloud while the body moves.
"""
import copy
from collections import deque

import numpy as np


class PlacementHeight:
    def __init__(self):
        self.pending = None
        self.samples = deque(maxlen=3)
        self.last_stamp = None

    def update(self, assist, snapshot, *, view_ready, fresh):
        pending = assist.pending
        if pending is not self.pending:
            self.pending = pending
            self.samples.clear()
            self.last_stamp = None
            self.reference_z = (float(pending['surface']['position_base_m'][2])
                if pending is not None and pending['kind'] == 'weighing' else None)
        if (pending is None or pending['kind'] != 'weighing'
                or pending.get('stage') != 'correct' or not view_ready or not fresh):
            self.samples.clear()
            return None
        height, stamp = float(snapshot['table_z_base_m']), float(snapshot['stamp'])
        previous = self.last_stamp
        self.last_stamp = stamp
        if (not np.isfinite([height, stamp, self.reference_z]).all() or abs(height-self.reference_z) > .012
                or (previous is not None and stamp <= previous)):
            self.samples.clear()
            return None
        if previous is not None and stamp-previous > .4:
            self.samples.clear()
        self.samples.append(height)
        if len(self.samples) < 3 or np.ptp(self.samples) > .001:
            return None
        height = float(np.median(self.samples))
        old_height = float(pending['surface']['position_base_m'][2])
        surface = copy.deepcopy(pending['surface'])
        surface['position_base_m'][2] = height
        pending['surface'] = surface
        assist.weighing_target = np.asarray(assist.weighing_target).copy()
        assist.weighing_target[2] = height
        return dict(reference_z_m=self.reference_z, previous_z_m=old_height,
                    observed_z_m=height, shift_from_reference_m=height-self.reference_z,
                    samples=3)
