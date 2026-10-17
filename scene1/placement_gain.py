"""Stage-local damping without changing placement decisions or stored offsets."""
import numpy as np


class PlacementGain:
    def __init__(self, scale=1.):
        if not 0 < scale <= 1:
            raise ValueError('Placement gain scale must be in (0, 1]')
        self.scale = scale
        self.original = None

    def before_step(self, assist):
        pending = assist.pending
        if (self.original is None and self.scale < 1 and pending is not None
                and pending['kind'] == 'weighing' and pending.get('level_ready')):
            self.original = assist.config
            assist.config = dict(self.original, gain=self.original['gain']*self.scale,
                max_joint_step_rad=self.original['max_joint_step_rad']*np.sqrt(self.scale))

    def after_step(self, assist):
        if self.original is not None and (assist.pending is None or assist.pending['kind'] != 'weighing'):
            assist.config = self.original
            self.original = None
