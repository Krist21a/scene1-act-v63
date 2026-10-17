import unittest
import numpy as np
from scene1.settlement import SettlementMonitor, track_resting_roof


class SettlementTests(unittest.TestCase):
    def observation(self, stamp, x=0., tilt=3., supported=True):
        return dict(stamp=stamp, center_xy_m=[x,0.], top_height_m=.1,
                    tilt_degrees=tilt, footprint_margin_m=.025,
                    support_height_ok=supported)

    def test_stable_supported_parcel_requires_elapsed_sensor_time(self):
        monitor = SettlementMonitor()
        for i in range(10):
            self.assertFalse(monitor.update(self.observation(i*.1))['confirmed'])
        self.assertTrue(monitor.update(self.observation(1.))['confirmed'])

    def test_repeated_frame_does_not_confirm_stability(self):
        monitor = SettlementMonitor()
        for _ in range(20):
            self.assertFalse(monitor.update(self.observation(1.))['confirmed'])

    def test_sliding_and_then_settling_uses_recent_window(self):
        monitor = SettlementMonitor()
        for i in range(20):
            self.assertFalse(monitor.update(self.observation(i*.1,i*.002))['confirmed'])
        for i in range(20,34):
            result = monitor.update(self.observation(i*.1,.04))
        self.assertTrue(result['confirmed'])

    def test_occlusion_unsupported_and_excessive_tilt_reset_confirmation(self):
        for bad in (None, self.observation(1.,supported=False), self.observation(1.,tilt=17.)):
            monitor = SettlementMonitor()
            for i in range(10):
                monitor.update(self.observation(i*.1))
            self.assertFalse(monitor.update(bad)['confirmed'])
            self.assertFalse(monitor.update(self.observation(1.1))['confirmed'])

    def test_roof_tracking_excludes_higher_gripper_and_detects_real_translation(self):
        k = np.array([[1000.,0,100],[0,1000.,100],[0,0,1.]])
        target = dict(position_base_m=[0.,0.,1.], top_z_base_m=1.,
                      support_z_base_m=.9, extent_xy_m=[.1,.04], major_axis_xy=[1.,0.])
        def scene(shift=0):
            depth = np.full((200,200),1.1)
            rgb = np.full((200,200,3),[0,150,0],np.uint8)
            depth[80:121,50+shift:151+shift] = 1.
            rgb[80:121,50+shift:151+shift] = [190,180,130]
            # Nearby bright robot part at another height must not move the box centroid.
            depth[50:79,60:110] = 1.03
            rgb[50:79,60:110] = 220
            return dict(depth_m=depth,rgb=rgb,intrinsic=k,camera_to_base=np.eye(4),table_z_base_m=.9)
        still = track_resting_roof(scene(),target,[0.,0.])
        moved = track_resting_roof(scene(20),target,[0.,0.])
        np.testing.assert_allclose(still['center_xy_m'],[0.,0.],atol=.002)
        self.assertGreater(np.linalg.norm(moved['center_xy_m']),.015)
        hidden = scene(); hidden['depth_m'][:,:] = 1.1
        self.assertIsNone(track_resting_roof(hidden,target,[0.,0.]))
