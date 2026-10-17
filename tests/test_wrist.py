import unittest
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation

from scene1.wrist import WristProjector


class IdentityArm:
    tool_xyz = np.zeros(3)

    def matrix(self, q, side):
        matrix = np.eye(4)
        matrix[:3, 3] = q[:3]
        return matrix


class WristProjectionTests(unittest.TestCase):
    def test_actual_visible_offset_uses_near_sides_and_rejects_pad_and_fingers(self):
        projector = WristProjector.__new__(WristProjector)
        projector.kinematics = IdentityArm()
        projector.config = dict(max_correction_m=.012)
        projector.project = lambda side, point, *args, **kwargs: dict(
            uv=np.array([414.,270.]) if point[2] else np.array([424.,380.]),
            depth_m=.18, focal_px=np.array([414.,414.]), rotation=np.eye(3))
        target = dict(position_base_m=[.4,-.55,.02], top_z_base_m=.02)
        path = Path(__file__).parent/'fixtures/regrasp_visible_offset.png'
        rgb = cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        match = projector.measure_outline(1,target,np.zeros(16),rgb)
        self.assertIsNotNone(match)
        self.assertEqual(match['label'],'outline_verified')
        self.assertAlmostEqual(match['observed_uv'][0],376.,delta=3.)
        self.assertLess(match['centering_error_px'],-40.)
        for color in ([255,240,20],[20,20,20]):
            blank = np.full_like(rgb,color)
            self.assertIsNone(projector.measure_outline(1,target,np.zeros(16),blank))

    def test_selected_parcel_is_matched_and_guides_open_finger_midpoint(self):
        projector = WristProjector.__new__(WristProjector)
        projector.kinematics = IdentityArm()
        projector.config = dict(min_area_px=600, max_bbox_px=320,
                                max_center_error_px=25, max_correction_m=.012)
        calibration = dict(local_xyz=np.zeros(3), local_rotation=np.eye(3),
                           intrinsic=np.array([[624,0,640],[0,624,360],[0,0,1.]]),
                           native_wh=np.array([1280.,720.]))
        projector.calibration = [calibration, calibration]
        measured = np.zeros(16)
        predicted = projector.project(1, [.02,-.03,-.3], measured, [848,480])
        np.testing.assert_allclose(predicted['uv'], [451.56,281.6], atol=.1)
        selected = dict(label='yellow', uv=[452.,282.], bbox=[425,250,55,65],
                        area_px=1200, stable=True, clipped=False, candidate_count=1)
        unrelated = dict(selected, label='orange', uv=[600.,282.], bbox=[580,250,55,65])
        match = projector.measure(1,[.02,-.03,-.3],measured,
                                  [selected,unrelated],[848,480])
        self.assertEqual(match['label'],'yellow')
        self.assertGreater(match['correction_base_xy'][0],0)
        self.assertLessEqual(np.linalg.norm(match['correction_base_xy']),.012+1e-9)
        self.assertIsNone(projector.measure(1,[.02,-.03,-.3],measured,
                                            [unrelated],[848,480]))
        top_only = dict(selected, uv=[452.,245.], bbox=[425,230,55,65])
        self.assertIsNone(projector.measure(1,[.02,-.03,-.3],measured,
                                            [top_only],[848,480]))
        self.assertIsNotNone(projector.measure(1,[.02,-.03,-.3],measured,
                                                [top_only],[848,480],
                                                max_center_error_px=45))
        pad_and_parcel = dict(selected, uv=[450.,250.],
                              bbox=[310,225,285,100], area_px=18000)
        self.assertIsNotNone(projector.measure(1,[.02,-.03,-.3],measured,
                                                [pad_and_parcel],[848,480],
                                                max_center_error_px=45))
        # A color match is only a candidate. The regrasp path requires physical
        # edges of the head-camera parcel hypothesis inside that component.
        target = dict(position_base_m=[.02,-.03,-.3], top_z_base_m=-.3,
                      extent_xy_m=[.1,.04], major_axis_xy=[1.,0.])
        pad = np.full((480,848,3), [250,240,30], np.uint8)
        self.assertIsNone(projector.measure_edges(1,target,measured,pad))
        parcel = pad.copy()
        polygon = []
        for a,b in ((-1,-1),(1,-1),(1,1),(-1,1)):
            point = np.array(target['position_base_m'])+[a*.05,b*.02,0]
            polygon.append(projector.project(1,point,measured,[848,480])['uv'])
        cv2.fillConvexPoly(parcel,np.rint(polygon).astype(np.int32),(170,160,120))
        match = projector.measure_edges(1,target,measured,parcel)
        self.assertIsNotNone(match)
        self.assertEqual(match['label'],'geometry_verified')
        self.assertGreaterEqual(sum(s>=.64 for s in match['edge_support']),2)
        outlined = projector.measure_outline(1,target,measured,parcel)
        self.assertIsNotNone(outlined)
        # Bright tape must not split the parcel into separately centered strips.
        taped = parcel.copy()
        cv2.rectangle(taped,(440,260),(465,300),(235,208,100),-1)
        taped_match = projector.measure_outline(1,target,measured,taped)
        self.assertIsNotNone(taped_match)
        self.assertAlmostEqual(taped_match['observed_uv'][0],outlined['observed_uv'][0],delta=2.)
        strip = pad.copy()
        cv2.rectangle(strip,(440,255),(465,310),(170,160,120),-1)
        self.assertIsNone(projector.measure_outline(1,target,measured,strip))
        fingers = pad.copy()
        cv2.fillConvexPoly(fingers,np.rint(polygon).astype(np.int32),(15,15,15))
        self.assertIsNone(projector.measure_edges(1,target,measured,fingers))

    def test_offscreen_target_provides_rotation_toward_wrist_view(self):
        projector = WristProjector.__new__(WristProjector)
        projector.kinematics = IdentityArm()
        projector.calibration = [dict(local_xyz=np.zeros(3),
                                      local_rotation=np.eye(3))]*2
        target = np.array([.3, 0., -.3])
        aim = projector.aim_error(1, target, np.zeros(16))
        self.assertAlmostEqual(aim['angle_degrees'], 45.)
        np.testing.assert_allclose(
            Rotation.from_rotvec(aim['error_base']).apply([0, 0, -1]),
            target/np.linalg.norm(target), atol=1e-8)
