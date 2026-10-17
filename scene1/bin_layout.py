"""Plan distinct drop slots relative to a visually observed bin rim."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def validate_layout(config):
    count = config.get('max_parcels', 1)
    offsets = np.asarray(config.get('bin_lateral_offsets_m', [0.]), float)
    depth = np.asarray(config.get('bin_depth_offsets_m', [0.]*count), float)
    angles = np.asarray(config.get('bin_tool_rpy_degrees', [70., 5., -99.]), float)
    if count not in (1, 2, 3, 4) or offsets.shape != (count,) or not np.isfinite(offsets).all():
        raise ValueError('Visual cycle needs one finite bin offset per parcel')
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError('Bin tool orientation must contain three finite angles')
    if depth.shape != (count,) or not np.isfinite(depth).all():
        raise ValueError('Visual cycle needs one finite bin depth offset per parcel')
    if count > 1 and min(abs(a-b) for i, a in enumerate(offsets) for b in offsets[i+1:]) < .06:
        raise ValueError('Bin slots must be separated by at least 60 mm')
    return offsets, angles


def slot_axis(rim):
    edges = np.roll(rim['corners'], -1, axis=0)-rim['corners']
    edges /= np.linalg.norm(edges, axis=1)[:, None]
    lateral = edges[np.argmax(abs(edges[:, 1]))]
    return -lateral if lateral[1] < 0 else lateral


def slot_points(points, rim, config, index):
    """Exclude neighboring parcels that become visible after hand withdrawal.

    Distinct drop slots own disjoint lateral regions. This association is in
    addition to before/after subtraction, never an object count by itself.
    """
    offsets, _ = validate_layout(config)
    if not 0 <= index < len(offsets):
        raise ValueError('No remaining bin slot')
    if len(offsets) == 1 or not len(points):
        return points
    lateral = (points[:, :2]-(rim['center']+[.025, 0.]))@slot_axis(rim)
    distances = abs(lateral[:, None]-offsets)
    others = np.min(np.delete(distances, index, axis=1), axis=1)
    # Exclude the shared boundary rather than assigning its noise to a slot.
    return points[distances[:, index]+.002 < others]


def plan_bin_drop(rim, geometry, config, index):
    offsets, angles = validate_layout(config)
    if not 0 <= index < len(offsets):
        raise ValueError('No remaining bin slot')
    rotation = Rotation.from_euler('xyz', angles, degrees=True)
    box_rotation = rotation*geometry['rotation']
    extent = np.asarray(geometry['extent'])
    height = abs(box_rotation.as_matrix()[2])@extent
    center = np.r_[rim['center'], rim['rim_z']+.065+height/2]
    center[0] += .025
    lateral = slot_axis(rim)
    center[:2] += offsets[index]*lateral
    # Depth is perpendicular to the slot axis, so a nearer reachable waypoint
    # cannot move a parcel into a neighboring slot's evidence region.
    depth = config.get('bin_depth_offsets_m', [0.]*len(offsets))[index]
    center[:2] += depth*np.array([lateral[1], -lateral[0]])
    corners = center+box_rotation.apply([[a, b, c] for a in (-extent[0]/2, extent[0]/2)
        for b in (-extent[1]/2, extent[1]/2) for c in (-extent[2]/2, extent[2]/2)])
    margin = min(cv2.pointPolygonTest(rim['corners'], tuple(map(float, p[:2])), True) for p in corners)
    if margin < .025:
        raise RuntimeError('Observed parcel does not fit its planned bin slot with 25 mm clearance')
    goal = np.eye(4)
    goal[:3, :3] = rotation.as_matrix()
    goal[:3, 3] = center-rotation.apply(geometry['center'])
    return goal, dict(slot=index, lateral_offset_m=float(offsets[index]),
                      depth_offset_m=float(depth),
                      target_center_base_m=center.tolist(), planned_margin_m=float(margin))
