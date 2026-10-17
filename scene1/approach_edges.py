"""Parcel motion from visible roof boundaries, independent of occluded centroids."""
import cv2
import numpy as np


def supported_edge(samples, axis, tangent_extent):
    """Keep one supported straight boundary, separating adjacent corner edges."""
    def evidence(points):
        if len(points) < 6:
            return None
        span = float(np.ptp(points[:, 1-axis]))
        spread = float(np.ptp(np.percentile(points[:, axis], [10, 90])))
        if span < max(.012, .25*tangent_extent) or spread > .009:
            return None
        return dict(position_m=float(np.median(points[:, axis])),
                    points=len(points), span_m=span, normal_spread_m=spread)

    full = evidence(samples)
    if full is not None or len(samples) < 6:
        return full
    # A thick RGB-D boundary at a corner can enter both axis association
    # windows. Separate discontinuous normal-coordinate groups rather than
    # treating a perpendicular edge as scatter of the desired straight edge.
    # No group may supply evidence unless it independently meets every old
    # check and contains a clear majority of the actual boundary samples.
    ordered = samples[np.argsort(samples[:, axis])]
    groups = np.split(ordered, np.flatnonzero(np.diff(ordered[:, axis]) > .004)+1)
    candidates = [evidence(group) for group in groups if len(group) >= .6*len(samples)]
    candidates = [item for item in candidates if item is not None]
    if len(candidates) != 1:
        return None
    result = candidates[0]
    result['discarded_boundary_points'] = len(samples)-result['points']
    return result


def roof_edges(snapshot, target):
    """Measure actual roof-to-lower-surface boundaries in the target frame.

    A higher occluding finger is not an exterior roof edge. Projection provides
    only a bounded association region; every supporting point comes from RGB-D.
    """
    if snapshot is None or 'depth_m' not in snapshot or target is None:
        return None
    major = np.asarray(target.get('major_axis_xy', [1., 0.]), float)
    size = np.asarray(target.get('oriented_extent_m', target['extent_xy_m']), float)
    if not np.isfinite(major).all() or np.linalg.norm(major) < .5 or np.any(size <= 0):
        return None
    major = major/np.linalg.norm(major)
    axes = np.column_stack((major, [-major[1], major[0]]))
    depth, intrinsic = snapshot['depth_m'], snapshot['intrinsic']
    v, u = np.indices(depth.shape)
    camera = np.stack(((u-intrinsic[0, 2])*depth/intrinsic[0, 0],
                       (v-intrinsic[1, 2])*depth/intrinsic[1, 1], depth), axis=-1)
    matrix = snapshot['camera_to_base']
    points = camera @ matrix[:3, :3].T+matrix[:3, 3]
    local = (points[:, :, :2]-np.asarray(target['position_base_m'])[:2]) @ axes
    hsv = cv2.cvtColor(snapshot['rgb'], cv2.COLOR_RGB2HSV)
    valid = (depth > .1) & np.isfinite(points).all(axis=2)
    top = target['top_z_base_m']
    roof = (valid & np.all(abs(local) < size/2+.025, axis=2)
            & (abs(points[:, :, 2]-top) < .010) & (hsv[:, :, 2] >= 90)
            & ~((hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 100)))
    lower = valid & (points[:, :, 2] < top-.015)
    boundary = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        for shift in (-2, 2):
            boundary |= roof & np.roll(lower, shift, axis=axis)
    boundary[:2] = boundary[-2:] = False
    boundary[:, :2] = boundary[:, -2:] = False
    samples = local[boundary]
    result, truncated = {}, []
    for axis in (0, 1):
        for sign in (-1, 1):
            selected = samples[(abs(samples[:, axis]-sign*size[axis]/2) < .018)
                               & (abs(samples[:, 1-axis]) < .35*size[1-axis])]
            edge = supported_edge(selected, axis, size[1-axis])
            if edge is not None:
                result[f'{axis}:{sign}'] = edge
            elif (len(selected) >= 6
                  and np.ptp(np.percentile(selected[:, axis], [10, 90])) <= .009
                  and np.ptp(selected[:, 1-axis]) < max(.012, .25*size[1-axis])):
                truncated.append((axis, sign))
    # Keep every already usable three-edge observation exactly unchanged.
    # When the central crop truncates an otherwise straight visible edge,
    # include its end portion within the measured parcel extent. Every added
    # point is still a real roof-to-lower-depth boundary; all support, corner
    # grouping and later opposing-edge/motion checks remain the same.
    if len(result) < 3:
        for axis, sign in truncated:
            selected = samples[(abs(samples[:, axis]-sign*size[axis]/2) < .018)
                               & (abs(samples[:, 1-axis]) < .5*size[1-axis])]
            edge = supported_edge(selected, axis, size[1-axis])
            if edge is not None:
                edge['extended_tangent_support'] = True
                result[f'{axis}:{sign}'] = edge
    return result


def roof_edge_motion(reference, current):
    """Require three real edges and agreement of any opposing pair."""
    if not reference or not current:
        return None
    common = sorted(set(reference) & set(current))
    if len(common) < 3:
        return None
    shifts = []
    for axis in (0, 1):
        values = [current[key]['position_m']-reference[key]['position_m']
                  for key in common if key.startswith(f'{axis}:')]
        if not values or np.ptp(values) > .004:
            return None
        shifts.append(float(np.median(values)))
    return dict(displacement_m=float(np.linalg.norm(shifts)),
                shift_local_m=shifts, matched_edges=common)
