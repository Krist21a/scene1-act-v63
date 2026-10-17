"""Conservative open-finger/parcel overlap checks from public robot geometry."""
import numpy as np


def boxes_overlap(center_a, rotation_a, half_a, center_b, rotation_b, half_b):
    axes = [*rotation_a.T, *rotation_b.T]
    axes += [np.cross(a,b) for a in rotation_a.T for b in rotation_b.T]
    delta = np.asarray(center_b)-center_a
    for axis in axes:
        if np.linalg.norm(axis) < 1e-8:
            continue
        radius_a = np.abs(axis@rotation_a)@half_a
        radius_b = np.abs(axis@rotation_b)@half_b
        if abs(axis@delta) > radius_a+radius_b:
            return False
    return True


def open_fingers_clear(tool_matrix, target, plane, margin=.003):
    """Check the two open inner fingers against the observed parcel volume.

    Finger bounds are derived from the public 2F85 inner_finger mesh and its
    open URDF chain, relative to the existing 148.8 mm pinch TCP. This checks
    the near-parcel fingers, not whole-robot collision freedom.
    """
    major = np.asarray(target.get('major_axis_xy',[1.,0.]),float)
    major /= np.linalg.norm(major)
    rotation = np.column_stack(([*major,0.],[-major[1],major[0],0.],[0.,0.,1.]))
    center = np.asarray(target['position_base_m'],float).copy()
    height = target['top_z_base_m']-plane
    center[2] = plane+height/2
    half = np.r_[target.get('oriented_extent_m',target['extent_xy_m']),height]/2+margin
    finger_half = np.array([.015852,.011,.028258])
    for sign in (-1,1):
        finger_center = tool_matrix[:3,3]+tool_matrix[:3,:3]@np.array([sign*.056252,0.,.025857])
        if boxes_overlap(finger_center,tool_matrix[:3,:3],finger_half,center,rotation,half):
            return False
    return True
