"""Bounded weighing waypoints from observed support and robot feedback."""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def support_margin(center, rotation, extent, footprint):
    corners = np.asarray(center)+rotation.apply([[x,y,z] for x in (-extent[0]/2,extent[0]/2)
        for y in (-extent[1]/2,extent[1]/2) for z in (-extent[2]/2,extent[2]/2)])
    hull = np.asarray(footprint, np.float32)
    return min(cv2.pointPolygonTest(hull, tuple(map(float,p[:2])), True) for p in corners)


def supported_candidate(candidates):
    """Prefer support among already-qualified, similarly reachable waypoints.

    A tilted roof can move horizontally as the parcel rests upright. Do not
    spend substantial support margin on a negligible IK residual improvement.
    All original feasibility/support filters run before this ranking.
    """
    if not candidates:
        return None
    error = min(c['predicted_error'] for c in candidates)
    near = [c for c in candidates if c['predicted_error'] <= error+max(.1*error, 1e-8)]
    margin = max(c['support_margin_m'] for c in near)
    # Millimetric boundary noise should not choose a worse pose when the
    # available support is effectively the same.
    return min((c for c in near if c['support_margin_m'] >= margin-.002),
               key=lambda c: c['predicted_error'])


def regrasp_posture_reserve(kinematics, measured, command, goal):
    """A bounded posture preference at the SAME Cartesian weighing goal.

    This is only a local model. The caller must physically servo to it before
    the original weighing controller captures its unchanged reference/budget.
    """
    measured, command = np.asarray(measured), np.asarray(command)
    bias = measured[8:15]-command[8:15]
    limits = kinematics.limits[1]
    low = np.maximum(limits[:, 0], limits[:, 0]-bias)
    high = np.minimum(limits[:, 1], limits[:, 1]-bias)
    if np.any(high-low < 1e-5):
        return None
    rotation = Rotation.from_matrix(goal[:3, :3])
    preferred = limits[4, 1]-np.deg2rad(25)
    def task_error(q):
        pose = kinematics.matrix(q+bias, 1)
        return np.r_[pose[:3, 3]-goal[:3, 3], .12*(rotation*
            Rotation.from_matrix(pose[:3, :3]).inv()).as_rotvec()]
    fit = least_squares(lambda q: np.r_[task_error(q), .06*(q[4]-preferred)],
        np.clip(command[8:15], low+1e-8, high-1e-8), bounds=(low, high), max_nfev=200)
    def qualified(solution):
        error = task_error(solution.x)
        return (solution.success and np.linalg.norm(error[:3]) <= .008
                and np.linalg.norm(error[3:])/.12 <= np.deg2rad(9.3)
                and limits[4, 1]-solution.x[4] >= np.deg2rad(15))
    refined = False
    if not qualified(fit):
        # The soft 25-degree preference can sacrifice a feasible task pose.
        # Refine the SAME goal with the existing minimum reserve as a bound;
        # a weak movement cost resolves the remaining arm redundancy.
        reserve_high = high.copy()
        reserve_high[4] = min(reserve_high[4], limits[4, 1]-np.deg2rad(15))
        if np.any(reserve_high-low < 1e-5):
            return None
        fit = least_squares(lambda q: np.r_[task_error(q), .001*(q-command[8:15])],
            np.clip(fit.x, low+1e-8, reserve_high-1e-8),
            bounds=(low, reserve_high), max_nfev=200)
        refined = True
    error = task_error(fit.x)
    position = float(np.linalg.norm(error[:3]))
    angle = float(np.linalg.norm(error[3:])/.12)
    reserve = float(limits[4, 1]-fit.x[4])
    # Leave prediction margin below the original actual 10-degree gate.
    # Do not trade an already valid weighing pose for a marginal model fit.
    if not fit.success or position > .008 or angle > np.deg2rad(9.3) or reserve < np.deg2rad(15):
        return None
    return dict(reference=fit.x, predicted_position_m=position,
                predicted_angle_degrees=float(np.rad2deg(angle)),
                wrist_reserve_degrees=float(np.rad2deg(reserve)),
                same_goal_refined=refined)


def alternate_weighing_pose(kinematics, measured, command, geometry, goal, surface):
    """Select a nearby supported Cartesian goal, never an executable joint action.

    Current tracking bias is only a local feasibility estimate. The caller must
    still servo with measured feedback and verify the actual parcel pose before
    handing over to the original placement controller.
    """
    extent = np.asarray(geometry['extent'])
    relative = geometry['rotation']
    center = goal[:3,3]+Rotation.from_matrix(goal[:3,:3]).apply(geometry['center'])
    box = Rotation.from_matrix(goal[:3,:3])*relative
    yaw = np.arctan2(box.as_matrix()[1,0],box.as_matrix()[0,0])
    heading = np.round(yaw/(np.pi/2))*(np.pi/2)
    q = np.asarray(measured)[8:15]
    bias = q-np.asarray(command)[8:15]
    limits = kinematics.limits[1]
    low = np.maximum(limits[:,0], limits[:,0]+bias)
    high = np.minimum(limits[:,1], limits[:,1]+bias)
    if np.any(high-low < 1e-5):
        return None
    seed = np.clip(q, low+1e-6, high-1e-6)
    def fit(target, rotation):
        def objective(joints):
            pose = kinematics.matrix(joints, 1)
            return np.r_[pose[:3,3]-target, .12*(rotation*
                Rotation.from_matrix(pose[:3,:3]).inv()).as_rotvec()]
        solution = least_squares(objective, seed, bounds=(low,high), max_nfev=70)
        error = objective(solution.x)
        return float(np.linalg.norm(error)), float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])/.12)
    baseline = fit(goal[:3,3], Rotation.from_matrix(goal[:3,:3]))[0]
    candidates = []
    footprint = surface['footprint_xy_m']
    for dx,dy in ((0.,0.),(-.01,.03),(0.,.03),(.01,.03),(-.01,0.),(.01,0.),(-.01,-.03),(0.,-.03),(.01,-.03)):
        destination = center+np.array([dx,dy,0.])
        for offset in (-10.,0.,10.):
            candidate_yaw = float(np.clip(yaw+np.deg2rad(offset),
                heading-np.deg2rad(10.), heading+np.deg2rad(10.)))
            box_rotation = Rotation.from_euler('z', candidate_yaw)
            # Stay within ten degrees of the existing goal and the established
            # heading range; no large rotation above the weighing pad.
            if (box_rotation*box.inv()).magnitude() > np.deg2rad(10.01):
                continue
            margin = support_margin(destination,box_rotation,extent,footprint)
            if margin < .025:
                continue
            tool_rotation = box_rotation*relative.inv()
            xyz = destination-tool_rotation.apply(geometry['center'])
            score, position, angle = fit(xyz,tool_rotation)
            if position > .01 or score >= baseline*.8:
                continue
            candidates.append(dict(center=destination, rotation=box_rotation, predicted_error=score,
                predicted_position_m=position, predicted_angle_rad=angle,
                baseline_error=baseline, support_margin_m=margin,
                shift_xy_m=[dx,dy], yaw_offset_degrees=float(np.rad2deg(candidate_yaw-yaw))))
    best = supported_candidate(candidates)
    if best is not None:
        minimum = min(candidates, key=lambda c: c['predicted_error'])
        best.update(minimum_predicted_error=minimum['predicted_error'],
                    minimum_error_support_margin_m=minimum['support_margin_m'],
                    support_priority=True)
    return best
