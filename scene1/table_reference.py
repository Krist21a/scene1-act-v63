"""Small rigid reference changes measured from four visible tabletop edges."""
import copy
import cv2
import numpy as np


def table_boundary(snapshot):
    """Actual tabletop-to-lower-surface RGB-D edges, before hull simplification.

    The image border and higher occluding hands provide no exterior edge
    evidence. Hull vertices alone may discard long, nearly collinear edges.
    """
    depth = snapshot['depth_m']
    intrinsic, matrix = snapshot['intrinsic'], snapshot['camera_to_base']
    v, u = np.indices(depth.shape)
    x = (u-intrinsic[0, 2])/intrinsic[0, 0]
    y = (v-intrinsic[1, 2])/intrinsic[1, 1]
    z = (matrix[2, 0]*x+matrix[2, 1]*y+matrix[2, 2])*depth+matrix[2, 3]
    plane = snapshot['table_z_base_m']
    hsv = cv2.cvtColor(snapshot['rgb'], cv2.COLOR_RGB2HSV)
    valid = (depth > .1) & np.isfinite(depth) & np.isfinite(z)
    top = (valid & (hsv[:, :, 0] >= 40) & (hsv[:, :, 0] <= 85)
           & (hsv[:, :, 1] >= 100) & (abs(z-plane) < .012))
    lower = valid & (z < plane-.015)
    boundary = np.zeros(depth.shape, bool)
    for axis in (0, 1):
        for shift in (-2, 2):
            boundary |= top & np.roll(lower, shift, axis=axis)
    boundary[:2] = boundary[-2:] = False
    boundary[:, :2] = boundary[:, -2:] = False
    camera = np.column_stack((x[boundary]*depth[boundary],
                              y[boundary]*depth[boundary], depth[boundary]))
    return (camera @ matrix[:3, :3].T+matrix[:3, 3])[:, :2]


def table_frame(hull, plane, boundary=None):
    if hull is None:
        return None
    points=np.asarray(hull,np.float32).reshape(-1,2)
    if len(points)<4 or not np.isfinite(points).all() or not np.isfinite(plane):
        return None
    rectangle=cv2.minAreaRect(points);corners=cv2.boxPoints(rectangle)
    edges=np.roll(corners,-1,axis=0)-corners
    lengths=np.linalg.norm(edges,axis=1)
    if min(lengths)<.3 or max(lengths)>2.:
        return None
    if cv2.contourArea(points)<.7*np.prod(rectangle[1]):
        return None
    supporting = points if boundary is None else np.asarray(boundary).reshape(-1, 2)
    if not np.isfinite(supporting).all():
        return None
    support=[]
    for i,edge in enumerate(edges):
        normal=np.array([-edge[1],edge[0]])/lengths[i]
        selected=supporting[abs((supporting-corners[i])@normal)<.008]
        if boundary is not None and len(selected) < 6:
            return None
        span=float(np.ptp(selected@(edge/lengths[i]))) if len(selected)>1 else 0.
        if span<max(.075,.08*lengths[i]):
            return None
        support.append(span)
    index=int(np.argmax(abs(edges[:,0])/lengths))
    xaxis=edges[index]/lengths[index]
    if xaxis[0]<0:xaxis=-xaxis
    return dict(center=np.r_[rectangle[0],plane],
        rotation=np.column_stack((xaxis,[-xaxis[1],xaxis[0]])),
        size=np.array([lengths[index],lengths[(index+1)%4]]),support=support)


def table_motion(reference,current):
    if reference is None or current is None:
        return None
    delta=current['center']-reference['center']
    rotation=current['rotation']@reference['rotation'].T
    angle=float(np.arctan2(rotation[1,0],rotation[0,0]))
    if (np.max(abs(current['size']-reference['size']))>.008
            or np.linalg.norm(delta[:2])>.03 or abs(delta[2])>.015
            or abs(angle)>np.deg2rad(2.)):
        return None
    transform=np.eye(4);transform[:2,:2]=rotation
    transform[:2,3]=current['center'][:2]-rotation@reference['center'][:2]
    transform[2,3]=delta[2]
    return dict(transform=transform,center_shift_m=delta.tolist(),angle_degrees=float(np.rad2deg(angle)))


def move_target(target,transform):
    result=copy.deepcopy(target)
    point=np.r_[target['position_base_m'],1.]
    result['position_base_m']=(transform@point)[:3].tolist()
    result['major_axis_xy']=(transform[:2,:2]@np.asarray(target['major_axis_xy'])).tolist()
    result['top_z_base_m']=target['top_z_base_m']+transform[2,3]
    if 'grasp_goal_xy' in target:
        result['grasp_goal_xy']=(transform[:2,:2]@np.asarray(target['grasp_goal_xy'])+transform[:2,3]).tolist()
    # The table already accounts for common motion. Do not subtract it twice
    # using the legacy parcel references inside the unchanged movement guard.
    result['_reference_centers_xy']=[]
    return result
