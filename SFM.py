import numpy as np
from math import sqrt


def calc_TFL_dist(prev_container, curr_container, focal, pp, EM):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp, EM)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif norm_prev_pts.size == 0:
        print('no prev points')
    elif norm_curr_pts.size == 0:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid, curr_container.distances = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)

    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp, EM):
    norm_prev_pts = normalize(prev_container.traffic_lights, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_lights, focal, pp)
    R, foe, tZ = decompose(np.array(EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    dist = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        dist.append(Z)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec, dist


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    return np.array([[pt[0] - pp[0], pt[1] - pp[1]] / focal for pt in pts])


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    return np.array([[pt[0] * focal + pp[0], pt[1] * focal + pp[1]] for pt in pts])


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]
    tX = EM[0, 3]
    tY = EM[1, 3]
    tZ = EM[2, 3]

    if (abs(tZ) > 0):
        foe = np.array([tX, tY]) / tZ
    else:
        foe = []
    return R, foe, tZ


def rotate(pts, R):
    # rotate the points - pts using R
    pts_3d = [R.dot([pt[0], pt[1], 1]) for pt in pts]
    pts_2d = np.array([[pt[0] / pt[2], pt[1] / pt[2]] for pt in pts_3d])  #
    return pts_2d


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index

    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - p[0] * foe[1]) / (foe[0] - p[0])
    # l: y = m * x + n

    min_ditance = abs((m * norm_pts_rot[0, 0] + n - norm_pts_rot[0, 1]) / (sqrt(m * m + 1)))
    closest_pt = norm_pts_rot[0]
    closest_index = 0
    for i, pt_rot in enumerate(norm_pts_rot[1:], 1):
        distance = abs((m * pt_rot[0] + n - pt_rot[1]) / (sqrt(m * m + 1)))
        if min_ditance > distance:
            min_ditance = distance
            closest_pt = pt_rot
            closest_index = i
    return closest_index, closest_pt


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    Z_x = tZ * (foe[0] - p_rot[0]) / (p_curr[0] - p_rot[0])
    Z_y = tZ * (foe[1] - p_rot[1]) / (p_curr[1] - p_rot[1])
    if abs(Z_x - Z_y) <= 5:
        Z = Z_x * 0.5 + Z_y * 0.5
    elif abs(foe[0] - p_rot[0]) > abs(foe[1] - p_rot[1]):
        Z = Z_x
    else:
        Z = Z_y
    return Z
