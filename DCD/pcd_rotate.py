import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import copy
import torch
import collections
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation

import os
import numpy as np
from evo.core.units import (Unit, ANGLE_UNITS, LENGTH_UNITS,
                            METER_SCALE_FACTORS)

def rotate_pcd(pcd, angle = 180):

    new_pcd = copy.deepcopy(pcd)
    # Define the rotation angle (in radians)
    angle = np.radians(angle)  # Convert degrees to radians

    # Create a rotation matrix
    R = pcd.get_rotation_matrix_from_xyz((0, 0, angle))  # Rotation around the z-axis

    # Apply the rotation
    rotate_pcd = new_pcd.rotate(R, center=(0, 0, 0))

    # o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/ground_truth_pcds/rotated_point_cloud.ply", rotate_pcd)
    return rotate_pcd


import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def transform_point_cloud(points, translation, rotation_axis, rotation_angle):
    """
    Transforms a point cloud by applying a translation and a rotation to both positions and quaternions.
    
    Parameters:
        points (np.array): Array of points with shape (N, 7), where each row is (x, y, z, q_x, q_y, q_z, q_w).
        translation (np.array): Translation vector as [tx, ty, tz].
        rotation_axis (np.array): Axis of rotation as [ax, ay, az].
        rotation_angle (float): Rotation angle in degrees.
    
    Returns:
        np.array: Transformed points with the same shape as input.
    """
    # Extract positions and quaternions
    positions = points[:, :3]
    quaternions = points[:, 3:]
    
    # Create rotation object
    rotation = R.from_rotvec(rotation_axis / np.linalg.norm(rotation_axis) * np.radians(rotation_angle))
    
    # Apply translation
    translated_positions = positions + translation
    
    # Apply rotation to positions
    rotated_positions = rotation.apply(translated_positions)
    
    # Apply rotation to quaternions
    rotated_quaternions = rotation * R.from_quat(quaternions)
    
    # Combine transformed positions and quaternions
    transformed_points = np.hstack((rotated_positions, rotated_quaternions.as_quat()))
    
    return transformed_points

from evo.core import lie_algebra as lie

def load_and_transform_point_cloud(pcd, translation, rotation_axis, rotation_angle):
    # Load the point cloud
    points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors)  # Extract color information

    # Augment points with identity quaternions
    identity_quaternions = np.tile([0, 0, 0, 1], (points.shape[0], 1))
    augmented_points = np.hstack((points, identity_quaternions))
    
    # Transform the point cloud
    transformed_points = transform_point_cloud(augmented_points, translation, rotation_axis, rotation_angle)
    
    # Create a new Open3D point cloud object for the transformed points
    transformed_pcd = o3d.geometry.PointCloud()
    transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points[:, :3])
    transformed_pcd.colors = o3d.utility.Vector3dVector(original_colors)  # Set color information
    
    o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/ground_truth_pcds/rotated_point_cloud_{}.ply".format(translation), transformed_pcd)

    file_name_gt = '/private/home/wangyu1369/DROID-SLAM/droid_slam/rotation/gt_est_poses/gt.txt'
    # print(traj)
    with open(file_name_gt, 'w') as file:
        for index, pose in enumerate(augmented_points[:10, :]):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')

    
    file_name_rotate = '/private/home/wangyu1369/DROID-SLAM/droid_slam/rotation/gt_est_poses/rotate.txt'
    # print(traj)
    with open(file_name_rotate, 'w') as file:
        for index, pose in enumerate(transformed_points[:10, :]):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')

    
    traj_ref = file_interface.read_tum_trajectory_file(file_name_gt)
    traj_est = file_interface.read_tum_trajectory_file( file_name_rotate)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    ape = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                                               pose_relation=PoseRelation.full_transformation, align=False, correct_scale=False).stats['mean']

    # print(ape)
    return transformed_pcd, ape

def calculate_chamfer_distance(new_pcd, gt_pcd):

    source_points = np.asarray(new_pcd.points)
    target_points = np.asarray(gt_pcd.points)

    tree = KDTree(source_points)
    dist_target_points = tree.query(target_points)[0]
    tree = KDTree(target_points)
    dist_source_points = tree.query(source_points)[0]

    chamfer_dist =  np.mean(dist_target_points) + np.mean(dist_source_points)
    
    # print(dist_target_points)
    return chamfer_dist/2

def calc_nn(new_pcd, gt_pcd):

    source_points = np.asarray(new_pcd.points)
    target_points = np.asarray(gt_pcd.points)

    tree = KDTree(source_points)
    dist_1, index1 = tree.query(target_points)
    tree = KDTree(target_points)

    dist_2, index2 = tree.query(source_points)

    # print(len(dist_1), len(index1), len(dist_2), len(index2))
    return dist_1, dist_2, index1, index2


def calc_dcd(new_pcd, gt_pcd, alpha=1000, n_lambda=1, return_raw=False, non_reg=False):

    source_points = np.asarray(new_pcd.points)
    target_points = np.asarray(gt_pcd.points)


    n_x = source_points.shape[0]
    n_gt = target_points.shape[0]

    if non_reg:
        frac_12 = max(1, n_x / n_gt)
        frac_21 = max(1, n_gt / n_x)
    else:
        frac_12 = n_x / n_gt
        frac_21 = n_gt / n_x

    dist1, dist2, idx1, idx2 = calc_nn(new_pcd, gt_pcd)
    dist1, dist2, idx1, idx2 = torch.from_numpy(dist1), torch.from_numpy(dist2), torch.from_numpy(idx1), torch.from_numpy(idx2)
    # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
    # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
    # dist2 and idx2: vice versa
    exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)


    count1 = torch.zeros_like(idx2)
    count1.scatter_add_(0, idx1.long(), torch.ones_like(idx1))
    weight1 = count1.gather(0, idx1.long()).float().detach() ** n_lambda
    weight1 = (weight1 + 1e-6) ** (-1) * frac_21
    loss1 = (1 - exp_dist1 * weight1).mean(dim=0)
 
    count2 = torch.zeros_like(idx1)
    count2.scatter_add_(0, idx2.long(), torch.ones_like(idx2))
    weight2 = count2.gather(0, idx2.long()).float().detach() ** n_lambda
    weight2 = (weight2 + 1e-6) ** (-1) * frac_12
    loss2 = (1 - exp_dist2 * weight2).mean(dim=0)

    loss = (loss1 + loss2) / 2

    # res = [loss, cd_p, cd_t]
    # if return_raw:
    #     res.extend([dist1, dist2, idx1, idx2])

    return loss

if __name__ == '__main__':


    category_name = 'kite'
    scene_name = '401_52055_102127'
    # category_name = 'ball'
    # scene_name = '113_13350_23632'

    ground_truth_pcd = o3d.io.read_point_cloud('/datasets01/co3dv2/080422/{}/{}/pointcloud.ply'.format(category_name, scene_name))


    # print(transform_point_cloud_and_save(pcd = ground_truth_pcd , translation = [1, 1, 1], axis = [0, 0, 1], angle_degrees= 90)[1])

    # load_and_transform_point_cloud(pcd = ground_truth_pcd, translation = [1, 1, 1], rotation_axis = [0, 0, 1], rotation_angle = 90)

    # import math

    # def rotation_matrix_to_scalar(rotation_matrix):
    #     """
    #     Convert a rotation matrix to a scalar value using SE(3).
    #     Args:
    #         rotation_matrix: 3x3 rotation matrix as a NumPy array
    #     Returns:
    #         scalar value representing the rotation angle in radians
    #     """

    #     R11, R12, R13, R21 = rotation_matrix.flatten()[:4]
    #     theta = math.atan2(math.sqrt(R11**2 + R12**2 + R13**2), R21)
    #     return theta

    # def axis_angle_to_scalar(axis, angle):
    #     """
    #     Convert an axis and angle to a rotation matrix.
    #     Args:
    #         axis: 3-element tuple or list representing the axis of rotation
    #         angle: float representing the angle of rotation in radians
    #     Returns:
    #         3x3 rotation matrix as a NumPy array
    #     """
    #     x, y, z = axis
    #     angle = np.radians(angle)  # Convert degrees to radians
    #     c = math.cos(angle)
    #     s = math.sin(angle)
    #     t = 1 - c
    #     n = math.sqrt(x**2 + y**2 + z**2)
    #     rotation_matrix = np.array([[c + t * x**2 / n**2, t * x * y / n**2 - s * z / n, t * x * z / n**2 + s * y / n],
    #                        [t * x * y / n**2 + s * z / n, c + t * y**2 / n**2, t * y * z / n**2 - s * x / n],
    #                        [t * x * z / n**2 - s * y / n, t * y * z / n**2 + s * x / n, c + t * z**2 / n**2]])
    #     return rotation_matrix_to_scalar(rotation_matrix)


    # for angle in [0, 5, 10, 20]:
    #     print(axis_angle_to_scalar(axis = [1, 0, 0], angle = angle))

    angle_tested = [x for x in range(10)] + [30, 60, 90, 120, 150, 180]
    # trans_test = [x for x in range(5)]

    ground_truth_pcd_0 = ground_truth_pcd.voxel_down_sample(0.01)
    ground_truth_pcd_1 = ground_truth_pcd.voxel_down_sample(0.03)
    ground_truth_pcd_2 = ground_truth_pcd.voxel_down_sample(0.05)


    cd_0 = collections.defaultdict()
    dcd_0 = collections.defaultdict()

    cd_1 = collections.defaultdict()
    dcd_1 = collections.defaultdict()

    cd_2 = collections.defaultdict()
    dcd_2 = collections.defaultdict()

    for angle in angle_tested:
    # for trans in trans_test:

        # print('gt_1: ', np.asarray(ground_truth_pcd.points)[:1])

        new_pcd_0, ape_0 = load_and_transform_point_cloud(pcd = ground_truth_pcd_0, translation = [0, 0, 0], 
                                                    rotation_axis = [0, 0, 1], rotation_angle = angle)
        
        new_pcd_1, ape_1 = load_and_transform_point_cloud(pcd = ground_truth_pcd_1, translation = [0, 0, 0], 
                                                    rotation_axis = [0, 0, 1], rotation_angle = angle)
        
        new_pcd_2, ape_2 = load_and_transform_point_cloud(pcd = ground_truth_pcd_2, translation = [0, 0, 0], 
                                                    rotation_axis = [0, 0, 1], rotation_angle = angle)     
                                                
        # new_pcd = rotate_pcd(ground_truth_pcd, angle = angle)
        # print(new_pcd)

        # print('gt_2: ', np.asarray(ground_truth_pcd.points)[:1])
        # print('rotated: ', np.asarray(new_pcd.points)[:1])

        cd_0[ape_0] = calculate_chamfer_distance(new_pcd_0, ground_truth_pcd_0)
        dcd_0[ape_0] = calc_dcd(new_pcd_0, ground_truth_pcd_0, alpha=1).item()

        cd_1[ape_1] = calculate_chamfer_distance(new_pcd_1, ground_truth_pcd_1)
        dcd_1[ape_1] = calc_dcd(new_pcd_1, ground_truth_pcd_1, alpha=1).item()

        cd_2[ape_2] = calculate_chamfer_distance(new_pcd_2, ground_truth_pcd_2)
        dcd_2[ape_2] = calc_dcd(new_pcd_2, ground_truth_pcd_2, alpha=1).item()
        # cd[np.radians(angle)] = calculate_chamfer_distance(new_pcd, ground_truth_pcd)
        # dcd[np.radians(angle)] = calc_dcd(new_pcd, ground_truth_pcd, alpha=1).item()


    # # cd_json = json.dumps(cd)
    # # dcd_json = json.dumps(dcd)

    # # with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/rotation/cd.json', 'w') as f:
    # #     f.write(cd_json)

    # # with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/rotation/dcd.json', 'w') as f:
    # #     f.write(dcd_json)

    ape_list_0 = []
    cd_list_0 = []
    dcd_list_0 = []

    cd_list_1 = []
    dcd_list_1 = []

    cd_list_2 = []
    dcd_list_2 = []

    for key, value in cd_0.items():
        ape_list_0.append(key)
        cd_list_0.append(value)

    for value in dcd_0.values():
        dcd_list_0.append(value)

    for key, value in cd_1.items():
        cd_list_1.append(value)

    for value in dcd_1.values():
        dcd_list_1.append(value)

    for key, value in cd_2.items():
        cd_list_2.append(value)

    for value in dcd_2.values():
        dcd_list_2.append(value)

    # print('ape: ', ape_list)
    # print('cd: ', cd_list)
    # print('dcd: ', dcd_list)

    plt.plot(ape_list_0, cd_list_0, label = 'CD, sample size = {}'.format(np.asarray(ground_truth_pcd_0.points).shape[0]), 
            c='green', linestyle = '-')
    plt.plot(ape_list_0, dcd_list_0, label = 'DCD, sample size = {}'.format(np.asarray(ground_truth_pcd_0.points).shape[0]), 
            c='blue', linestyle = '-')

    plt.plot(ape_list_0, cd_list_1, label = 'CD, sample size = {}'.format(np.asarray(ground_truth_pcd_1.points).shape[0]), 
            c='green', linestyle = '--')
    plt.plot(ape_list_0, dcd_list_1, label = 'DCD, sample size = {}'.format(np.asarray(ground_truth_pcd_1.points).shape[0]), 
            c='blue', linestyle = '--')

    plt.plot(ape_list_0, cd_list_2, label = 'CD, sample size = {}'.format(np.asarray(ground_truth_pcd_2.points).shape[0]), 
            c='green', linestyle = '-.')
    plt.plot(ape_list_0, dcd_list_2, label = 'DCD, sample size = {}'.format(np.asarray(ground_truth_pcd_2.points).shape[0]), 
            c='blue', linestyle = '-.')
    plt.xlabel('APE')
    plt.ylabel('Loss')
    plt.title('{}'.format(category_name))
    plt.legend()
    plt.savefig('/private/home/wangyu1369/DROID-SLAM/droid_slam/rotation/plots/rotation_{}.png'.format(category_name), bbox_inches='tight')
