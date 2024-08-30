import sys
sys.path.append('droid_slam')

import os
import open3d as o3d
import torch
import numpy as np
from scipy.spatial import KDTree
import math
# from pytorch3d.loss.chamfer import chamfer_distance
# from pytorch3d.io import load_ply

def normalize_point_cloud(pcd):
    # Compute the centroid of the point cloud
    centroid = pcd.get_center()
    # Translate the centroid to the origin
    pcd.translate(-centroid)
    # Scale the point cloud to fit within a unit sphere
    distances = np.linalg.norm(np.asarray(pcd.points), axis=1)
    max_distance = np.max(distances)
    pcd.scale(1 / max_distance, center=(0, 0, 0))
    return pcd

def compute_icp(source, target):
    # Set ICP convergence parameters
    threshold = 0.02  # Distance threshold
    trans_init = np.eye(4)  # Initial transformation
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation, reg_p2p

def calculate_chamfer_distance(category_name, scene_name, background = False):
    if not background:
        est_file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_wobg.ply'.format(category_name, scene_name)
    else:
        est_file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_withbg.ply'.format(category_name, scene_name)

    gt_file_path = '/datasets01/co3dv2/080422/{}/{}/pointcloud.ply'.format(category_name, scene_name)

    source = o3d.io.read_point_cloud(est_file_path)
    source = normalize_point_cloud(source)

    target = o3d.io.read_point_cloud(gt_file_path)
    target = normalize_point_cloud(target)

    icp = compute_icp(source, target)
    transformation = icp[0]

    # print(f"ICP Fitness: {icp.fitness:.4f}") 
    # print(f"ICP Inlier RMSE: {icp.inlier_rmse:.4f}")

    source.transform(transformation)


    o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_gt_new.ply".format(category_name, scene_name), 
                             target)
    
    if background:
        o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_withbg_new.ply".format(category_name, scene_name), 
                                source)
    else:
        o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_wobg_new.ply".format(category_name, scene_name), 
                                source)

    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # # Compute minimum distances from source to target
    # dists1 = np.min(np.linalg.norm(source_points[:, None] - target_points, axis=2), axis=1)
    # dists2 = np.min(np.linalg.norm(target_points[:, None] - source_points, axis=2), axis=1)

    # # Chamfer distance is the sum of mean of distances from both directions
    # chamfer_dist = np.mean(dists1) + np.mean(dists2)

    tree = KDTree(source_points)
    dist_target_points = tree.query(target_points)[0]
    tree = KDTree(target_points)
    dist_source_points = tree.query(source_points)[0]

    chamfer_dist =  np.mean(dist_target_points) + np.mean(dist_source_points)

    print("D {}".format(category_name), round(chamfer_dist, 4))
    return chamfer_dist


def calculate_chamfer_distance_new(category_name, scene_name):

    est_file_path = '/private/home/wangyu1369/COLMAP/est_point_clouds/point_cloud_{}_{}.ply'.format(category_name, scene_name)
    if not os.path.exists(est_file_path):
        return math.inf

    gt_file_path = '/datasets01/co3dv2/080422/{}/{}/pointcloud.ply'.format(category_name, scene_name)

    source = o3d.io.read_point_cloud(est_file_path)
    source = normalize_point_cloud(source)

    target = o3d.io.read_point_cloud(gt_file_path)
    target = normalize_point_cloud(target)

    icp = compute_icp(source, target)
    transformation = icp[0]

    source.transform(transformation)

    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)


    tree = KDTree(source_points)
    dist_target_points = tree.query(target_points)[0]
    tree = KDTree(target_points)
    dist_source_points = tree.query(source_points)[0]

    chamfer_dist =  np.mean(dist_target_points) + np.mean(dist_source_points)

    # print("D {}".format(category_name), round(chamfer_dist, 4))
    return chamfer_dist





if __name__ == '__main__':
        cd = []
        for cur_scene in [
        # "apple/189_20393_38136/", 
        # "cake/374_42274_84517/",
        # "pizza/102_11950_20611/"
        "kite/401_52055_102127/",
        "book/247_26469_51778/",
        "ball/113_13350_23632",
        "couch/105_12576_23188",
        "suitcase/102_11951_20633"
        ]:
            cur_category_name = cur_scene.split("/")[0]
            cur_scene_name = cur_scene.split("/")[1]
            # calculate_chamfer_distance(category_name = cur_category_name, scene_name = cur_scene_name, background = True)
            cd.append(calculate_chamfer_distance(category_name = cur_category_name, scene_name = cur_scene_name, background = False))
        print(cd)
        print(np.mean(cd))

