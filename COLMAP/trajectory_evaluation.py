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

def compare_trajectory(category_name, scene_name, traj_est = None):
    # if traj_est is None:
    #     file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/est_trajectory/{}_{}.npy'
    #     est_traj_path = file_path.format(category_name, scene_name)
    #     traj_est = np.load(est_traj_path)
    # # tstamp_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/reconstructions_obj/{}_{}_tstamps.npy'
    # # tstamp = np.load(tstamp_path.format(category_name, scene_name))
    # # print(tstamp)

    # tstamps = []
    # image_list = sorted(os.listdir(os.path.join("/datasets01/co3dv2/080422/{}/{}/".format(category_name, scene_name), "images/")))
    # for t in range(1, len(image_list)+1, stride):
    #     tstamps.append(t)
    
    # traj_est = PoseTrajectory3D(positions_xyz= traj_est[:,:3],orientations_quat_wxyz=traj_est[:,3:],timestamps=np.array(tstamps))
    
    gt_file_path = '/private/home/wangyu1369/DROID-SLAM/poses_gt_est/{}_{}_gt.txt'.format(category_name, scene_name)
    est_file_path = '/private/home/wangyu1369/DROID-SLAM/poses_gt_est/{}_{}_est.txt'.format(category_name, scene_name)
    
    traj_ref = file_interface.read_tum_trajectory_file(gt_file_path)
    traj_est = file_interface.read_tum_trajectory_file(est_file_path)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result_translation_part_ape = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                                               pose_relation=PoseRelation.translation_part, align=True, correct_scale=True).stats['mean']
    result_rotation_part_ape = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                                            pose_relation=PoseRelation.rotation_part, align=True, correct_scale=True).stats['mean']

    # print('{} ape:{}'.format(category_name, result_translation_part_ape))
    # print('{} ape:{}'.format(category_name, result_rotation_part_ape))

    result_translation_part_rpe = main_rpe.rpe(traj_ref, traj_est, est_name='traj', delta = 1.0, delta_unit = Unit.frames,
                                               pose_relation=PoseRelation.translation_part, align=True, correct_scale=True).stats['mean']
    result_rotation_part_rpe = main_rpe.rpe(traj_ref, traj_est, est_name='traj', delta = 1.0, delta_unit = Unit.frames,
                                            pose_relation=PoseRelation.rotation_part, align=True, correct_scale=True).stats['mean']

    # print('{} ape:{}'.format(category_name, result_translation_part_rpe))
    # print('{} ape:{}'.format(category_name, result_rotation_part_rpe))
    return result_translation_part_ape, result_rotation_part_ape, result_translation_part_rpe, result_rotation_part_rpe

def compare_trajectory_full(category_name, scene_name, traj_est = None):
    # if traj_est is None:
    #     file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/est_trajectory/{}_{}.npy'
    #     est_traj_path = file_path.format(category_name, scene_name)
    #     traj_est = np.load(est_traj_path)
    # # tstamp_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/reconstructions_obj/{}_{}_tstamps.npy'
    # # tstamp = np.load(tstamp_path.format(category_name, scene_name))
    # # print(tstamp)

    # tstamps = []
    # image_list = sorted(os.listdir(os.path.join("/datasets01/co3dv2/080422/{}/{}/".format(category_name, scene_name), "images/")))
    # for t in range(1, len(image_list)+1, stride):
    #     tstamps.append(t)
    
    # traj_est = PoseTrajectory3D(positions_xyz= traj_est[:,:3],orientations_quat_wxyz=traj_est[:,3:],timestamps=np.array(tstamps))
    
    gt_file_path = '/private/home/wangyu1369/DROID-SLAM/poses_gt_est/{}_{}_gt.txt'.format(category_name, scene_name)
    est_file_path = '/private/home/wangyu1369/DROID-SLAM/poses_gt_est/{}_{}_est.txt'.format(category_name, scene_name)
    
    traj_ref = file_interface.read_tum_trajectory_file(gt_file_path)
    traj_est = file_interface.read_tum_trajectory_file(est_file_path)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    ape = main_ape.ape(traj_ref, traj_est, est_name='traj', 
                                               pose_relation=PoseRelation.full_transformation, align=True, correct_scale=True).stats['mean']

    return ape



if __name__ == '__main__':
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
        compare_trajectory(category_name = cur_category_name, scene_name = cur_scene_name)
