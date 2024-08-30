from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import os
import open3d as o3d
import torch
import numpy as np
from get_pose import save_est_traj, get_ground_truth_pose
import json
from scipy.spatial.transform import Rotation as Rot

# from visualization import mask_read
import torch
import cv2
import time
import collections
from trajectory_evaluation import compare_trajectory
from point_clouds_evaluations import calculate_chamfer_distance_new
from select_objects import select_object
import glob
from ego_exo4d_mask import mask_read_ego4d
import shutil
from ego_exo4d_mask import get_aria_frame



# scene = "cmu_bike01_1"
# frame_dir = "/large_experiments/eht/egopose/user/tmp_0_80000_1/cache/{}/"
# images = sorted(glob.glob(os.path.join(frame_dir.format(scene), "hand/halo/images/aria01_rgb_*.jpg")))
# poses = np.array(torch.load("/checkpoint/xiaodongwang/flow/EgoExo4D/existing_frames/{}/true_poses_on_camera.pt".format(scene)))
# # assert len(images) == len(poses)
# print("The size of dataset is", len(images), len(poses))


def poses_to_traj(poses):
    poses = poses.cpu().detach().numpy()
    num_imgs = poses.shape[0]

    traj = []
    for i in range(num_imgs):
        RT = poses[i,:, :]
        R = Rot.from_matrix(RT[:3, :3]).as_quat()
        T = RT[:3, 3]
        traj.append(np.append(T, R))
    
    return traj


def get_img_selected(scene_name, stride=1, half = True):

    frame_dir = "/large_experiments/eht/egopose/user/tmp_0_80000_1/cache/{}/"
    image_list = sorted(glob.glob(os.path.join(frame_dir.format(scene_name), "hand/halo/images/aria01_rgb_*.jpg")))
    l = len(image_list)
    # if half:
    #     image_list = image_list[l//2:]
    
    image_selected = []
    for t in range(0, len(image_list), stride):
        
        image_selected.append(image_list[t])

    image_selected_new = []
    for frame_path in image_selected:

        index = str(int((frame_path.split('/')[-1]).split('.')[-2][-6:]))

        if 4600 <= int(index) <= 4700:
            dst_path = os.path.join(parent_dir, frame_path.split('/')[-1])
            image_selected_new.append(dst_path)

    return image_selected_new


def get_ground_truth_pose_ego4d(scene_name, stride= 1, half= True):

    all_poses = np.array(torch.load("/checkpoint/xiaodongwang/flow/EgoExo4D/existing_frames/{}/true_poses_on_camera.pt".format(scene_name)))
    
    file_path = '/private/home/wangyu1369/dust3r/ego_exo_4d/poses_gt_est/{}_gt.txt'.format(scene_name)

    l = all_poses.shape[0]
    if half:
        all_poses = all_poses[l//2:, :]
    with open(file_path, 'w') as file:
        cur_index = 0
        for index in range(0, len(all_poses), stride):
            line = str(cur_index) + ' '+  ' '.join(map(str, all_poses[index, :]))
            file.write(line + '\n')
            cur_index += 1

def save_est_traj(poses, scene_name, stride):
    traj = poses_to_traj(poses)
    file_name = '/private/home/wangyu1369/dust3r/ego_exo_4d/poses_gt_est/{}_est.txt'.format(scene_name)
    with open(file_name, 'w') as file:
        for index, pose in enumerate(traj):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')

def normalize_point_cloud(pcd):
    # Compute the centroid of the point cloud
    centroid = pcd.get_center()
    # Translate the centroid to the origin
    pcd.translate(-centroid)
    # Scale the point cloud to fit within a unit sphere
    distances = np.linalg.norm(np.asarray(pcd.points), axis=1)
    max_distance = np.max(distances)
    pcd.scale(10 / max_distance, center=(0, 0, 0))
    return pcd

def standardize_point_cloud(pcd):
    # Compute the mean and standard deviation of the point cloud
    mean = pcd.get_center()
    std_dev = np.std(np.asarray(pcd.points), axis=0)
    
    # Standardize the point cloud by subtracting the mean and dividing by the standard deviation
    pcd.translate(-mean)
    pcd.scale(1 / std_dev, center=(0, 0, 0))
    
    return pcd


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    stride = 10
    total_run_time = 0 

    zoom_para = 1.55

    # result_mean_std = collections.defaultdict(dict)
    # result = collections.defaultdict(dict)
    
    scene_list = [
        # "cmu_bike01_1",
        # "sfu_cooking015_1",
        # "sfu_cooking_008_3"
        # 'sfu_cooking_003_1',
        # 'sfu_cooking025_2',
        # 'sfu_cooking_013_3',
        # 'unc_basketball_03-16-23_01_31',
        # 'upenn_0726_Duet_Violin_1_1_2',
        # 'upenn_0707_Guitar_2_7',
        # 'sfu_cooking_002_7',
        # 'sfu_cooking_003_5',
        # 'sfu_cooking_004_1',
        # 'sfu_cooking_004_3',
        # 'sfu_cooking_006_1',
        'sfu_cooking_005_6',
        # 'sfu_cooking_008_5',
        # 'sfu_cooking_013_1',
        # 'iiith_cooking_23_1',
        # 'upenn_0711_Cooking_3_3',
        # 'fair_cooking_06_6'
        ]

    #object_name_list = [
                #     'broken egg_0', 
                #    'chopsticks_0', 
                #    'egg carton_0', 
                #    'egg mixture_0', 
                #    'egg_0', 'egg_1', 
                #    'oil spray_0', 
                #    'plastic spoon_0', 
                #    'salt container_0', 
                #    'scrambled egg_0', 
                #    'seasoning blend container_0', 
                #    'skillet_0', 
                #    'spatula_0', 
                #   'white bowl_0']
    # object_name = 'skillet_0'
    # object_name = 'wooden chopping board_0'
    # object_name = 'chopping board_0'

    for scene_name in scene_list:
        # print(scene_name)
        # start_time = time.time()

        # category_name = scene.split("/")[0]
        # scene_name = scene.split("/")[1]
        ## get the selected frames

        masks_path = '/checkpoint/haotang/data/egoexo/interpolated_mask/{}.json'.format(scene_name)

        with open(masks_path, "r") as f:
            masks = json.load(f)

        all_objects = masks.keys()
        print( all_objects )

        os.makedirs('/private/home/wangyu1369/egoexo4d_selected_frames/{}'.format(scene_name), exist_ok=True)
        parent_dir = '/private/home/wangyu1369/egoexo4d_selected_frames/{}'.format(scene_name)

        selected_frames = get_img_selected(scene_name = scene_name, stride=stride, half= False)
        print(selected_frames)
    
        for frame_path in selected_frames:

            # image = cv2.imread(frame_path)

            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            index = str(int((frame_path.split('/')[-1]).split('.')[-2][-6:]))

            image = get_aria_frame(scene_name = scene_name, idx = int(index), zoom = zoom_para)

            # mask = mask_read_ego4d(scene_name = scene_name, object = object_name, frame = frame_path, 
            #                     size1 = 512, size2 = 512, zoom = zoom_para)

            # mask = mask.astype(np.float64)

            # result = image.astype(np.float64) * mask[:, :, None]

            # # Convert the result back to uint8
            # result = np.clip(result, 0, 255).astype(np.uint8)

            dst_path = os.path.join(parent_dir, frame_path.split('/')[-1])
            # dst_path += '{}.png'.format(object_name)

            # print(dst_path)
            # Save the processed image
            cv2.imwrite(dst_path, image)


        for object_name in all_objects:
            for frame_path in selected_frames:

                # image = cv2.imread(frame_path)

                # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

                index = str(int((frame_path.split('/')[-1]).split('.')[-2][-6:]))

                image = get_aria_frame(scene_name = scene_name, idx = int(index), zoom = zoom_para)

                mask = mask_read_ego4d(scene_name = scene_name, object = object_name, frame = frame_path, 
                                    size1 = 512, size2 = 512, zoom = zoom_para)

                mask = mask.astype(np.float64)

                result = image.astype(np.float64) * mask[:, :, None]

                # Convert the result back to uint8
                result = np.clip(result, 0, 255).astype(np.uint8)

                dst_path = os.path.join(parent_dir, frame_path.split('/')[-1].split('.')[0])
                dst_path += '{}.png'.format(object_name)

                # print(dst_path)
                # Save the processed image
                cv2.imwrite(dst_path, result)

            

