import glob
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import pickle
import random
import collections
import json
import math

from torch.multiprocessing import Process
from droid import Droid
# from data_readers.co3d import CO3D
from visualization_new import generate_point_cloud, generate_point_cloud_ego4d
from point_clouds_evaluations import calculate_chamfer_distance_new
from trajectory_evaluation import compare_trajectory
from select_objects import select_object
from scipy.spatial.transform import Rotation as Rot

import torch.nn.functional as F


def get_ground_truth_pose_ego4d(scene_name, stride= 1, half= False):

    all_poses = np.array(torch.load("/checkpoint/xiaodongwang/flow/EgoExo4D/existing_frames/{}/true_poses_on_camera.pt".format(scene_name)))

    l = all_poses.shape[0]
    if half:
        all_poses = all_poses[l//2:, :]
    
    file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/poses_gt_est/{}_gt.txt'.format(scene_name)
    with open(file_path, 'w') as file:
        cur_index = 0
        for index in range(0, len(all_poses), stride):
            line = str(cur_index) + ' '+  ' '.join(map(str, all_poses[index, :]))
            file.write(line + '\n')
            cur_index += 1

def get_est_pose_ego4d(scene_name):
    file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/est_trajectory/{}.npy'
    est_traj_path = file_path.format(scene_name)
    traj = np.load(est_traj_path)
    # poses = trajectory_to_poses(traj)
    file_name = '/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/poses_gt_est/{}_est.txt'.format(scene_name)
    # print(traj)
    with open(file_name, 'w') as file:
        for index, pose in enumerate(traj):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')
    # return traj

def crop_resize_image(img, new_size=(480, 480), new_focal_length=225):
    # Read the image using cv2
    # img = cv2.imread(image_path)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # Original calibration parameters
    original_principal_point = [255.5, 255.5] 
    original_focal_length = [150, 150]
    
    # Calculate crop box dimensions
    center_x = original_principal_point[0]
    center_y = original_principal_point[1]
    half_width = (new_size[0] / 2) / (new_focal_length / original_focal_length[0])
    half_height = (new_size[1] / 2) / (new_focal_length / original_focal_length[1])
    
    # Define crop box
    left = int(center_x - half_width)
    top = int(center_y - half_height)
    right = int(center_x + half_width)
    bottom = int(center_y + half_height)
    
    # Crop and resize image
    cropped_img = img[top:bottom, left:right]
    resized_img = cv2.resize(cropped_img, new_size, interpolation=cv2.INTER_AREA)

    return resized_img

def image_read(image_file):
    img = cv2.imread(image_file)
    img = crop_resize_image(img)
    return img


def image_stream(imagedir, stride, half= False):
    """ image generator """

    image_list = sorted(glob.glob(os.path.join(imagedir, "hand/halo/images/aria01_rgb_*.jpg")))
    if half:
        image_list = image_list[len(image_list)//2:]

    for t in range(0, len(image_list), stride):
        
        image_file = image_list[t-1]

        image = image_read(image_file)

        image = torch.as_tensor(image).permute(2, 0, 1).float()
        # print(image.shape)
        intrinsics = torch.as_tensor([225, 225, 240, 240])
        yield t, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path, scene_name):

    from pathlib import Path
    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    # Path("reconstructions_obj/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/{}/{}_tstamps.npy".format(reconstruction_path, scene_name), tstamps)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/{}/{}_images.npy".format(reconstruction_path, scene_name), images)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/{}/{}_disps.npy".format(reconstruction_path, scene_name), disps)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/{}/{}_poses.npy".format(reconstruction_path, scene_name), poses)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/{}/{}_intrinsics.npy".format(reconstruction_path, scene_name), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--clip_uid", type=str, help="video clip_uid")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=1, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()


    args.weights = '/private/home/wangyu1369/DROID-SLAM/checkpoints/droid.pth'
    args.stride = 1
    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    # scene = "cmu_bike01_1"
    # frame_dir = "/large_experiments/eht/egopose/user/tmp_0_80000_1/cache/{}/"
    # images = sorted(glob.glob(os.path.join(frame_dir.format(scene), "hand/halo/images/aria01_rgb_*.jpg")))
    # poses = np.array(torch.load("/checkpoint/xiaodongwang/flow/EgoExo4D/existing_frames/{}/true_poses_on_camera.pt".format(scene)))
    # # assert len(images) == len(poses)
    # print("The size of dataset is", len(images), len(poses))


    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True
    imagedir_template = "/large_experiments/eht/egopose/user/tmp_0_80000_1/cache/{}/"

    # video_names = os.listdir("/private/home/xingyuchen/xingyuchen/object_pose/extracted/")
    # video_names = [vname[:-4] for vname in video_names]
    # video_names = ["extracted_GOPR7388"]
    # print(video_names)

    total_run_time = 0 

    # result_mean_std = collections.defaultdict(dict)
    # result = collections.defaultdict(dict)


        
    
    scene_list = [
        #"cmu_bike01_1",
        "sfu_cooking015_1"
                  ]

    for scene_name in scene_list:
        # print(scene_name)
        start_time = time.time()

        # print(anno_json[scene_id])
        # print(scene_id)

        # intrinsics = torch.as_tensor(calib_read(anno_json, scene_id))

        # print(intrinsics)
        droid = None
        # maskdir =  "/datasets01/co3dv2/080422/{}/{}/masks/".format(category_name, scene_name)

        imagedir = imagedir_template.format(scene_name)

        maskdir =  None
        tstamps = []

        
        # if os.path.exists('./co3d_pose/trajectory_est_masked_{}.npy'.format(category_name)):
        #     continue
        # for (t, image, intrinsics) in tqdm(image_stream(imagedir.format(category_name), args.calib, args.stride)):


        try:
            for (t, image, intrinsics) in tqdm(image_stream(imagedir, args.stride, half= True)):
                if t < args.t0:
                    continue
                # print(t)
                # if not args.disable_vis:
                #     show_image(image[0])
                if droid is None:
                    args.image_size = [image.shape[2], image.shape[3]]
                    droid = Droid(args)
                droid.track(t, image, intrinsics=intrinsics)

            if args.reconstruction_path is not None:
                # print(args.reconstruction_path)
                save_reconstruction(droid, 'reconstructions_obj', scene_name)

            traj_est = droid.terminate(image_stream(imagedir, args.stride, half= True))
            np.save('/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/est_trajectory/{}.npy'.format(scene_name), traj_est)


            # save GT and EST poses as txt files for EVO package
            get_ground_truth_pose_ego4d(scene_name = scene_name, stride= args.stride, half= True)
            get_est_pose_ego4d(scene_name = scene_name)

            # cur_ape_trans, cur_ape_rotat, cur_rpe_trans, cur_rpe_rotat = compare_trajectory(category_name, scene_name)

            # cur_error['ape_translation_list'].append(round(cur_ape_trans, 4))
            # cur_error['ape_rotation_list'].append(round(cur_ape_rotat, 4))
            # cur_error['rpe_translation_list'].append(round(cur_rpe_trans, 4))
            # cur_error['rpe_rotation_list'].append(round(cur_rpe_rotat, 4))

            # Save the estimated point clouds
            generate_point_cloud_ego4d(droid.video, scene_name = scene_name, background=False)
            # cur_chamfer_distance = calculate_chamfer_distance_new(category_name, scene_name)
            # if not math.isinf(cur_chamfer_distance):
            #     cur_error['chamfer_distance_list'].append(round(cur_chamfer_distance, 4))

            print("Finished Processing of {}".format(scene_name))

            
            end_time = time.time()
            # Calculate total time taken
            total_time = end_time - start_time
            # cur_error['time_list'].append(total_time)

            print("time for {}".format(scene_name), round(total_time, 4))
            # print(cur_error)
            # total_run_time += total_time


        except Exception as e:
            print("Failed Processing of {}".format(scene_name))
            print(e)
            raise e
            continue

    # for key in cur_error.keys():
    #     result[category_name] = cur_error
    #     result_mean_std[category_name][key] = [np.mean(cur_error[key]), np.std(cur_error[key])]


    # result_string = json.dumps(result)
    # result_mean_std_string = json.dumps(result_mean_std)

    # if (202//args.stride+1)>100:

    #     with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_super_dense.json', 'w') as f:
    #         f.write(result_string)

    #     with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_mean_std_string_super_dense.json', 'w') as f:
    #         f.write(result_mean_std_string)

    # elif 40<=(202//args.stride+1)<=100:

    #     with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_dense.json', 'w') as f:
    #         f.write(result_string)

    #     with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_mean_std_string_dense.json', 'w') as f:
    #         f.write(result_mean_std_string)
    # elif 10<=(202//args.stride+1)<40:

    #     with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_mediate.json', 'w') as f:
    #         f.write(result_string)

    #     with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_mean_std_string_mediate.json', 'w') as f:
    #         f.write(result_mean_std_string)
    # else:
    #     with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_sparse.json', 'w') as f:
    #         f.write(result_string)

    #     with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_mean_std_string_sparse.json', 'w') as f:
    #         f.write(result_mean_std_string)
# print("total_run_time:", total_run_time)


