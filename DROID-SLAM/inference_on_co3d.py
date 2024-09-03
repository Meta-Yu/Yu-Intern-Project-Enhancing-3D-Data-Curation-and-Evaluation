# import sys
# sys.path.append('droid_slam')

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
from get_pose import get_ground_truth_pose, get_est_pose
from visualization_new import generate_point_cloud
from point_clouds_evaluations import calculate_chamfer_distance_new, calc_dcd
from trajectory_evaluation import compare_trajectory, compare_trajectory_full
from select_objects import select_object

import torch.nn.functional as F

def mask_read(mask_file):
    mask = cv2.imread(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY )
    mask = mask / 255.0
    h, w = mask.shape[:2]
    s = max(h, w)
    mask_padding = np.zeros((s, s))
    x = (s - w) // 2
    y = (s - h) // 2
    mask_padding[y:y+h, x:x+w] = mask
    mask_padding = cv2.resize(mask_padding, (400, 400))
    mask_padding[mask_padding < 0.05] = 0
    mask_padding[mask_padding >= 0.05] = 1
    return mask_padding

def image_read(image_file, mask_file=None):
    img = cv2.imread(image_file)
    # if mask_file is not None:
    #     mask = cv2.imread(mask_file)
    #     mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY )
    #     mask_gray = mask_gray / 255.0
    #     mask_gray[mask_gray < 0.05] = 0.0
    #     mask_gray[mask_gray >= 0.05] = 1.0
    #     img_masked = mask_gray[:, :, None] * img
    
    h, w = img.shape[:2]
    s = max(h, w)
    img_padding = np.zeros((s, s, 3))
    x = (s - w) // 2
    y = (s - h) // 2
    img_padding[y:y+h, x:x+w] = img
    img_padding = cv2.resize(img_padding, (400, 400))
    return img_padding

def calib_read(anno_json, scene_id):
    fx_ndc, fy_ndc = anno_json[scene_id]['viewpoint']['focal_length']
    px_ndc, py_ndc = anno_json[scene_id]['viewpoint']['principal_point']
    h, w = anno_json[scene_id]['image']['size']
    s = min(h, w)

    fx_screen = fx_ndc * s / 2.0
    fy_screen = fy_ndc * s / 2.0

    px_screen = w / 2.0 - px_ndc * s / 2.0 
    py_screen = h / 2.0 - py_ndc * s / 2.0
    dx = (max(h, w) - w) // 2   
    dy = (max(h, w) - h) // 2
    px_screen += dx
    py_screen += dy
    intrinsic_padded = np.array([fx_screen, fy_screen, px_screen, py_screen])
    scale_factor = 400 / (max(h, w))
    return intrinsic_padded * scale_factor
    # return intrinsic_padded


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, stride, intrinsics, maskdir = None):
    """ image generator """

    image_list = sorted(os.listdir(os.path.join(imagedir, "images/")))
    if len(image_list)<= (202//stride)+1:
        step_size = 1
    else:
        step_size = stride

    for t in range(1, len(image_list)+1, step_size):

        # update the time stamps list

        # print(os.path.join(imagedir, "{:04d}.png".format(t)))
        # print(os.path.join(imagedir, "images/frame{:06d}.jpg".format(t)))

        ## adjust image_file read
        # image_file = os.path.join(imagedir, "images/frame{:06d}.jpg".format(t))
        
        image_file = os.path.join(imagedir, "images/{}".format(image_list[t-1]))
        img_index = int(image_list[t-1].split('.')[-2][-3:])

        if maskdir is not None:

            mask_file = os.path.join(maskdir, "{}.png".format(image_list[t-1].split('.')[0]))
            # print(image_file, mask_file)
            # mask_file = os.path.join(imagedir, "images/{}".format(image_list[t-1]))

        image = image_read(image_file, "")
        # image = image_read(image_file, "")
        if maskdir is not None:
            mask = mask_read(mask_file)
            # print("image_shape:", image[:,:,0])
            # print("mask_shape:", mask)
            # image = image * mask[..., None]
            for i in range(3):
                image[:,:,i] = image[:,:,i] * mask
        #     dist = np.array([-2.45608996e-01, 7.34894642e-02, -6.97478538e-05, -5.26105036e-04, -1.04689920e-02])
        #     image = cv2.undistort(image, K, dist)
        # print(os.path.join(imagedir, "{}.png".format(t)))
        # if len(calib) > 4:
        #     image = cv2.undistort(image, K, calib[4:])
        # image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1).float()
        # print(image.shape)
        # intrinsics = torch.as_tensor([fx, fy, cx, cy])
        yield img_index, image[None], intrinsics


def save_reconstruction(droid, reconstruction_path, category_name, scene_name):

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
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/{}/{}_{}_tstamps.npy".format(reconstruction_path, category_name, scene_name), tstamps)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/{}/{}_{}_images.npy".format(reconstruction_path, category_name, scene_name), images)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/{}/{}_{}_disps.npy".format(reconstruction_path, category_name, scene_name), disps)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/{}/{}_{}_poses.npy".format(reconstruction_path, category_name, scene_name), poses)
    np.save("/private/home/wangyu1369/DROID-SLAM/droid_slam/{}/{}_{}_intrinsics.npy".format(reconstruction_path, category_name, scene_name), intrinsics)


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

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')


    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True
    imagedir_template = "/datasets01/co3dv2/080422/{}/{}/"

    # video_names = os.listdir("/private/home/xingyuchen/xingyuchen/object_pose/extracted/")
    # video_names = [vname[:-4] for vname in video_names]
    # video_names = ["extracted_GOPR7388"]
    # print(video_names)

    total_run_time = 0 

    result_mean_std = collections.defaultdict(dict)
    result = collections.defaultdict(dict)

    for category_name in [
                          "ball", 
                        #   "book", 
                        #   "couch", 
                          "kite", 
                        #   "sandwich",
         
                        #   "frisbee", 
                        #   "hotdog", 
                        #   "skateboard", 
                        #   "suitcase"
                          ]:
        
        cur_error = collections.defaultdict(list)
    
        scene_list = select_object(category_name, number_of_frames=100)
        # scene_list = ['113_13363_23419']

        for scene_name in scene_list:
            # print(scene_name)
            start_time = time.time()

            f = open(
                "/checkpoint/haotang/data/co3d_v2_annotation/{}/{}.pkl".format(category_name, scene_name),
                'rb'
            )
            anno_json = pickle.load(f)

            # "/datasets01/co3dv2/080422/{}/{}/"
            
            cur_img_list = sorted(os.listdir("/datasets01/co3dv2/080422/{}/{}/images/".format(category_name, scene_name)))

            img_name = cur_img_list[0]
            scene_id = "{}/{}/images/{}".format(category_name, scene_name, img_name)

            # print(anno_json[scene_id])
            # print(scene_id)

            intrinsics = torch.as_tensor(calib_read(anno_json, scene_id))

            # print(intrinsics)
            droid = None
            imagedir = imagedir_template.format(category_name, scene_name)
            # maskdir =  "/datasets01/co3dv2/080422/{}/{}/masks/".format(category_name, scene_name)
            maskdir =  None
            tstamps = []

            
            # if os.path.exists('./co3d_pose/trajectory_est_masked_{}.npy'.format(category_name)):
            #     continue
            # for (t, image, intrinsics) in tqdm(image_stream(imagedir.format(category_name), args.calib, args.stride)):


            try:
                for (t, image, intrinsics) in tqdm(image_stream(imagedir, args.stride, intrinsics, maskdir =  maskdir)):
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
                    save_reconstruction(droid, args.reconstruction_path, category_name, scene_name)

                traj_est = droid.terminate(image_stream(imagedir, args.stride, intrinsics, maskdir =  maskdir))
                np.save('/private/home/wangyu1369/DROID-SLAM/droid_slam/est_trajectory/{}_{}.npy'.format(category_name, scene_name), traj_est)


                # save GT and EST poses as txt files for EVO package
                get_ground_truth_pose(category_name, scene_name, stride= args.stride)
                get_est_pose(category_name, scene_name)

                # cur_ape_trans, cur_ape_rotat, cur_rpe_trans, cur_rpe_rotat = compare_trajectory(category_name, scene_name)

                # cur_error['ape_translation_list'].append(round(cur_ape_trans, 4))
                # cur_error['ape_rotation_list'].append(round(cur_ape_rotat, 4))
                # cur_error['rpe_translation_list'].append(round(cur_rpe_trans, 4))
                # cur_error['rpe_rotation_list'].append(round(cur_rpe_rotat, 4))

                full_ape = compare_trajectory_full(category_name, scene_name)

                cur_error['full_ape'].append(round(full_ape, 4))

                ## Save the estimated point clouds
                number_pcd = generate_point_cloud(droid.video, category_name, scene_name, background=False)
                cur_chamfer_distance = calculate_chamfer_distance_new(category_name, scene_name)
                if not math.isinf(cur_chamfer_distance):
                    cur_error['chamfer_distance_list'].append(round(cur_chamfer_distance, 4))

                
                cur_error['pcd_points_list'].append(number_pcd)

                cur_dcd = calc_dcd(category_name, scene_name, alpha=1)
                cur_error['dcd'].append(round(cur_dcd.item(), 4))

                print("Finished Processing of {}".format(category_name))

                

                end_time = time.time()
                # Calculate total time taken
                total_time = end_time - start_time
                cur_error['time_list'].append(total_time)
                print("time for {}_{}".format(category_name, scene_name), round(total_time, 4))
                # print(cur_error)
                # total_run_time += total_time


            except Exception as e:
                print("Failed Processing of {}".format(category_name))
                print(e)
                raise e
                continue

        for key in cur_error.keys():
            result[category_name] = cur_error
            result_mean_std[category_name][key] = [np.mean(cur_error[key]), np.std(cur_error[key])]


        result_string = json.dumps(result)
        result_mean_std_string = json.dumps(result_mean_std)


        with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_{}frames.json'.format(202//args.stride+1), 'w') as f:
            f.write(result_string)

        with open('/private/home/wangyu1369/DROID-SLAM/droid_slam/errors/result_mean_std_{}frames.json'.format(202//args.stride+1), 'w') as f:
            f.write(result_mean_std_string)


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
