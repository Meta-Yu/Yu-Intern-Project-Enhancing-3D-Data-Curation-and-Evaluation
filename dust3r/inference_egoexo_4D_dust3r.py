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
from ego_exo4d_mask import mask_read_ego4d, get_aria_frame




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


def get_img_selected(scene_name, stride=1, half = True, zoom_para=1.55):

    frame_dir = "/large_experiments/eht/egopose/user/tmp_0_80000_1/cache/{}/"
    image_list = sorted(glob.glob(os.path.join(frame_dir.format(scene_name), "hand/halo/images/aria01_rgb_*.jpg")))
    l = len(image_list)
    # if half:
    #     image_list = image_list[l//2:]
    
    image_selected = []
    for t in range(0, len(image_list), stride):
        image_selected.append(image_list[t])

    os.makedirs('/private/home/wangyu1369/dust3r/Ego_exo4d_raw/{}'.format(scene_name), exist_ok=True)
    parent_dir = '/private/home/wangyu1369/dust3r/Ego_exo4d_raw/{}'.format(scene_name)

    image_selected_new = []
    for frame_path in image_selected:

        index = str(int((frame_path.split('/')[-1]).split('.')[-2][-6:]))

        if 670 <= int(index) <= 880:

            image = get_aria_frame(scene_name = scene_name, idx = int(index), zoom = zoom_para)

            dst_path = os.path.join(parent_dir, frame_path.split('/')[-1])

            # Save the processed image
            cv2.imwrite(dst_path, image)

            image_selected_new.append(dst_path)

    return image_selected_new

# def get_img_selected_with_object(scene_name, object, stride=1):
#     masks_path = '/checkpoint/haotang/data/egoexo/interpolated_mask/{}.json'.format(scene_name)
#     with open(masks_path, "r") as f:
#         cur_masks = json.load(f)

#     all_frames_list = set(cur_masks[object]['aria01_214-1']["annotation"].keys())

#     frame_dir = "/large_experiments/eht/egopose/user/tmp_0_80000_1/cache/{}/"
#     image_list = sorted(glob.glob(os.path.join(frame_dir.format(scene_name), "hand/halo/images/aria01_rgb_*.jpg")))
#     l = len(image_list)
    
#     image_selected = []
#     for t in range(0, len(image_list), stride):
#         if str(int((image_list[t].split('/')[-1]).split('.')[-2][-6:])) in all_frames_list:
#             image_selected.append(image_list[t])
    
#     return image_selected


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
    if np.asarray(pcd.points).shape[0]==0:
        return pcd
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

# from PIL import Image


if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    stride = 5
    total_run_time = 0 

    # result_mean_std = collections.defaultdict(dict)
    # result = collections.defaultdict(dict)

    
    scene_list = [
        # "cmu_bike01_1",
        # "sfu_cooking015_1"
        # "sfu_cooking_008_3"
        # 'sfu_cooking_003_1'
        # 'uniandes_cooking_001_3'
        # 'sfu_cooking025_2'
        # 'unc_basketball_03-16-23_01_31'
        'sfu_cooking_003_5'
        # 'fair_cooking_06_6'
        # 'sfu_cooking_005_6',
        ]

    # cmu_bike01_1

    # object_name = 'wooden chopping board_0'
    # object_name = 'skillet_0'
    # object_name = 'brown ceramic bowl_0'

    for scene_name in scene_list:
        # print(scene_name)
        start_time = time.time()

        masks_path = '/checkpoint/haotang/data/egoexo/interpolated_mask/{}.json'.format(scene_name)

        with open(masks_path, "r") as f:
            masks = json.load(f)

        # all_objects = masks.keys()

        all_objects = [
                        # 'chopped spring onions_0', 
                    #    'knife_0', 
                    #    'oil bottle_0', 
                    #    'stainless frying pan_0', 
                    #    'stainless pot_0'
                       ]

        # all_objects = ['electric kettle jug_0', 'electric kettle_0']
        # all_objects = ['electric kettle jug_0']
        # print(all_objects)
        # all_objects = ['black pepper container_0']

        # category_name = scene.split("/")[0]
        # scene_name = scene.split("/")[1]
        ## get the selected frames
        selected_frames = get_img_selected(scene_name = scene_name, stride=stride, half= False)
        # selected_frames = get_img_selected_with_object(scene_name = scene_name, object = 'wooden chopping board_0', stride=stride)

        # print(selected_frames)

        # # # print(selected_frames[:2])
        model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        # you can put the path to a local checkpoint in model_name if needed
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
        # load_images can take a list of images or a directory

        images = load_images(selected_frames, size=512, rotate = False)
        # images = load_images(selected_frames, size=200)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)
        

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']


        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # retrieve useful values from scene:
        imgs = scene.imgs
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()


        ## save ground truth and estimated trajectories for each category and scene for evaluation
        # get_ground_truth_pose_ego4d(scene_name, stride= stride, half= False)

        # save_est_traj(poses=poses, scene_name=scene_name, stride = stride)

        # cur_ape_trans, cur_ape_rotat, cur_rpe_trans, cur_rpe_rotat = compare_trajectory(category_name, scene_name)

        # cur_error['ape_translation_list'].append(round(cur_ape_trans, 4))
        # cur_error['ape_rotation_list'].append(round(cur_ape_rotat, 4))
        # cur_error['rpe_translation_list'].append(round(cur_rpe_trans, 4))
        # cur_error['rpe_rotation_list'].append(round(cur_rpe_rotat, 4))


        
        # Create an Open3D point cloud object

        # for object_name in all_objects:
        #     pts3d_list_wobg = []
        #     colors_list_wobg = []
        #     pts3d_list_bg = []
        #     colors_list_bg = []
        #     for i in range(len(pts3d)):
        #         ## read mask information 
        #         conf_i = confidence_masks[i].cpu().numpy()

        #         mask_img = mask_read_ego4d(scene_name, object = object_name, frame = selected_frames[i], size1 = conf_i.shape[1], size2 =conf_i.shape[0])
        #         # print(mask_img.shape, conf_i.shape)

        #         mask_i = conf_i&mask_img

        #         pts3d_list_bg.append(pts3d[i].detach().cpu().numpy()[mask_i])

        #         colors_list_bg.append(imgs[i][mask_i])

        #         # print(imgs[i][mask_i].shape)


        #     pts3d_merge_bg = np.concatenate(pts3d_list_bg, axis=0)
        #     colors_merge_bg = np.concatenate(colors_list_bg, axis=0)

        #     pts3d_merge_bg = pts3d_merge_bg.reshape(-1, 3)
        #     colors_merge_bg = colors_merge_bg.reshape(-1, 3)

        #     print('3d points shape: ', pts3d_merge_bg.shape)
        #     print('colors points shape: ', colors_merge_bg.shape)
            
        #     point_cloud_bg = o3d.geometry.PointCloud()
        #     point_cloud_bg.points = o3d.utility.Vector3dVector(pts3d_merge_bg)
        #     point_cloud_bg.colors = o3d.utility.Vector3dVector(colors_merge_bg)


        #     o3d.io.write_point_cloud("/private/home/wangyu1369/dust3r/ego_exo_4d/est_pcd/dust3r_pcd_{}_{}.ply".format(scene_name, object_name), normalize_point_cloud(point_cloud_bg))


        pts3d_list_wobg = []
        colors_list_wobg = []
        pts3d_list_bg = []
        colors_list_bg = []
        for i in range(len(pts3d)):
            ## read mask information 
            conf_i = confidence_masks[i].cpu().numpy()

            mask_i = conf_i

            pts3d_list_bg.append(pts3d[i].detach().cpu().numpy()[mask_i])

            colors_list_bg.append(imgs[i][mask_i])

            # print(imgs[i][mask_i].shape)


        pts3d_merge_bg = np.concatenate(pts3d_list_bg, axis=0)
        colors_merge_bg = np.concatenate(colors_list_bg, axis=0)

        pts3d_merge_bg = pts3d_merge_bg.reshape(-1, 3)
        colors_merge_bg = colors_merge_bg.reshape(-1, 3)

        print('3d points shape: ', pts3d_merge_bg.shape)
        print('colors points shape: ', colors_merge_bg.shape)
        
        point_cloud_bg = o3d.geometry.PointCloud()
        point_cloud_bg.points = o3d.utility.Vector3dVector(pts3d_merge_bg)
        point_cloud_bg.colors = o3d.utility.Vector3dVector(colors_merge_bg)


        o3d.io.write_point_cloud("/private/home/wangyu1369/dust3r/ego_exo_4d/est_pcd/dust3r_pcd_sccen_{}.ply".format(scene_name), normalize_point_cloud(point_cloud_bg))



        # # # cur_chamfer_distance = calculate_chamfer_distance_new(category_name, scene_name)
        # # # cur_error['chamfer_distance_list'].append(round(cur_chamfer_distance, 4))
        
        # end_time = time.time()
        # # Calculate total time taken
        # total_time = end_time - start_time

        # print("time: ", total_time)

    #     cur_error['time_list'].append(total_time)

    # for key in cur_error.keys():
    #     result[category_name] = cur_error
    #     result_mean_std[category_name][key] = [np.mean(cur_error[key]), np.std(cur_error[key])]


    # result_string = json.dumps(result)
    # result_mean_std_string = json.dumps(result_mean_std)

    # if (202//stride+1)>=20:

    #     with open('/private/home/wangyu1369/dust3r/errors/result_dense.json', 'w') as f:
    #         f.write(result_string)

    #     with open('/private/home/wangyu1369/dust3r/errors/result_mean_std_string_dense.json', 'w') as f:
    #         f.write(result_mean_std_string)
    # elif 10<=(202//stride+1)<20:

    #     with open('/private/home/wangyu1369/dust3r/errors/result_mediate.json', 'w') as f:
    #         f.write(result_string)

    #     with open('/private/home/wangyu1369/dust3r/errors/result_mean_std_string_mediate.json', 'w') as f:
    #         f.write(result_mean_std_string)
    # else:
    #     with open('/private/home/wangyu1369/dust3r/errors/result_sparse.json', 'w') as f:
    #         f.write(result_string)

    #     with open('/private/home/wangyu1369/dust3r/errors/result_mean_std_string_sparse.json', 'w') as f:
    #         f.write(result_mean_std_string)

    #     print("time for {}".format(category_name), total_time)


    # total_run_time += total_time

    # print("total_run_time:", total_run_time)

    # torch.save(pts3d, "/private/home/wangyu1369/dust3r/estimated_point_clouds/pcd_{}_{}.pt".format('apple', '189_20393_38136'))
    # print(poses)
    # print(pts3d.shape)

# print(result_mean_std)


