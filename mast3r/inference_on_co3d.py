from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from get_pose import save_est_traj, get_ground_truth_pose
import collections
import numpy as np

import open3d as o3d
import os
import time
from trajectory_evaluation import compare_trajectory
import cv2
from dust3r.image_pairs import make_pairs


def select_object(category_name, number_of_frames=10):

    scene_list = os.listdir('/datasets01/co3dv2/080422/{}'.format(category_name))[:-5]

    selected = []
    for scene in scene_list:
        cur_path = os.listdir('/datasets01/co3dv2/080422/{}/{}/images/'.format(category_name, scene))
        if len(cur_path)>=202:
            selected.append(scene)
        if len(selected)==number_of_frames:
            break
    
    return selected


def mask_read(mask_file, size1, size2):
    mask = cv2.imread(mask_file)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY )
    mask = mask / 255.0
    h, w = mask.shape[:2]
    s = max(h, w)
    mask_padding = np.zeros((s, s))
    x = (s - w) // 2
    y = (s - h) // 2
    mask_padding[y:y+h, x:x+w] = mask
    mask_padding = cv2.resize(mask_padding, (size1, size2))
    # mask_padding[mask_padding < 0.05] = 0
    # mask_padding[mask_padding >= 0.05] = 1
    return mask_padding >=0.05


def get_img_selected(category_name, scene_name, stride=1):
    co3d_file_path = '/datasets01/co3dv2/080422/{}/{}/images/'.format(category_name, scene_name)
    image_list = os.listdir(co3d_file_path)
    
    image_selected = []
    for t in range(1, len(image_list)+1, stride):
        image_file = os.path.join(co3d_file_path, image_list[t-1])
        image_selected.append(image_file)
    return image_selected


if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    stride = 12

    # for scene in [
    # # "apple/189_20393_38136/", 
    # # "cake/374_42274_84517/",
    # # "pizza/102_11950_20611/"
    # "kite/401_52055_102127/",
    # # "book/247_26469_51778/",
    # # "ball/113_13350_23632",
    # # "couch/105_12576_23188",
    # # "suitcase/102_11951_20633"
    # ]:
    result_mean_std = collections.defaultdict(dict)
    result = collections.defaultdict(dict)

    for category_name in [
                          "ball", 
                        #   "book", 
                        #   "couch", 
                        #   "kite", 
                        #   "sandwich",
         
                        #   "frisbee", 
                        #   "hotdog", 
                        #   "skateboard", 
                        #   "suitcase"
                          ]:
        cur_error = collections.defaultdict(list)
    
        # scene_list = select_object(category_name = category_name, number_of_frames=1)

        scene_list = ['113_13350_23632']
        # scene_list = ['401_52055_102127']


        for scene_name in scene_list:
            # print(scene_name)
            # start_time = time.time()

            # category_name = scene.split("/")[0]
            # scene_name = scene.split("/")[1]
            ## get the selected frames
            selected_frames = get_img_selected(category_name = category_name, scene_name = scene_name, stride=stride)

            model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            # model_name = '/private/home/wangyu1369/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'

            # you can put the path to a local checkpoint in model_name if needed


            model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

            # model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)


            images = load_images(selected_frames, size=512)
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, model, device, batch_size=1, verbose=False)

            # at this stage, you have the raw dust3r predictions
            # view1, pred1 = output['view1'], output['pred1']
            # view2, pred2 = output['view2'], output['pred2']

            # desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
            loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
            
            # retrieve useful values from scene:
            imgs = scene.imgs
            focals = scene.get_focals()
            poses = scene.get_im_poses()
            pts3d = scene.get_pts3d()
            confidence_masks = scene.get_masks()

            ## save ground truth and estimated trajectories for each category and scene for evaluation
            get_ground_truth_pose(category_name, scene_name, stride= stride)
            save_est_traj(poses=poses, category_name=category_name, scene_name=scene_name)

            cur_ape_trans, cur_ape_rotat, cur_rpe_trans, cur_rpe_rotat = compare_trajectory(category_name, scene_name)

            cur_error['ape_translation_list'].append(round(cur_ape_trans, 4))
            cur_error['ape_rotation_list'].append(round(cur_ape_rotat, 4))
            cur_error['rpe_translation_list'].append(round(cur_rpe_trans, 4))
            cur_error['rpe_rotation_list'].append(round(cur_rpe_rotat, 4))

            # Create an Open3D point cloud object
            pts3d_list_wobg = []
            colors_list_wobg = []
            pts3d_list_bg = []
            colors_list_bg = []
            for i in range(len(pts3d)):
                ## read mask information 
                mask_path = "/datasets01/co3dv2/080422/{}/{}/masks/{}.png".format(category_name, scene_name, selected_frames[i].split("/")[-1][:-4])
                conf_i = confidence_masks[i].cpu().numpy()
                mask_img = mask_read(mask_path, conf_i.shape[1], conf_i.shape[0])
                # print(mask_img.shape, conf_i.shape)
                mask_i = conf_i&mask_img
                # mask_i = conf_i
                pts3d_list_wobg.append(pts3d[i].detach().cpu().numpy()[mask_i])
                colors_list_wobg.append(imgs[i][mask_i]) 

                pts3d_list_bg.append(pts3d[i].detach().cpu().numpy()[conf_i])
                colors_list_bg.append(imgs[i][conf_i]) 


            pts3d_merge_wobg = np.concatenate(pts3d_list_wobg, axis=0)
            colors_merge_wobg = np.concatenate(colors_list_wobg, axis=0)

            pts3d_merge_wobg = pts3d_merge_wobg.reshape(-1, 3)
            colors_merge_wobg = colors_merge_wobg.reshape(-1, 3)

            # print('3d points shape: ', pts3d_merge_wobg.shape)
            # print('colors points shape: ', colors_merge.shape)
            
            point_cloud_wobg = o3d.geometry.PointCloud()
            point_cloud_wobg.points = o3d.utility.Vector3dVector(pts3d_merge_wobg)
            point_cloud_wobg.colors = o3d.utility.Vector3dVector(colors_merge_wobg)

            o3d.io.write_point_cloud("/private/home/wangyu1369/mast3r/estimated_point_clouds/dust3r_pcd_{}_{}.ply".format(category_name, scene_name), point_cloud_wobg)

        print(cur_error)