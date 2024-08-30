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
from ego_exo4d_mask import mask_read_ego4d


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
    niter = 1000
    match_prop = []

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    # selected_frames = ['/private/home/wangyu1369/egoexo4d_selected_frames/fair_cooking_06_6/aria01_rgb_000000.jpg',
    #                    '/private/home/wangyu1369/egoexo4d_selected_frames/fair_cooking_06_6/aria01_rgb_000080.jpg']

    # selected_frames = ['/private/home/wangyu1369/egoexo4d_selected_frames/fair_cooking_06_6/aria01_rgb_000000knife_0.png',
    #                    '/private/home/wangyu1369/egoexo4d_selected_frames/fair_cooking_06_6/aria01_rgb_000080knife_0.png']


    # selected_frames = ['/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000808.jpg',
    #                    '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000828.jpg']

    # selected_frames_mask = ['/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000808electric kettle jug_0.png',
    #                     '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000828electric kettle jug_0.png']

    selected_frames_list = [
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000668.jpg',
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000808.jpg',
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000838.jpg',
                            '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000848.jpg',
                            '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000858.jpg',
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000863.jpg',
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000873.jpg',
                            ]

    selected_frames_mask_list = [
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000668electric kettle jug_0.png',
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000808electric kettle jug_0.png',
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000838electric kettle jug_0.png',
                            '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000848electric kettle jug_0.png',
                            '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000858electric kettle jug_0.png',
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000863electric kettle jug_0.png',
                            # '/private/home/wangyu1369/egoexo4d_selected_frames/sfu_cooking_003_5/aria01_rgb_000873electric kettle jug_0.png',
                            ]
    l = len(selected_frames_list)
    
    for i in range(l-1):
        selected_frames = selected_frames_list[i:i+2]

        images = load_images(selected_frames, size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1, verbose=False)

        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()


        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                    device=device, dist='dot', block_size=2**13)

        # ignore small border around the edge
        H0, W0 = view1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = view2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]


        selected_frames_mask = selected_frames_mask_list[i:i+2]
        
        images_mask = load_images(selected_frames_mask, size=512)
        # pairs_mask = make_pairs(images_mask, scene_graph='complete', prefilter=None, symmetrize=True)
        # output_mask = inference(pairs_mask, model, device, batch_size=1, verbose=False)

        output_mask = inference([tuple(images_mask)], model, device, batch_size=1, verbose=False)

        # scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # at this stage, you have the raw dust3r predictions
        view1_mask, pred1_mask = output_mask['view1'], output_mask['pred1']
        view2_mask, pred2_mask = output_mask['view2'], output_mask['pred2']

        desc1_mask, desc2_mask = pred1_mask['desc'].squeeze(0).detach(), pred2_mask['desc'].squeeze(0).detach()

        # visualize a few matches
        import numpy as np
        import torch
        import torchvision.transforms.functional
        from matplotlib import pyplot as pl

        num_matches = matches_im0.shape[0]
        n_viz = 200
        
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            rgb_tensor = view['img'] * image_std + image_mean
            viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)


        viz_imgs_mask = []
        for i, view_mask in enumerate([view1_mask, view2_mask]):
            rgb_tensor_mask = view_mask['img'] * image_std + image_mean
            viz_imgs_mask.append(rgb_tensor_mask.squeeze(0).permute(1, 2, 0).cpu().numpy())

        H0, W0, H1, W1 = *viz_imgs_mask[0].shape[:2], *viz_imgs_mask[1].shape[:2]
        img0_mask = np.pad(viz_imgs_mask[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1_mask = np.pad(viz_imgs_mask[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        

        img = np.concatenate((img0*img0_mask, img1*img1_mask), axis=1)
        # img = np.concatenate((img0, img1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            # if 400 <= x0 <= 500 and 100 <= y0 <= 200:
            if sum(img0_mask[y0][x0])>0:
                pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)
        pl.savefig('kettle_correspondence_original_nomove_mask.png', bbox_inches='tight')

        cur_n, cur_m = 0, 0
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            # if 400 <= x0 <= 500 and 100 <= y0 <= 200:
            if sum(img0_mask[y0][x0])>0:
                cur_n += 1
                if sum(img1_mask[y1][x1])>0:
                    cur_m += 1
        match_prop.append(cur_m/cur_n)


    
    