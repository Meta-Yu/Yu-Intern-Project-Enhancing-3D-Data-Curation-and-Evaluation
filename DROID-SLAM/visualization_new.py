import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops


CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])


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
    mask_padding = cv2.resize(mask_padding, (50, 50))
    # mask_padding[mask_padding < 0.05] = 0
    # mask_padding[mask_padding >= 0.05] = 1
    return mask_padding >=0.05


def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def create_point_actor(points, colors):
    """ open3d point cloud from numpy array """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def generate_point_cloud(video, category_name, scene_name, background=False, downsampling = False, number_points = 2048):

    filter_thresh = 0.005
    pts_result = np.zeros((0, 3))
    clr_result = np.zeros((0, 3))
    with torch.no_grad():
        with video.get_lock():
            t = video.counter.value 
            dirty_index, = torch.where(video.dirty.clone())
            dirty_index = dirty_index

        # print('dirty index:_{}'.format(dirty_index))

        if len(dirty_index) == 0:
            return
        video.dirty[dirty_index] = False
        # convert poses to 4x4 matrix
        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)


        Ps = SE3(poses).inv().matrix().cpu().numpy()

        images = torch.index_select(video.images, 0, dirty_index)
        images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
        points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

        # print('points:{}'.format(points))        

        thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
        
        count = droid_backends.depth_filter(
            video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
        
        # print('count shape:{}'.format(count.shape))

        count = count.cpu()
        disps = disps.cpu()
        masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

        # print('masks:{}'.format(masks))   
        
        for i in range(len(dirty_index)):
            pose = Ps[i]

            ix = dirty_index[i].item()

            mask_ind = int(video.tstamp[i])

            # mask = masks[i].reshape(-1)

            # print(torch.count_nonzero(mask), mask.shape)

            # print("mask    ", mask.shape, torch.max(mask), torch.min(mask))
            # print(mask_ind)
            # mask_path = "/private/home/haotang/dev/Gen6D/data/custom/{}/colmap/masks/{}.jpg".format("pingpongpaddle", mask_ind)
            # print(video.name)
            # mask_path = "/private/home/haotang/dev/Gen6D/data/custom/{}/colmap/masks/{}.jpg".format(video.name, mask_ind)

            mask_path = "/datasets01/co3dv2/080422/{}/{}/masks/frame{:06d}.png".format(category_name, scene_name, mask_ind)
            mask_img = mask_read(mask_path)
            
            if not background:
                # print(mask_img)
                mask = (mask_img & masks[i].cpu().numpy()).reshape(-1)
            else:
                mask = masks[i].reshape(-1)

            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
            # pts = points[i].reshape(-1, 3).cpu().numpy()
            # clr = images[i].reshape(-1, 3).cpu().numpy()
            # print(pts.shape)
            pts_result = np.append(pts_result, pts, axis=0)
            clr_result = np.append(clr_result, clr, axis=0)
            
            # print("length: ", len(pts_result))
        result_point_actor = create_point_actor(pts_result, clr_result)
        # print(pts_result.shape)
        if background:
            o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_withbg.ply".format(category_name, scene_name), 
                                    result_point_actor)
        else:
            o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}.ply".format(category_name, scene_name), 
                                    result_point_actor)
            
        
        return len(pts_result)


def generate_point_cloud_ego4d(video, scene_name, background=False):

    filter_thresh = 0.005
    pts_result = np.zeros((0, 3))
    clr_result = np.zeros((0, 3))
    with torch.no_grad():
        with video.get_lock():
            t = video.counter.value 
            dirty_index, = torch.where(video.dirty.clone())
            dirty_index = dirty_index

        # print('dirty index:_{}'.format(dirty_index))

        if len(dirty_index) == 0:
            return
        video.dirty[dirty_index] = False
        # convert poses to 4x4 matrix
        poses = torch.index_select(video.poses, 0, dirty_index)
        disps = torch.index_select(video.disps, 0, dirty_index)


        Ps = SE3(poses).inv().matrix().cpu().numpy()

        images = torch.index_select(video.images, 0, dirty_index)
        images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
        points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

        # print('points:{}'.format(points))        

        thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
        
        count = droid_backends.depth_filter(
            video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
        
        # print('count shape:{}'.format(count.shape))

        count = count.cpu()
        disps = disps.cpu()
        masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

        # print('masks:{}'.format(masks))   
        
        for i in range(len(dirty_index)):
            pose = Ps[i]

            ix = dirty_index[i].item()

            mask_ind = int(video.tstamp[i])

            mask = masks[i].reshape(-1)

            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
            # pts = points[i].reshape(-1, 3).cpu().numpy()
            # clr = images[i].reshape(-1, 3).cpu().numpy()
            # print(pts.shape)
            pts_result = np.append(pts_result, pts, axis=0)
            clr_result = np.append(clr_result, clr, axis=0)
            
            # print("length: ", len(pts_result))
        result_point_actor = create_point_actor(pts_result, clr_result)
        # print(pts_result.shape)
        if background:
            o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/est_pcd/pcd_{}withbg.ply".format(scene_name), 
                                    result_point_actor)
        else:
            o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/egoexo_4d/est_pcd/pcd_{}.ply".format(scene_name), 
                                    result_point_actor)
            
        
        return len(pts_result)


        

def droid_visualization(video, category_name, scene_name, device="cuda:1"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 0.005
    print("Here!")
    print("*"*32)

    def increase_filter(vis):
        droid_visualization.filter_thresh *= 2
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def decrease_filter(vis):
        droid_visualization.filter_thresh *= 0.5
        with droid_visualization.video.get_lock():
            droid_visualization.video.dirty[:droid_visualization.video.counter.value] = True

    def animation_callback(vis):
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        print("Here!")
        with torch.no_grad():

            with video.get_lock():
                t = video.counter.value 
                dirty_index, = torch.where(video.dirty.clone())
                dirty_index = dirty_index

            if len(dirty_index) == 0:
                return

            video.dirty[dirty_index] = False

            # convert poses to 4x4 matrix
            poses = torch.index_select(video.poses, 0, dirty_index)
            disps = torch.index_select(video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(video.images, 0, dirty_index)
            images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

            thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
            
            count = droid_backends.depth_filter(
                video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))
            print("*" * 32)
            print(len(dirty_index))
            for i in range(len(dirty_index)):
                pose = Ps[i]
                ix = dirty_index[i].item()

                if ix in droid_visualization.cameras:
                    vis.remove_geometry(droid_visualization.cameras[ix])
                    del droid_visualization.cameras[ix]

                if ix in droid_visualization.points:
                    vis.remove_geometry(droid_visualization.points[ix])
                    del droid_visualization.points[ix]

                ### add camera actor ###
                cam_actor = create_camera_actor(True)
                cam_actor.transform(pose)
                # vis.add_geometry(cam_actor)
                droid_visualization.cameras[ix] = cam_actor

                mask = masks[i].reshape(-1)
                pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
                clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
                global pts_result, clr_result
                pts_result = np.append(pts_result, pts, axis=0)
                clr_result = np.append(clr_result, clr, axis=0)
                ## add point actor ###
                point_actor = create_point_actor(pts, clr)
                # vis.add_geometry(point_actor)
                droid_visualization.points[ix] = point_actor

            # hack to allow interacting with vizualization during inference
            if len(droid_visualization.cameras) >= droid_visualization.warmup:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)

            droid_visualization.ix += 1
            result_point_actor = create_point_actor(pts_result, clr_result)
            o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}.ply".format(category_name, scene_name), 
                                 result_point_actor)
            vis.poll_events()
            vis.update_renderer()

    ### create Open3D visualization ###
    print("1")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    print("2")
    vis.register_animation_callback(animation_callback)
    # vis.register_key_callback(ord("S"), increase_filter)
    # vis.register_key_callback(ord("A"), decrease_filter)
    print("3")
    vis.create_window(height=540, width=960)
    # vis.get_render_option().load_from_json("misc/renderoption.json")

    print("4")
    vis.run()
    vis.destroy_window()
