import open3d as o3d
import torch
import numpy as np

def normalize_point_cloud(pcd):
    # Compute the centroid of the point cloud
    centroid = pcd.get_center()
    # Translate the centroid to the origin
    pcd.translate(-centroid)
    # Scale the point cloud to fit within a unit sphere
    distances = np.linalg.norm(np.asarray(pcd.points), axis=1)
    max_distance = np.max(distances)
    print(max_distance)
    pcd.scale(1 / max_distance, center=(0, 0, 0))
    return pcd

def normalize_gt_point_cloud(category_name, scene_name):
    path = '/datasets01/co3dv2/080422/{}/{}/pointcloud.ply'.format(category_name, scene_name)
    pcd = o3d.io.read_point_cloud(path)
    pcd_normalized = normalize_point_cloud(pcd)

    o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_gt.ply".format(category_name, scene_name), 
                             pcd_normalized)

def normalize_est_point_cloud(category_name, scene_name, background=False):
    if not background:
        est_file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_withbg.ply'.format(category_name, scene_name)
    else:
        est_file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_wobg.ply'.format(category_name, scene_name)
    pcd = o3d.io.read_point_cloud(est_file_path)
    pcd_normalized = normalize_point_cloud(pcd)

    if background:
        o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_withbg.ply".format(category_name, scene_name), 
                                pcd_normalized)
    else:
        o3d.io.write_point_cloud("/private/home/wangyu1369/DROID-SLAM/droid_slam/estimated_point_clouds/pcd_{}_{}_wobg.ply".format(category_name, scene_name), 
                                pcd_normalized)


if __name__ == '__main__':
        for cur_scene in ["apple/189_20393_38136/",
            # "cake/374_42274_84517/",
            # "kite/401_52055_102127",
            # "pizza/102_11950_20611",
            # "book/247_26469_51778"
            ]:
            cur_category_name = cur_scene.split("/")[0]
            cur_scene_name = cur_scene.split("/")[1]
            
            normalize_gt_point_cloud(category_name= cur_category_name, scene_name = cur_scene_name)

            normalize_est_point_cloud(category_name = cur_category_name, scene_name = cur_scene_name, background = False)
            
            normalize_est_point_cloud(category_name = cur_category_name, scene_name = cur_scene_name, background = True)