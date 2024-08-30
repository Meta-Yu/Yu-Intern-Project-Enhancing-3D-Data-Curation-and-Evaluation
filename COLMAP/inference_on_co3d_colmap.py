import numpy as np
import open3d as o3d

from read_write_model import read_points3D_text, read_points3D_binary
import json
from scipy.spatial.transform import Rotation as Rot
import os
import pickle
from colmap_converter.colmap_utils import read_cameras_binary, read_images_binary, read_points3d_binary
from select_videos import select_object
import collections
from trajectory_evaluation import compare_trajectory
from point_clouds_evaluations import calculate_chamfer_distance_new
import math


def poses_to_traj(poses):

    traj = []
    for index, value in poses.items():
        RT = np.array(value)
        R = Rot.from_matrix(RT[:3, :3]).as_quat()
        T = RT[:3, 3]
        traj.append(np.append(T, R))
    
    return traj

# data = read_images_binary('/private/home/wangyu1369/COLMAP/co3d_result_nobg/sparse/0/images.bin')

# print(data)

# pcd = read_points3D_binary('/private/home/wangyu1369/COLMAP/co3d_result_nobg/sparse/0/points3D.bin')
# points = []
# colors = []
# img_index = []
# for key, values in pcd.items():
#     points.append(values.xyz)
#     colors.append(values.rgb)
#     img_index.append(values.image_ids)

# points = np.concatenate(points, axis=0).reshape(-1, 3)
# colors = np.concatenate(colors, axis=0).reshape(-1, 3)/255.0
# # print(img_index[1])

# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)
# point_cloud.colors = o3d.utility.Vector3dVector(colors)

# o3d.io.write_point_cloud("/private/home/wangyu1369/COLMAP/est_point_clouds/point_cloud.ply", point_cloud)

# with open('/private/home/wangyu1369/COLMAP/co3d_result/meta.json', 'r') as f:
#     # Load the JSON data from the file
#     data = json.load(f)
# # print(data['poses'].keys())
# # print(poses_to_traj(data['poses'])[0])

def save_est_traj(poses,category_name, scene_name):
    traj = poses_to_traj(poses)
    file_name = '/private/home/wangyu1369/COLMAP/poses_gt_est/{}_{}_est.txt'.format(category_name, scene_name)
    with open(file_name, 'w') as file:
        for index, pose in enumerate(traj):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')

def get_ground_truth_pose(category_name, scene_name, stride= 1):
    image_dir_template = "/datasets01/co3dv2/080422/{}/{}/images/"
    image_dir = image_dir_template.format(category_name, scene_name)
    img_list = sorted(os.listdir(image_dir))

    f = open("/checkpoint/haotang/data/co3d_v2_annotation/{}/{}.pkl".format(category_name, scene_name),'rb')
    anno_json = pickle.load(f)
    traj = []
    for t in range(1, len(img_list)+1, stride):

        scene_id = "{}/{}/images/{}".format(category_name, scene_name, img_list[t-1])

        R = np.array(anno_json[scene_id]['viewpoint']['R']).T
        # rot_q = R.from_matrix(rot_mat).as_quat()
        T = np.array(anno_json[scene_id]['viewpoint']['T'])
        # Convert to camera to world
        RT = np.eye(4)
        RT[:3, :3] = R
        RT[:3, 3] = T
        # Change x, y axis's orientation
        RT[0:2] *= -1
        RT = np.linalg.inv(RT)
        # RT[..., 1:3] *= -1
        c2wR = Rot.from_matrix(RT[:3, :3]).as_quat()
        c2wT = RT[:3, 3]
        # c2wT /= 1000.0
        traj.append(np.append(c2wT, c2wR))
    # print(img_list)

    # poses = trajectory_to_poses(traj)
    
    file_path = '/private/home/wangyu1369/COLMAP/poses_gt_est/{}_{}_gt.txt'.format(category_name, scene_name)
    with open(file_path, 'w') as file:
        for index, pose in enumerate(traj):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')


# save_est_traj(poses = data['poses'],category_name = 'kite', scene_name = '401_52055_102127')

# get_ground_truth_pose(category_name = 'kite', scene_name = '401_52055_102127', stride= 1)

stride = 4


result_mean_std = collections.defaultdict(dict)
result = collections.defaultdict(dict)

for category_name in [
                    "ball", 
                    #   "book", 
                    #   "couch", 
                    #     "kite", 
                    #   "sandwich",
                    #   "skateboard", 
                    #    "suitcase"

                    #   "hotdog", 
                    #   "frisbee", 
                        ]:
    
    cur_error = collections.defaultdict(list)

    scene_list = select_object(category_name, number_of_frames=5)

    for scene_name in scene_list:
        with open('/private/home/wangyu1369/COLMAP/co3d_colmap_result/object_5/stride_4/{}_{}/meta.json'.format(category_name, scene_name), 'r') as f:
            # Load the JSON data from the file
            data = json.load(f)

        cur_poses = data['poses']

        save_est_traj(poses = cur_poses,category_name = category_name, scene_name = scene_name)

        get_ground_truth_pose(category_name = category_name, scene_name = scene_name, stride= 4)

        cur_ape_trans, cur_ape_rotat, cur_rpe_trans, cur_rpe_rotat = compare_trajectory(category_name, scene_name)

        cur_error['ape_translation_list'].append(round(cur_ape_trans, 4))
        cur_error['ape_rotation_list'].append(round(cur_ape_rotat, 4))
        cur_error['rpe_translation_list'].append(round(cur_rpe_trans, 4))
        cur_error['rpe_rotation_list'].append(round(cur_rpe_rotat, 4))



        pcd = read_points3D_binary('/private/home/wangyu1369/COLMAP/co3d_colmap_result/object_5/stride_4/{}_{}/sparse/0/points3D.bin'.format(category_name, scene_name))
        points = []
        colors = []
        img_index = []
        for key, values in pcd.items():
            points.append(values.xyz)
            colors.append(values.rgb)
            img_index.append(values.image_ids)

        points = np.concatenate(points, axis=0).reshape(-1, 3)
        colors = np.concatenate(colors, axis=0).reshape(-1, 3)/255.0
        # print(img_index[1])

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)


        o3d.io.write_point_cloud("/private/home/wangyu1369/COLMAP/est_point_clouds/point_cloud_{}_{}.ply".format(category_name, scene_name), point_cloud)
        cur_chamfer_distance = calculate_chamfer_distance_new(category_name, scene_name)
        print('cd: ', cur_chamfer_distance)
        if not math.isinf(cur_chamfer_distance):
            cur_error['chamfer_distance_list'].append(round(cur_chamfer_distance, 4))

    print("Finished Processing of {}".format(category_name))

    for key in cur_error.keys():
        result[category_name] = cur_error
        result_mean_std[category_name][key] = [np.mean(cur_error[key]), np.std(cur_error[key])]

result_string = json.dumps(result)
result_mean_std_string = json.dumps(result_mean_std)

if (202//stride+1)>=40:

    with open('/private/home/wangyu1369/COLMAP/errors/result_dense.json', 'w') as f:
        f.write(result_string)

    with open('/private/home/wangyu1369/COLMAP/errors/result_mean_std_string_dense.json', 'w') as f:
        f.write(result_mean_std_string)
elif 10<=(202//stride+1)<40:

    with open('/private/home/wangyu1369/COLMAP/errors/result_mediate_{}.json'.format(category_name), 'w') as f:
        f.write(result_string)

    with open('/private/home/wangyu1369/COLMAP/errors/result_mean_std_string_mediate_{}.json'.format(category_name), 'w') as f:
        f.write(result_mean_std_string)
else:
    with open('/private/home/wangyu1369/COLMAP/errors/result_sparse.json', 'w') as f:
        f.write(result_string)

    with open('/private/home/wangyu1369/COLMAP/errors/result_mean_std_string_sparse.json', 'w') as f:
        f.write(result_mean_std_string)



