import pickle
import os
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import matplotlib.pyplot as plt


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion into a 3x3 rotation matrix.
    """
    # Normalize the quaternion to avoid numerical issues
    quaternion = quaternion / np.linalg.norm(quaternion)
    
    # Extract quaternion components
    qx, qy, qz, qw = quaternion
    
    # Compute rotation matrix from quaternion
    rotation_matrix = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    return rotation_matrix

def trajectory_to_poses(trajectory):
    """
    Convert a trajectory of 7-dimensional vectors (position and quaternion) into a list of 12-dimensional poses (position and rotation matrix).
    """
    poses = []
    for vector in trajectory:
        position = vector[:3]
        quaternion = vector[3:]
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        pose = np.concatenate([position, rotation_matrix.flatten()])
        poses.append(pose)
    
    return poses


def get_ground_truth_pose(category_name, scene_name, stride= 1):
    image_dir_template = "/datasets01/co3dv2/080422/{}/{}/images/"
    image_dir = image_dir_template.format(category_name, scene_name)
    img_list = sorted(os.listdir(image_dir))

    f = open("/checkpoint/haotang/data/co3d_v2_annotation/{}/{}.pkl".format(category_name, scene_name),'rb')
    anno_json = pickle.load(f)
    traj = []
    for t in range(1, len(img_list)+1, stride):

        scene_id = "{}/{}/images/{}".format(category_name, scene_name, img_list[t-1])
        # if not os.path.exists(scene_id):
        #     continue
            # scene_id = "{}/{}/images/frame{:06d}.jpg".format(category_name, scene_name, t)

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
    
    file_path = '/private/home/wangyu1369/DROID-SLAM/poses_gt_est/{}_{}_gt.txt'.format(category_name, scene_name)
    with open(file_path, 'w') as file:
        for index, pose in enumerate(traj):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')
    
    # return np.array(traj)

def get_est_pose(category_name, scene_name):
    file_path = '/private/home/wangyu1369/DROID-SLAM/droid_slam/est_trajectory/{}_{}.npy'
    est_traj_path = file_path.format(category_name, scene_name)
    traj = np.load(est_traj_path)
    # poses = trajectory_to_poses(traj)
    file_name = '/private/home/wangyu1369/DROID-SLAM/poses_gt_est/{}_{}_est.txt'.format(category_name, scene_name)
    # print(traj)
    with open(file_name, 'w') as file:
        for index, pose in enumerate(traj):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')
    # return traj


if __name__ == "__main__":
    ## Test of traj plot 
    for cur_scene in ["apple/189_20393_38136/",
                "cake/374_42274_84517/",
                "pizza/102_11950_20611",
                "kite/401_52055_102127"
                ]:
        cur_category_name = cur_scene.split("/")[0]
        cur_scene_name = cur_scene.split("/")[1]
        # print(cur_scene)

        get_ground_truth_pose(category_name = cur_category_name, scene_name = cur_scene_name)
        get_est_pose(category_name = cur_category_name, scene_name = cur_scene_name)
