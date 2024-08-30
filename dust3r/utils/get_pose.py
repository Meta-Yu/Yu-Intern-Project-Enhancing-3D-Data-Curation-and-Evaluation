from scipy.spatial.transform import Rotation as Rot
import torch
import numpy as np
import os
import pickle

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
    
    file_path = '/private/home/wangyu1369/dust3r/poses_gt_est/{}_{}_gt.txt'.format(category_name, scene_name)
    with open(file_path, 'w') as file:
        for index, pose in enumerate(traj):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')

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

def save_est_traj(poses,category_name, scene_name):
    traj = poses_to_traj(poses)
    file_name = '/private/home/wangyu1369/dust3r/poses_gt_est/{}_{}_est.txt'.format(category_name, scene_name)
    with open(file_name, 'w') as file:
        for index, pose in enumerate(traj):
            line = str(index) + ' '+  ' '.join(map(str, pose))
            file.write(line + '\n')


if __name__ == "__main__":
    tensor_data = torch.randn(3, 4, 4)
    print(poses_to_traj(tensor_data))