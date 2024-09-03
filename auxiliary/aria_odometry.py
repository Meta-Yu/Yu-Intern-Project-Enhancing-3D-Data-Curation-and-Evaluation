# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse
import copy
import json
import os
import torch
import math

import numpy as np
from egostatic.eval_utils import (
    check_box_in_frame,
    gen_gt_tracklet,
    get_mask_for_first_image,
    get_maskrcnn_model,
    get_pose,
    interpolate_pose
)
from egostatic.visualization_utils import gen_bounding_box
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R


def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y) / np.sum(X**2)
    return scale


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    assert x.shape == y.shape, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


class KittiEvalOdom:
    """Evaluate odometry result
    Usage example:
        vo_eval = KittiEvalOdom()
        vo_eval.eval(gt_pose_txt_dir, result_pose_txt_dir)
    """

    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    def load_poses_from_txt(self, file_name):
        """Load poses from txt (KITTI format)
        Each line in the file should follow one of the following structures
            (1) idx pose(3x4 matrix in terms of 12 numbers)
            (2) pose(3x4 matrix in terms of 12 numbers)
        Args:
            file_name (str): txt file path
        Returns:
            poses (dict): {idx: 4x4 array}
        """
        f = open(file_name, "r")
        s = f.readlines()
        f.close()
        poses = {}
        for cnt, line in enumerate(s):
            P = np.eye(4)
            line_split = [float(i) for i in line.split(" ") if i != ""]
            withIdx = len(line_split) == 13
            for row in range(3):
                for col in range(4):
                    P[row, col] = line_split[row * 4 + col + withIdx]
            if withIdx:
                frame_idx = line_split[0]
            else:
                frame_idx = cnt
            poses[frame_idx] = P
        return poses

    def trajectory_distances(self, poses):
        """Compute distance for each pose w.r.t frame-0
        Args:
            poses (dict): {idx: 4x4 array}
        Returns:
            dist (float list): distance of each pose w.r.t frame-0
        """
        dist = [0]
        sort_frame_idx = sorted(poses.keys())
        for i in range(len(sort_frame_idx) - 1):
            cur_frame_idx = sort_frame_idx[i]
            next_frame_idx = sort_frame_idx[i + 1]
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx**2 + dy**2 + dz**2))
        return dist

    def rotation_error(self, pose_error):
        """Compute rotation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            rot_error (float): rotation error
        """
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    def translation_error(self, pose_error):
        """Compute translation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            trans_error (float): translation error
        """
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
        return trans_error

    def last_frame_from_segment_length(self, dist, first_frame, length):
        """Find frame (index) that away from the first_frame with
        the required distance
        Args:
            dist (float list): distance of each pose w.r.t frame-0
            first_frame (int): start-frame index
            length (float): required distance
        Returns:
            i (int) / -1: end-frame index. if not found return -1
        """
        for i in range(first_frame, len(dist), 1):
            # print(dist[i], (dist[first_frame] + length))
            if dist[i] > (dist[first_frame] + length):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        """calculate sequence error
        Args:
            poses_gt (dict): {idx: 4x4 array}, ground truth poses
            poses_result (dict): {idx: 4x4 array}, predicted poses
        Returns:
            err (list list): [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        """
        err = []
        dist = self.trajectory_distances(poses_gt)
        self.step_size = 10

        for first_frame in range(0, len(poses_gt), self.step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(
                    dist, first_frame, len_
                )
                # print(last_frame)
                # Continue if sequence not long enough
                if (
                    last_frame == -1
                    or not (last_frame in poses_result.keys())
                    or not (first_frame in poses_result.keys())
                ):
                    continue

                # print(first_frame, len_)
                # compute rotational and translational errors
                pose_delta_gt = np.dot(
                    np.linalg.inv(poses_gt[first_frame]), poses_gt[last_frame]
                )
                pose_delta_result = np.dot(
                    np.linalg.inv(poses_result[first_frame]), poses_result[last_frame]
                )
                pose_error = np.dot(np.linalg.inv(pose_delta_result), pose_delta_gt)

                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)

                # compute speed
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)

                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
        return err

    def save_sequence_errors(self, err, file_name):
        """Save sequence error
        Args:
            err (list list): error information
            file_name (str): txt file for writing errors
        """
        with open(file_name, "w") as fp:
            for i in err:
                line_to_write = " ".join([str(j) for j in i])
                fp.writelines(line_to_write + "\n")

    def compute_overall_err(self, seq_err):
        """Compute average translation & rotation errors
        Args:
            seq_err (list list): [[r_err, t_err],[r_err, t_err],...]
                - r_err (float): rotation error
                - t_err (float): translation error
        Returns:
            ave_t_err (float): average translation error
            ave_r_err (float): average rotation error
        """
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        if seq_len > 0:
            for item in seq_err:
                r_err += item[1]
                t_err += item[2]
            ave_t_err = t_err / seq_len
            ave_r_err = r_err / seq_len
            return ave_t_err, ave_r_err
        else:
            return 0, 0

    def plot_trajectory(self, poses_gt, poses_result, scene_name):
        """Plot trajectory for both GT and prediction
        Args:
            poses_gt (dict): {idx: 4x4 array}; ground truth poses
            poses_result (dict): {idx: 4x4 array}; predicted poses
            seq (int): sequence index.
        """
        key2 = "EgoSLAM"
        plot_keys = ["Ground Truth", key2]
        fontsize_ = 20

        poses_dict = {}
        poses_dict["Ground Truth"] = poses_gt
        poses_dict[key2] = poses_result
        # print(len(poses_gt))
        # print(len(poses_result))

        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        # ax.set_aspect("equal")

        for key in plot_keys:
            pos_xz = []
            frame_idx_list = sorted(poses_dict[key2].keys())
            # print(frame_idx_list)
            for frame_idx in frame_idx_list:
                # pose = np.linalg.inv(poses_dict[key][frame_idx_list[0]]) @ poses_dict[key][frame_idx]
                pose = poses_dict[key][frame_idx]
                pos_xz.append([pose[0, 3], pose[2, 3]])
            pos_xz = np.asarray(pos_xz)
            plt.plot(pos_xz[:, 0], pos_xz[:, 1], label=key)

        plt.legend(loc="upper right", prop={"size": fontsize_})
        plt.xticks(fontsize=fontsize_)
        plt.yticks(fontsize=fontsize_)
        plt.xlabel("x (m)", fontsize=fontsize_)
        plt.ylabel("z (m)", fontsize=fontsize_)
        fig.set_size_inches(10, 10)
        png_title = "sequence_{}_{}".format(scene_name, model_name)
        fig_pdf = self.plot_path_dir + "/" + png_title + ".png"
        print("Save to {}".format(fig_pdf))
        plt.savefig(fig_pdf, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def plot_error(self, avg_segment_errs, scene_name):
        """Plot per-length error
        Args:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}
            seq (int): sequence index.
        """
        # Translation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][0] * 100)
            else:
                plot_y.append(0)
        fontsize_ = 10
        fig = plt.figure()
        plt.plot(plot_x, plot_y, "bs-", label="Translation Error")
        plt.ylabel("Translation Error (%)", fontsize=fontsize_)
        plt.xlabel("Path Length (m)", fontsize=fontsize_)
        plt.legend(loc="upper right", prop={"size": fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = self.plot_error_dir + "/trans_err_{}.pdf".format(scene_name)
        plt.savefig(fig_pdf, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # Rotation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][1] / np.pi * 180 * 100)
            else:
                plot_y.append(0)
        fontsize_ = 10
        fig = plt.figure()
        plt.plot(plot_x, plot_y, "bs-", label="Rotation Error")
        plt.ylabel("Rotation Error (deg/100m)", fontsize=fontsize_)
        plt.xlabel("Path Length (m)", fontsize=fontsize_)
        plt.legend(loc="upper right", prop={"size": fontsize_})
        fig.set_size_inches(5, 5)
        fig_pdf = self.plot_error_dir + "/rot_err_{}.pdf".format(scene_name)
        plt.savefig(fig_pdf, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def compute_segment_error(self, seq_errs):
        """This function calculates average errors for different segment.
        Args:
            seq_errs (list list): list of errs; [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        Returns:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}
        """

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_] != []:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    def compute_ATE(self, gt, pred):
        """Compute RMSE of ATE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        """
        t_errors = []
        r_errors = []
        for i in pred:
            # cur_gt = np.linalg.inv(gt_0) @ gt[i]
            cur_gt = gt[i]
            gt_xyz = cur_gt[:3, 3]

            # cur_pred = np.linalg.inv(pred_0) @ pred[i]
            cur_pred = pred[i]
            pred_xyz = cur_pred[:3, 3]
            # print("-" * 32)
            # print(cur_gt)
            # print(cur_pred)
            # print("-" * 32)

            rel_err = np.linalg.inv(cur_gt) @ cur_pred
            align_err = gt_xyz - pred_xyz

            t_errors.append(np.sqrt(np.sum(align_err**2)))
            r_errors.append(self.rotation_error(rel_err))

        ate_t = np.sqrt(np.mean(np.asarray(t_errors) ** 2))
        ate_r = np.sqrt(np.mean(np.asarray(r_errors) ** 2))
        np.save('ATE_rot.npy', np.asarray(r_errors))

        return ate_t, ate_r

    def compute_ARE(self, gt, pred):
        """Compute RMSE of ARE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        """
        errors = []
        idx_0 = list(pred.keys())[0]
        gt_0 = gt[idx_0]
        pred_0 = pred[idx_0]

        for i in pred:
            cur_gt = np.linalg.inv(gt_0) @ gt[i]
            cur_pred = np.linalg.inv(pred_0) @ pred[i]
            rel_err = np.linalg.inv(cur_gt) @ cur_pred
            errors.append(self.rotation_error(rel_err))
        np.save('ARE_rot.npy', np.asarray(errors))
        are = np.sqrt(np.mean(np.asarray(errors) ** 2))
        return are

    def compute_RPE(self, gt, pred):
        global rot_errors_aggr
        global trans_errors_aggr
        """Compute RPE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            rpe_trans
            rpe_rot
        """
        trans_errors = []
        rot_errors = []
        for i in list(pred.keys())[:-1]:
            # print(i)
            gt1 = gt[i]
            gt2 = gt[i + 1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred[i]
            if i + 1 not in pred:
                continue
            pred2 = pred[i + 1]
            pred_rel = np.linalg.inv(pred1) @ pred2
            # print("-" * 32)
            # print(gt_rel)
            # print(pred_rel)
            # print("-" * 32)
            rel_err = np.linalg.inv(gt_rel) @ pred_rel

            trans_errors.append(self.translation_error(rel_err))
            rot_errors.append(self.rotation_error(rel_err))
        # rpe_trans = np.sqrt(np.mean(np.asarray(trans_errors) ** 2))
        # rpe_rot = np.sqrt(np.mean(np.asarray(rot_errors) ** 2))
        rot_errors_aggr += rot_errors
        trans_errors_aggr += trans_errors
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        rpe_trans_median = np.median(np.asarray(trans_errors))
        rpe_rot_median = np.median(np.asarray(rot_errors))
        # print(np.min(np.asarray(trans_errors)))
        return rpe_trans, rpe_rot, rpe_trans_median, rpe_rot_median

    def scale_optimization(self, gt, pred):
        """Optimize scaling factor
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            new_pred (4x4 array dict): predicted poses after optimization
        """
        pred_updated = copy.deepcopy(pred)
        xyz_pred = []
        xyz_ref = []
        for i in pred:
            pose_pred = pred[i]
            pose_ref = gt[i]
            xyz_pred.append(pose_pred[:3, 3])
            xyz_ref.append(pose_ref[:3, 3])
        xyz_pred = np.asarray(xyz_pred)
        xyz_ref = np.asarray(xyz_ref)
        scale = scale_lse_solver(xyz_pred, xyz_ref)
        for i in pred_updated:
            pred_updated[i][:3, 3] *= scale
        return pred_updated

    def write_result(self, f, scene_name, errs):
        """Write result into a txt file
        Args:
            f (IOWrapper)
            seq (int): sequence number
            errs (list): [ave_t_err, ave_r_err, ate, rpe_trans, rpe_rot]
        """
        (
            ave_t_err,
            ave_r_err,
            ate,
            rpe_trans,
            rpe_rot,
            rpe_trans_median,
            rpe_rot_median,
        ) = errs
        lines = []
        lines.append("Sequence: \t {} \n".format(scene_name))
        lines.append("Trans. err. (%): \t {:.3f} \n".format(ave_t_err * 100))
        lines.append(
            "Rot. err. (deg/100m): \t {:.3f} \n".format(ave_r_err / np.pi * 180 * 100)
        )
        lines.append("ATE (m): \t {:.3f} \n".format(ate))
        lines.append("RPE (m) mean: \t {:.3f} \n".format(rpe_trans))
        lines.append("RPE (m) median: \t {:.3f} \n".format(rpe_trans_median))
        lines.append("RPE (deg) mean: \t {:.3f} \n\n".format(rpe_rot * 180 / np.pi))
        lines.append(
            "RPE (deg) median: \t {:.3f} \n\n".format(rpe_rot_median * 180 / np.pi)
        )
        for line in lines:
            f.writelines(line)

    def eval(  # noqa: C901
        self, gt_txt_path, result_txt_path, alignment=None, seqs=None, scene_names=None, model_name="",
    ):
        """Evaulate required/available sequences
        Args:
            gt_dir (str): ground truth poses txt files directory
            result_dir (str): pose predictions txt files directory
            alignment (str): if not None, optimize poses by
                - scale: optimize scale factor for trajectory alignment and evaluation
                - scale_7dof: optimize 7dof for alignment and use scale for trajectory evaluation
                - 7dof: optimize 7dof for alignment and evaluation
                - 6dof: optimize 6dof for alignment and evaluation
            seqs (list/None):
                - None: Evalute all available seqs in result_dir
                - list: list of sequence indexs to be evaluated
        """
        # Initialization
        if scene_names is None:
            return {}
        ave_t_errs = []
        ave_r_errs = []
        seq_ate_t = []
        seq_ate_r = []
        seq_are = []
        seq_rpe_trans = []
        seq_rpe_rot = []

        # Create result directory
        file_name = "test.txt"
        result_dir = "/private/home/xingyuchen/xingyuchen/egohowto/external/egostatic/"
        error_dir = result_dir + "/errors"
        self.plot_path_dir = result_dir + "/plot_aria"
        self.plot_error_dir = result_dir + "/plot_error"
        result_txt = os.path.join(result_dir, "result.txt")
        f = open(result_txt, "w")

        if not os.path.exists(error_dir):
            os.makedirs(error_dir)
        if not os.path.exists(self.plot_path_dir):
            os.makedirs(self.plot_path_dir)
        if not os.path.exists(self.plot_error_dir):
            os.makedirs(self.plot_error_dir)

        # evaluation
        # print(result_txt_path)
        for scene_name in scene_names:
            poses_result = self.load_poses_from_txt(result_txt_path)
            poses_gt = self.load_poses_from_txt(gt_txt_path)
            # print(poses_result.shape)
            # print(poses_gt.shape)
            # Pose alignment to first frame
            idx_0 = sorted(poses_result.keys())[0]
            pred_0 = poses_result[idx_0]
            gt_0 = poses_gt[idx_0]
            for cnt in poses_result:
                poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
                poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

            if alignment == "scale":
                poses_result = self.scale_optimization(poses_gt, poses_result)
            elif (
                alignment == "scale_7dof" or alignment == "7dof" or alignment == "6dof"
            ):
                # get XYZ
                xyz_gt = []
                xyz_result = []
                euler_gt = []
                euler_result = []
                for cnt in poses_result:
                    xyz_gt.append(
                        [poses_gt[cnt][0, 3], poses_gt[cnt][1, 3], poses_gt[cnt][2, 3]]
                    )
                    xyz_result.append(
                        [
                            poses_result[cnt][0, 3],
                            poses_result[cnt][1, 3],
                            poses_result[cnt][2, 3],
                        ]
                    )
                    gt_rot_mat = np.array(poses_gt[cnt][:3, :3])
                    euler_gt.append(
                        R.from_matrix(gt_rot_mat).as_euler("zxy", degrees=True)
                    )
                    result_rot_mat = np.array(poses_result[cnt][:3, :3])
                    euler_result.append(
                        R.from_matrix(result_rot_mat).as_euler("zxy", degrees=True)
                    )
                xyz_gt = np.asarray(xyz_gt).transpose(1, 0)
                xyz_result = np.asarray(xyz_result).transpose(1, 0)
                euler_gt = np.asarray(euler_gt).transpose(1, 0)
                euler_result = np.asarray(euler_result).transpose(1, 0)
                r, t, scale = umeyama_alignment(xyz_result, xyz_gt, alignment != "6dof")

                poses_result_xyz_alignment = {}
                # poses_result_euler_alignment = {}
                align_transformation = np.eye(4)
                align_transformation[:3:, :3] = r
                align_transformation[:3, 3] = t

                for cnt in poses_result:
                    poses_result_xyz_alignment[cnt] = np.copy(poses_result[cnt])
                    poses_result_xyz_alignment[cnt][:3, 3] *= scale
                    if alignment == "7dof" or alignment == "6dof":
                        poses_result_xyz_alignment[cnt] = (
                            align_transformation @ poses_result_xyz_alignment[cnt]
                        )

                # r, t, scale = umeyama_alignment(
                #     euler_result, euler_gt, alignment != "6dof"
                # )
                # align_transformation = np.eye(4)
                # align_transformation[:3:, :3] = r
                # align_transformation[:3, 3] = t
                # for cnt in poses_result:
                #     poses_result_euler_alignment[cnt] = np.copy(poses_result[cnt])
                #     poses_result_euler_alignment[cnt][:3, 3] *= scale
                #     if alignment == "7dof" or alignment == "6dof":
                #         poses_result_euler_alignment[cnt] = (
                #             align_transformation @ poses_result_euler_alignment[cnt]
                #         )

            # compute sequence errors
            seq_err = self.calc_sequence_errors(poses_gt, poses_result_xyz_alignment)
            self.save_sequence_errors(seq_err, error_dir + "/" + file_name)

            # Compute segment errors
            avg_segment_errs = self.compute_segment_error(seq_err)

            # compute overall error
            ave_t_err, ave_r_err = self.compute_overall_err(seq_err)
            # print(ave_t_err, ave_r_err)
            # print("Sequence: " + scene_name)
            # print("Translational error (%): ", ave_t_err * 100)
            # print("Rotational error (deg/100m): ", ave_r_err / np.pi * 180 * 100)
            ave_t_errs.append(ave_t_err)
            ave_r_errs.append(ave_r_err)

            # Compute ATE
            ate_t, ate_r = self.compute_ATE(poses_gt, poses_result_xyz_alignment)
            seq_ate_t.append(ate_t)
            seq_ate_r.append(ate_r)
            are = self.compute_ARE(poses_gt, poses_result)
            seq_are.append(are)
            # print("ATE (trans) (m): ", ate_t)
            # print("ATE (rot) (m): ", ate_r)
            # print("ARE (deg): ", are)

            # Compute RPE
            rpe_trans, rpe_rot, rpe_trans_median, rpe_rot_median = self.compute_RPE(
                poses_gt, poses_result
            )
            # rpe_trans, rpe_rot = 0.0, 0.0
            seq_rpe_trans.append(rpe_trans)
            seq_rpe_rot.append(rpe_rot)
            # print("RPE (m) mean: {} and median: {}".format(rpe_trans, rpe_trans_median))
            # print(
            #     "RPE (deg) mean: {} and median: {}".format(
            #         rpe_rot * 180 / np.pi, rpe_rot_median * 180 / np.pi
            #     )
            # )

            # Plotting

            # Save result summary
            # self.write_result(
            #     f,
            #     scene_name,
            #     [
            #         ave_t_err,
            #         ave_r_err,
            #         ate,
            #         rpe_trans,
            #         rpe_rot,
            #         rpe_trans_median,
            #         rpe_rot_median,
            #     ],
            # )

        f.close()
        if model_name != "colmap_test":
            self.plot_trajectory(poses_gt, poses_result_xyz_alignment, scene_name)
            self.plot_error(avg_segment_errs, scene_name)
            print("-------------------- For Copying ------------------------------")
            for i in range(len(ave_t_errs)):
                print("{0:.3f}".format(ave_t_errs[i] * 100))
                print("{0:.3f}".format(ave_r_errs[i] / np.pi * 180 * 100))
                print("{0:.2f}".format(seq_ate_t[i]))
                print("{0:.2f}".format(seq_ate_r[i]))
                print("{0:.3f}".format(seq_rpe_trans[i]))
                print("{0:.3f}".format(seq_rpe_rot[i] * 180 / np.pi))
        # print({
        #     "ATE (trans) (m)": seq_ate_t[i],
        #     "ATE (rot) (deg)": seq_ate_r[i],
        #     "ARE (deg)": seq_are[i],
        #     "RPE (m)": seq_rpe_trans[i],
        #     "RPE (deg)": seq_rpe_rot[i] * 180 / np.pi,
        #     "RPE (m) median": rpe_trans_median,
        #     "RPE (deg) median": rpe_rot_median * 180 / np.pi,
        #     # "predicted pose num": len(poses_result),
        #     # "gt pose num": len(poses_gt),
        #     # "scale": scale,
        # })
        return {
            "ATE": seq_ate_t[0],
            "ATE(rot)": seq_ate_r[0] * 180 / np.pi,
            "RPE_median": rpe_rot_median * 180 / np.pi,
            "RPE_mean": rpe_rot * 180 / np.pi,
        }


def convert_pose_to_txt(  # noqa: C901
    pose, txt_path, annotated_indices=None, inv=False
):
    with open(txt_path, "w") as f:
        if annotated_indices is None:
            for i in range(pose.shape[0]):
                mat_id = 0
                if inv:
                    tmp_pose = np.linalg.inv(pose[i])
                else:
                    tmp_pose = pose[i]
                for j in range(3):
                    for k in range(4):
                        f.write(str(tmp_pose[j, k]))
                        mat_id += 1
                        if mat_id != 12:
                            f.write(" ")
                f.write("\n")
        else:
            for ind in annotated_indices:
                if ind >= pose.shape[0]:
                    # print(ind)
                    continue
                f.write(str(int(ind)))
                f.write(" ")
                mat_id = 0
                if inv:
                    tmp_pose = np.linalg.inv(pose[ind])
                else:
                    tmp_pose = pose[ind]
                for j in range(3):
                    for k in range(4):
                        f.write(str(tmp_pose[j, k]))
                        mat_id += 1
                        if mat_id != 12:
                            f.write(" ")
                f.write("\n")


parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--model",
    type=str,
    default="ssl_nyuv2",
    help="the model to run for evaluation script",
)
parser.add_argument(
    "--intracklet", help="if set, evaluates intracklet solution", action="store_true"
)

def get_est_pose(model_name="droid_resnet", suffix="", world2cam=False, finetuned=True, pose_num=-1):
    # DROID_SLAM_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/trajectory_aria/trajectory_.npy"
    if finetuned:
        DROID_SLAM_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/trajectory_aria/trajectory_ft{}.npy".format(suffix)
    else:
        DROID_SLAM_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/trajectory_aria/trajectory{}.npy".format(suffix)
    DROID_SLAM_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/trajectory_aria/trajectory_75k_{}.npy".format(suffix)
    # DROID_SLAM_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/trajectory_aria/trajectory{}.npy".format(suffix)
    DROID_SLAM_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/egoslam_traj/trajectory_{}.npy".format(suffix)
    # DROID_SLAM_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/trajectory_ablation/trajectory_{}.npy".format(suffix)
    print(DROID_SLAM_PATH)
    droid_pose = np.load(DROID_SLAM_PATH)

    ind = 0
    if pose_num / droid_pose.shape[0] > 1.5:
        pose_4x4 = {}
        for frame_ind in range(0, droid_pose.shape[0]):
            rot_mat = R.from_quat(droid_pose[frame_ind][3:]).as_matrix()
            cur_poses = np.eye(4)
            cur_poses[:3, :3] = rot_mat
            cur_poses[:3 , 3] = droid_pose[frame_ind][:3]
            if world2cam:
                cur_poses = np.linalg.inv(cur_poses)
            pose_4x4[ind] = cur_poses
            ind += int(pose_num / droid_pose.shape[0])
        print(pose_num, len(pose_4x4))
        return interpolate_pose(
            pose_4x4, pose_num=pose_num
        )
    else:
        pose_4x4 = []
        for frame_ind in range(0, droid_pose.shape[0]):
            try:
                rot_mat = R.from_quat(droid_pose[frame_ind][3:]).as_matrix()
            except Exception:
                print(droid_pose)
            cur_poses = np.eye(4)
            cur_poses[:3, :3] = rot_mat
            cur_poses[:3 , 3] = droid_pose[frame_ind][:3]
            if world2cam:
                cur_poses = np.linalg.inv(cur_poses)
            pose_4x4.append(cur_poses)
        return np.array(pose_4x4)

def get_orbslam3_pose(prefix):
    # frame_num = 1756
    txt_path = "/checkpoint/xiaodongwang/flow/Aria/ORB_SLAM3/poses/{}.txt".format(prefix)
    poses_dict = {}
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n").split(" ")
            k = int(float(line[0]))
            translation = np.array([float(line[i]) for i in range(1, 4)])
            quaternion = np.array([float(line[i]) for i in range(4, 8)])
            rotation = R.from_quat(quaternion).as_matrix()
            rt = np.hstack((rotation, translation[:, None]))
            pose = np.eye(4)
            pose[:3, :] = rt
            # pose = np.linalg.inv(pose)
            poses_dict[int(k)] = pose
    return interpolate_pose(poses_dict, pose_num=max(poses_dict.keys()) + 1)

def get_orbslam2_pose(prefix):
    # frame_num = 1756
    txt_path = "/checkpoint/xiaodongwang/flow/Aria/ORB_SLAM2/poses/{}.txt".format(prefix)
    poses_dict = {}
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip("\n").split(" ")
            k = int(float(line[0]))
            translation = np.array([float(line[i]) for i in range(1, 4)])
            quaternion = np.array([float(line[i]) for i in range(4, 8)])
            rotation = R.from_quat(quaternion).as_matrix()
            rt = np.hstack((rotation, translation[:, None]))
            pose = np.eye(4)
            pose[:3, :] = rt
            # pose = np.linalg.inv(pose)
            poses_dict[int(k)] = pose
    # print(poses_dict)
    return interpolate_pose(poses_dict, pose_num=max(poses_dict.keys()) + 1)

def get_colmap_pose(location, script, seq, recording, world2cam=False):
    with open(
        "/checkpoint/weiyaowang/barf/aria/colmap/{}/{}/{}/{}/meta.json".format(location, script, seq, recording)
        # "/private/home/gleize/code/tmp/silk/var/colmap/xingyu_aria/script_5/seq_2/recording_1/meta.json"
    ) as f:
        data = json.load(f)
    pose_converted = {}
    for k, v in data["poses"].items():
        pose = np.array(v)
        pose4x4 = np.eye(4)
        pose4x4[:3, :] = pose
        # pose4x4[..., 1:3] *= -1
        pose_converted[int(k)] = pose4x4
    # print(len(pose_converted))
    # print(get_frame_num(dataset, scene_name, scenario))
    # print(len(pose_converted))
    colmap_pose = interpolate_pose(
        pose_converted, pose_num=max(pose_converted.keys()) + 1
    )
    # print(colmap_pose.shape)
    return colmap_pose


def get_silk_pose(location, script, seq, recording, world2cam=False):
    with open(
        # "/private/home/gleize/code/tmp/silk/var/colmap/xingyu_aria/script_5/seq_2/recording_1/meta.json"
        "/private/home/gleize/code/tmp/silk/var/colmap/xingyu_aria/{}/{}/{}/meta-0.json".format(script, seq, recording)
    ) as f:
        data = json.load(f)
    pose_converted = {}
    for k, v in data["poses"].items():
        pose = np.array(v)
        pose4x4 = np.eye(4)
        pose4x4[:3, :] = pose
        # pose4x4[..., 1:3] *= -1
        pose_converted[int(k)] = pose4x4
    print(len(pose_converted))
    # print(get_frame_num(dataset, scene_name, scenario))
    # print(len(pose_converted))
    colmap_pose = interpolate_pose(
        pose_converted, pose_num=max(pose_converted.keys()) + 1
    )
    # print(colmap_pose.shape)
    return colmap_pose


def get_gt_pose(ARIA_PATH, world2cam=False):
    ARIA_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/location_1_indoor/script_1/seq_3/recording_1/traj.npy"
    gt_pose = np.load(ARIA_PATH)

    pose_4x4 = []
    ind = 0
    for frame_ind in range(0, gt_pose.shape[0]):
        # print(gt_pose[ind])
        rot_mat = R.from_quat(gt_pose[frame_ind][[4, 5, 6, 3]]).as_matrix()
        # rot_mat = np.linalg.inv(rot_mat)
        # rot_mat = R.from_quat(gt_pose[ind][3:]).as_matrix()
        cur_poses = np.eye(4)
        cur_poses[:3, :3] = rot_mat
        cur_poses[:3 , 3] = gt_pose[frame_ind][:3]

        # tmp = cur_poses[..., [0]]
        cur_poses[..., [0, 2]] *= -1
        # cur_poses[..., [0]] = cur_poses[..., [1]]
        # cur_poses[..., [1]] = tmp
        # cur_poses[..., [2]] *= -1 
        # cur_poses[[2], ...] *= -1
        # cur_poses[1, 2] *= -1
        # cur_poses[2, 1] *= -1
        pose_4x4.append(cur_poses)
        ind += 1
    return np.array(pose_4x4)


def get_gt_pose_new(location, script, sequence, recording):
    script = script.replace("_", "")
    sequence = sequence.replace("_", "")
    recording = recording.replace("recording", "rec").replace("_", "")
    ARIA_PATH = "/checkpoint/xiaodongwang/flow/Aria_2024/{}_{}_{}_{}/true_poses_on_camera.pt".format(location, script, sequence, recording)
    gt_pose = torch.load(ARIA_PATH).numpy()

    pose_4x4 = []
    ind = 0
    for frame_ind in range(0, gt_pose.shape[0]):
        # print(gt_pose[ind])
        rot_mat = R.from_quat(gt_pose[frame_ind][[4, 5, 6, 3]]).as_matrix()
        rot_mat = R.from_quat(gt_pose[ind][3:]).as_matrix()
        cur_poses = np.eye(4)
        cur_poses[:3, :3] = rot_mat
        cur_poses[:3 , 3] = gt_pose[frame_ind][:3]

        pose_4x4.append(cur_poses)
        ind += 1
    return np.array(pose_4x4)


if __name__ == "__main__":
    args = parser.parse_args()
    vo_eval = KittiEvalOdom()
    model_name = args.model
    log_metric = {}
    first_frame_ind = 100
    base_depth_path = "/private/home/xingyuchen/xingyuchen/pixar_star/SensReader/scannet_testset/{}/depth/{}.npy"
    base_img_path = "/private/home/xingyuchen/xingyuchen/pixar_star/SensReader/scannet_testset/{}/color/{}.jpg"
    cam_int = np.array(
        [
            [577.87, 0.0, 319.5, 0.0],
            [0.0, 577.87, 239.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    model = get_maskrcnn_model()
    print("Loading maskrcnn finished")
    # scene_name = "ab5ac9c9-aef9-4301-9a7f-e46534b62ced"
    # scene_name = "148833ae-0717-4119-a700-3d04205375a9"
    # scene_name = "63ab5fc9-bccf-42b9-898c-2fb8b620b713"
    # scene_name = "5d0a4731-3a90-44ad-b2b4-cd2d6022eeff"
    # scene_name = "9b491af0-67ca-4280-b664-4b1ae78616c1"
    # scene_name = "0c9c4d10-b21c-40c6-a048-6f65d3ee9318"
    # scene_name = "1faee988-40b2-4618-92d4-265fd6564a83"
    # scene_name = "63ab5fc9-bccf-42b9-898c-2fb8b620b713"
    # scene_name = "1a3273d3-f6a7-4ff3-8339-c5eeca5b6999"
    # scene_name = "5d0a4731-3a90-44ad-b2b4-cd2d6022eeff"
    # scene_name = "0942be33-0df7-4e57-a71d-54c9966fd6f4"
    # scene_name = "cf5a31d4-fc50-4aaf-ab6d-c5d7c879fa55"
    scene_name = "aria_location2"
    # location = "location_1_indoor"
    # script = "script_1"
    # sequence = "seq_1"
    # recording = "recording_1"
    ates = []
    ates2 = []
    rot_errors_aggr = []
    trans_errors_aggr = []
    failed_cnt = 0
    cnt = 0
    rpes = []
    ater = []
    root_folder = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/"
    result_json = {}
    for location in ["location_1_indoor"]:
        script_list = os.listdir(os.path.join(root_folder, location))
        for script in script_list:
            if "script" in script:
                seq_list = os.listdir(os.path.join(root_folder, location, script))
                for seq in seq_list:
                    recording_list = os.listdir(os.path.join(root_folder, location, script, seq))
                    for recording in recording_list:
                        # script = "script_2"
                        # seq = "seq_3"
                        # recording = "recording_1"
                        ARIA_PATH = "/private/home/xingyuchen/xingyuchen/DROID-SLAM/{}/{}/{}/{}/traj.npy".format(location, script, seq, recording)
                        # try:
                        #     # result_pose = get_colmap_pose(location, script, seq, recording)[0:500]
                        #     scene_name = "{}_{}_{}_{}".format(location, script, seq, recording)
                        #     gt_pose = get_gt_pose_new("loc1", script, seq, recording)
                        #     # gt_pose = gt_pose[0:500]
                        #     gt_pose_txt_dir = "./tmp/gt_{}_test.txt".format(scene_name)
                        #     convert_pose_to_txt(gt_pose, gt_pose_txt_dir)
                        #     model_name = "colmap_test"
                        #     # result_pose_txt_dir = "./tmp/pred_{}_{}_test.txt".format(scene_name, model_name)
                        #     # convert_pose_to_txt(
                        #     #     result_pose, result_pose_txt_dir
                        #     # )
                        #     log_metric = vo_eval.eval(
                        #         gt_pose_txt_dir,
                        #         result_pose_txt_dir,
                        #         alignment="7dof",
                        #         scene_names=[scene_name],
                        #         model_name=model_name,
                        #     )
                        #     # if math.isnan(log_metric["ATE"]):
                        #     #     continue
                        # except Exception as e: 
                        #     print("Failed for {} {} {} {}".format(location, script, seq, recording))
                        #     print(e)
                        #     # raise e
                        #     # continue
                        try:
                            scene_name = "{}_{}_{}_{}".format(location, script, seq, recording)
                            gt_pose = get_gt_pose_new("loc1", script, seq, recording)
                            print(gt_pose.shape)
                            # gt_pose = get_colmap_pose(location, script, seq, recording)[0:500]
                            # gt_pose = gt_pose[0:500]
                            # gt_pose = gt_pose[0:500]                        
                            gt_pose_txt_dir = "./tmp/gt_{}_test.txt".format(scene_name)
                            print(gt_pose_txt_dir)
                            # convert_pose_to_txt(gt_pose, gt_pose_txt_dir, annotated_indices=annotated_indices)
                            if script == "script_2" and seq == "seq_6" and recording == "recording_2":
                                continue
                            convert_pose_to_txt(gt_pose, gt_pose_txt_dir)
                            suffix = "droid_dino_freeze_full_dist_lc_10k_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_dino_freeze_full_nolc_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_dino_freeze_full_dist_lc_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            orbslam_prefix = "{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_dino_freeze_full_030000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "droid_diffusion_fmin16_30k_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "droid_diffusion_lc_10k_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "droid_diffusion_tartan_dist_010000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "droid_resnet_tartan_dist_060000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "original_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_diffusion_tartan_dist_050000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "droid_diffusion_tartan_dist_contd_040000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "droid_original_dist_egotuning_030000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_diffusion_tartan_dist_lc_010000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_dino_tartan_dist_080000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_diffusion_fmin16_contd_035000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_dino_freeze_full_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "original_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "droid_dino_tartan_ego4d_dist_v2_070000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_dino_freeze_full_dist_lc_75k_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "droid_dino_freeze_full_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_diffusion_tartan_dist_contd_080000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_dino_freeze_noflow_005000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_dino_freeze_full_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            suffix = "original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # result_pose = get_est_pose(suffix=suffix, finetuned=False)[:500]
                            suffix = "egoslam_diffusion_dist_freeze_050000_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            # suffix = "droid_resnet_000005_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            
                            suffix = "egoslam_diffusion_dist_freeze_000001_original_{}_{}_{}_{}".format(location, script, seq, recording)
                            result_pose = get_est_pose(suffix=suffix, finetuned=False, pose_num=gt_pose.shape[0])
                            print(result_pose.shape)
                            # result_pose = get_colmap_pose(location, script, seq, recording)
                            # result_pose = get_orbslam3_pose()
                            # print(result_pose.shape)
                            # result_pose = get_colmap_pose(location, script, seq, recording)[0:500]
                            # result_pose = get_orbslam3_pose(orbslam_prefix)
                            # result_pose = get_orbslam2_pose(orbslam_prefix)
                            # result_pose = get_silk_pose(location, script, seq, recording)[0:500]
                            # result_pose = result_pose[:500]
                            model_name = "droid_slam_test"
                            result_pose_txt_dir = "./tmp/pred_{}_{}_test.txt".format(scene_name, model_name)
                            print(result_pose_txt_dir)
                            convert_pose_to_txt(
                                result_pose, result_pose_txt_dir
                            )
                            # convert_pose_to_txt(result_pose, result_pose_txt_dir)
                            log_metric = vo_eval.eval(
                                gt_pose_txt_dir,
                                result_pose_txt_dir,
                                alignment="7dof",
                                scene_names=[scene_name],
                                model_name=model_name,
                            )
                            import math
                            result_json[orbslam_prefix] = log_metric["ATE"]
                            if not math.isnan(log_metric["ATE"]):
                                ates.append(log_metric["ATE"])
                                rpes.append(log_metric["RPE_mean"])
                                ater.append(log_metric["ATE(rot)"])
                                # print(log_metric["RPE"], log_metric2["RPE"])
                            else:
                                failed_cnt += 1
                                # print(result_pose[:10], result_pose.shape)
                                print("Failed" * 32)

                            # print("ATE is {} for {}".format(log_metric["ATE"], ARIA_PATH))
                            # print("RPE median is {} for {}".format(log_metric["RPE_median"], ARIA_PATH))
                            # print("RPE mean is {} for {}".format(log_metric["RPE_mean"], ARIA_PATH))
                            # raise Exception
                            # cnt += 1
                            # if cnt == 10:
                            #     print("((((((((()))))))))")
                            #     sleep(10000)
                            # raise Exception
                        except Exception as e:
                            print(e)
                            # raise e
                            print("Failed at {}".format(ARIA_PATH))
                            # print("Final result:")
                            # print(np.mean(np.array(ates)))
                            # print(len(ates), failed_cnt)
                            # raise e
                            failed_cnt += 1
                            # raise e
                            pass

                        # raise Exception
                        # if args.intracklet:
                        #     with open(
                        #         "./scannet_results_0603/ate_scannet_{}_intracklet.json".format(model_name), "w"
                        #     ) as outfile:
                        #         outfile.write(json.dumps(log_metric, indent=4))
                        # else:
                        #     with open("./scannet_results_0603/ate_scannet_{}_test.json".format(model_name), "w") as outfile:
                        #         outfile.write(json.dumps(log_metric, indent=4))
    # global rot_errors_aggr
    # global trans_errors_aggr
    np.save('RPE_rot_origina.npy', np.asarray(rot_errors_aggr))
    np.save('RPE_trans_original.npy', np.asarray(trans_errors_aggr))
    print("Final result:")
    print(np.mean(np.array(ates)))
    print(np.mean(np.array(ater)))
    # print(np.median(np.array(rpes)))
    print(np.mean(np.array(rpes)))
    print(failed_cnt, len(ates))
    import json
    with open('aria_colmap.json', 'w') as f:
        json.dump(result_json, f, indent=4)