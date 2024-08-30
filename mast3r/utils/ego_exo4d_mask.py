import json

import numpy as np

# from dynamic.monocular.egoexo4d import is_exo_cam_id

from ego4d.research.util.masks import decode_mask
import cv2

import cv2
import numpy as np
from projectaria_tools.core import calibration
from projectaria_tools.core import data_provider, mps
import os


class MostlySeqFrameReader:
    def __init__(self, clip_filename):
        self.clip_filename = clip_filename
        self.cap = cv2.VideoCapture(clip_filename)

        self.current_idx = 0

    def get_frame(self, frame_idx):
        # If at init or frame idx can't be reached by reading forward, seek
        if (
            self.current_idx == 0
            or frame_idx < self.current_idx
            or frame_idx - self.current_idx > 90
        ):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            self.current_idx = frame_idx

        # Iterate through video until desired frame
        while self.current_idx < frame_idx:
            _, _ = self.cap.read()
            self.current_idx += 1

        # Read and return the desired frame
        _, img = self.cap.read()
        self.current_idx += 1

        return img


class MaskProvider:
    def __init__(self, take):
        masks_path = f"/checkpoint/haotang/data/egoexo/interpolated_mask/{take}.json"
        with open(masks_path, "r") as f:
            self.masks = json.load(f)

        self.cam_ids = set(
            [
                cam_id
                for obj_id in self.masks.keys()
                for cam_id in self.masks[obj_id].keys()
            ]
        )
        self.exo_cam_ids = [cam_id for cam_id in self.cam_ids if is_exo_cam_id(cam_id)]
        if len(self.exo_cam_ids) > 1:
            print("WARNING: More than one exo cam with Relations masks?")

    def is_valid_obj_mask(self, mask):
        if mask is None:
            return False
        if np.sum(mask) == 0:
            return False

        return True

    def get_mask(
        self, idx, obj_id, cam_id, undistort_func=None
    ):  # TODO: add mask loading from exo for scoring
        # Not Exo => Aria; add suffix
        cam_id = f"{cam_id}_214-1" if not is_exo_cam_id(cam_id) else cam_id

        # No mask for this object in this cam
        if cam_id not in self.cam_ids:
            return None

        # No mask for this frame idx
        obj_cam_masks = self.masks[obj_id][cam_id]["annotation"]
        if str(int(idx)) not in obj_cam_masks.keys():
            return None

        mask = decode_mask(obj_cam_masks[str(idx)])

        # Stack one-channel mask into "RGB" for undistortion
        if undistort_func is not None:
            mask = undistort_func(np.stack((mask.copy(),) * 3, axis=-1))[:, :, 0]

        # Back to single channel mask
        return mask

    def calc_mask_center(self, mask):
        # Center of bounding box (TODO: replace with centroid of mask?)
        _h, _w = np.nonzero(mask)

        # Origin: Upper left corner of image. x-axis to the right, y-axis down
        if len(_h) == 0:
            return None

        return (np.mean([_w.min(), _w.max()]), np.mean([_h.min(), _h.max()]), 1)

# masks_path = '/checkpoint/haotang/data/egoexo/interpolated_mask/sfu_cooking_008_3.json'

# with open(masks_path, "r") as f:
#     masks = json.load(f)

# print(masks.keys())
# print(set(masks['wooden chopping board_0']['aria01_214-1']["annotation"].keys()))
# print(decode_mask(masks['Big Knife_0']['aria01_214-1']["annotation"]['150']))

def mask_read_ego4d(scene_name, object, frame, size1 = 50, size2 =50, zoom = 1.55):

    index = str(int((frame.split('/')[-1]).split('.')[-2][-6:]))
    

    masks_path = '/checkpoint/haotang/data/egoexo/interpolated_mask/{}.json'.format(scene_name)

    DATA_BASEPATH = "/datasets01/egoexo4d/v2/takes"

    vrs_path = os.path.join(DATA_BASEPATH, '{}/aria01.vrs'.format(scene_name))

    cur_provider = data_provider.create_vrs_data_provider(vrs_path) 

    with open(masks_path, "r") as f:
        masks = json.load(f)
    
    if index not in set(masks[object]['aria01_214-1']["annotation"].keys()):

        mask_padding = np.zeros((size2, size1))
        
        final_mask = mask_padding > 0

    else:
        cur_mask = decode_mask(masks[object]['aria01_214-1']["annotation"][index])

        new_mask = undistort_img(np.stack((cur_mask.copy(),) * 3, axis=-1), provider = cur_provider, zoom = zoom)[:, :, 0]

        mask_padding = cv2.resize(cur_mask, (size1, size2))

        final_mask = mask_padding > 0

    # print(final_mask.shape)
    return final_mask


def undistort_aria(
    image_array, provider, sensor_name, focal_length, width=640, height=480
):
    device_calib = provider.get_device_calibration()
    src_calib = device_calib.get_camera_calib(sensor_name)

    # create output calibration: a linear model of image size 512x512 and focal length 150
    # Invisible pixels are shown as black.
    dst_calib = calibration.get_linear_camera_calibration(
        height, width, focal_length, sensor_name
    )

    # distort image
    rectified_array = calibration.distort_by_calibration(
        image_array, dst_calib, src_calib
    )
    return (
        rectified_array,
        dst_calib.get_principal_point(),
        dst_calib.get_focal_lengths(),
    )



def undistort_img(img, provider, zoom):
    img_undistorted, _, _ = undistort_aria(
        img,
        provider,
        "camera-rgb",
        150 * zoom,
        512,
        512,
    )
    return img_undistorted

def get_aria_frame(scene_name, idx, zoom):

    vrs_path = '/datasets01/egoexo4d/v2/takes/{}/aria01.vrs'.format(scene_name)

    cur_provider = data_provider.create_vrs_data_provider(vrs_path)

    aria_video_filepath = '/datasets01/egoexo4d/v2/takes/{}/frame_aligned_videos/aria01_214-1.mp4'.format(scene_name)

    all_frames = MostlySeqFrameReader(aria_video_filepath)

    # Retrieve Aria frame and undistort
    img = all_frames.get_frame(idx)
    return undistort_img(img, cur_provider, zoom = zoom)

if __name__ == '__main__':
    
    # print(1)

    masks_path = '/checkpoint/haotang/data/egoexo/interpolated_mask/sfu_cooking_008_3.json'
    # masks_path = '/checkpoint/haotang/data/egoexo/interpolated_mask/sfu_cooking_003_1.json'

    # with open(masks_path, "r") as f:
    #     masks = json.load(f)

    # # print(masks.keys())

    # test_mask = decode_mask(masks['wooden chopping board_0']['aria01_214-1']["annotation"]['150'])

    DATA_BASEPATH = "/datasets01/egoexo4d/v2/takes"

    vrs_path = os.path.join(DATA_BASEPATH, 'sfu_cooking_008_3/aria01.vrs')

    cur_provider = data_provider.create_vrs_data_provider(vrs_path)

    # new_mask = undistort_img(np.stack((test_mask.copy(),) * 3, axis=-1), provider = cur_provider)[:, :, 0]

    
    # print(new_mask)

    print(get_aria_frame(scene_name = 'sfu_cooking_008_3', idx = 150))


