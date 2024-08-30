import os

import cv2
import numpy as np
from tqdm import tqdm


def convert_file_format(x, format_trg="bmp"):
    if format_trg is None:
        return x
    fn, ext = os.path.splitext(x)
    return fn + os.path.extsep + format_trg


def resize_image(cv2_imge, h, w):
    if [h, w] != cv2_imge.shape[:2]:
        im = cv2.resize(cv2_imge, (int(w), int(h)))
    else:
        im = cv2_imge
    return im


def load_image(path):
    return cv2.imread(path)


def undistort_img(img, meta):
    K = np.array(meta["intrinsics"])
    camera = meta["camera"]
    if camera["model"] != "RADIAL_FISHEYE":
        return img
    dist = np.array([-camera["params"][-2], -camera["params"][-1], 0.0, 0.0])
    # a heuristic Xingyu found
    dist *= 2
    img = cv2.undistort(img, K, dist)
    return img


def save_frames(
    dir_src,
    dir_dst,
    meta,
    format_src=None,
    format_trg=None,
    undistort=False,
):
    for k in tqdm(meta["ids_all"]):
        name_src = convert_file_format(meta["images"][k], format_src)
        path_src = os.path.join(dir_src, name_src)
        name_trg = convert_file_format(name_src, format_trg)
        path_trg = os.path.join(dir_dst, name_trg)
        frame = resize_image(load_image(path_src), meta["image_h"], meta["image_w"])
        # frame.save(path_trg)
        if undistort:
            frame = undistort_img(frame, meta)
        cv2.imwrite(path_trg, frame)
