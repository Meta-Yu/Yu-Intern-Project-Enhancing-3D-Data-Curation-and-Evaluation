#!/usr/bin/env bash

script=5
seq=6
recording=1

# category_name = kite
# scene_name = 401_52055_102127

python -m colmap_converter \
--colmap_dir /private/home/wangyu1369/COLMAP/co3d_result_nobg/sparse/0 \
--frames_dir /datasets01/co3dv2/080422/kite/401_52055_102127/images/ \
--scale=1 \
--split_nth=8 \
--dir_dst /private/home/wangyu1369/COLMAP/co3d_result/ \
--save_frames 0 \
--undistort 0 \
--c2w 1 \
--to_PT3d 1
