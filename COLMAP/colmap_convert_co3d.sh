#!/usr/bin/env bash

python -m colmap_converter \
--colmap_dir /private/home/wangyu1369/COLMAP/visualization/co3d_result/ball_113_13350_23632/sparse/0/ \
--frames_dir /private/home/wangyu1369/COLMAP/visualization/selected_frames/ball_113_13350_23632/ \
--scale=1 \
--split_nth=8 \
--dir_dst /private/home/wangyu1369/COLMAP/visualization/co3d_result/ball_113_13350_23632 \
--save_frames 0 \
--undistort 0 \
--c2w 1 \
--to_PT3d 0

