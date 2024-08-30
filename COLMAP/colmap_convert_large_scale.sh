#!/usr/bin/env bash


objects=(
ball
kite
book 
couch 
sandwich 
skateboard
suitcase
hotdog
frisbee
)
number_object=10

stride=12

for object in "${objects[@]}"; do
    while read path; do

    python -m colmap_converter \
    --colmap_dir /private/home/wangyu1369/COLMAP/co3d_colmap_result/object_${number_object}/stride_${stride}/${path}/sparse/0 \
    --frames_dir /private/home/wangyu1369/COLMAP/selected_frames_co3d/object_${number_object}/stride_${stride}/${path} \
    --scale=1 \
    --split_nth=8 \
    --dir_dst /private/home/wangyu1369/COLMAP/co3d_colmap_result/object_${number_object}/stride_${stride}/${path} \
    --save_frames 0 \
    --undistort 0 \
    --c2w 1 \
    --to_PT3d 0

    done < /private/home/wangyu1369/COLMAP/selected_frames_co3d_text/object_${number_object}/stride_${stride}#/${object}.txt
done
