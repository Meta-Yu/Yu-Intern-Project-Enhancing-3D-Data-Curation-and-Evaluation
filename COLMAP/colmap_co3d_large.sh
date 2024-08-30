#!/usr/bin/env bash

objects=(
ball
kite
book 
couch 
sandwich 
frisbee
hotdog
skateboard
suitcase
)

for object in "${objects[@]}"; do
    while read path; do

        colmap automatic_reconstructor \
            --image_path /private/home/wangyu1369/COLMAP/selected_frames_co3d/object_10/stride_12/${path} \
            --workspace_path /private/home/wangyu1369/COLMAP/co3d_colmap_result/object_10/stride_12/${path} \
            --use_gpu 0 
    done < /private/home/wangyu1369/COLMAP/selected_frames_co3d_text/object_10/stride_100/${object}.txt
done
