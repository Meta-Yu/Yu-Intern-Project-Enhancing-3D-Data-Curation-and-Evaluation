# :plate_with_cutlery: Enhancing-3D-Data-Curation-and-Evaluation
This repo contains detailed code implementation for Yu's intern project. For more details about this project, please refer to [doc](https://docs.google.com/document/d/1d3pc8mGhV6vpq-X5mqVx4hgQLamb4pk5sw6IcTcu21c/edit#heading=h.tpv8ntwk9eaw) and [slides](https://docs.google.com/presentation/d/1hSeTPODO1YXXciAybBdh-ti5hBPs8lWOmUFTfkfewSw/edit#slide=id.g27fbec2f167_0_0).

# :mate: Installation

## 1. Install DROID-SLAM:
  1. Check the requirements and install DROID-SLAM following the GitHub [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM).
  2. Download the model from google drive: [droid.pth](https://drive.google.com/file/d/1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh/view).

## 2. Install DUSt3R:
  1. Check the requirements and install DUSt3R following the GitHub [DUSt3R](https://github.com/naver/dust3r).
  2. Download a checkpoint `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`:
     
     ```
     mkdir -p checkpoints
     /wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints/
     ```

## 3. Install MASt3R:
  1. Check the requirements and install MASt3R following the GitHub [MASt3R](https://github.com/naver/mast3r).
  2. Download a checkpoint `DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth`:
     
     ```
     mkdir -p checkpoints/
     wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
     ```

## 4. Install COLMAP:
  1. Check the requirements and install MASt3R following the GitHub [COLMAP](https://github.com/colmap/colmap).
  2. Or run the following commands using devfair:
     ```
     colmap automatic_reconstructor
     --image_path
     --workspace_path
     --use_gpu 0
     ```
## 5. Install EVO for evaluating camera trajectory
  1. Check the requirements and install EVO following the GitHub [EVO](https://github.com/MichaelGrupp/evo/tree/master).
  2. Check files format before using EVO via: [EVO-formats](https://github.com/MichaelGrupp/evo/wiki/Formats).
     

# :ice_cube: Code Organization

## 1. code for running inference using DROID-SLAM are saved in folder `DROID-SLAM`:
- The repo for `DROID-SLAM` has the following structures:
```md
├── main/ # main code
│   ├── inference_on_co3d.py
│   └── inference_on_egoexo_4D.py /# main inferece code on CO3D and Ego-Exo 4D using DROID-SLAM 
├── utils/
│   ├── get_pose.py
│   ├── trajectory_evaluation.py
│   ├── generate_point_clouds.py
│   ├── normalize_pcd.py
│   ├── point_clouds_evaluations.py
└── └── pcd_visualization.py/ # utils functions for running inference using DROID-SLAM
```
- Explanation for code:
  - inference_on_co3d.py: inference code on CO3D dataset using DROID-SLAM
  - inference_on_egoexo_4D.py: inference code on Ego-Exo 4D dataset using DROID-SLAM
  - get_pose.py: save the camera poses estimated by DROID-SLAM and access the ground truth camera poses
  - trajectory_evaluation.py: evaluate the camera poses estimated by DROID-SLAM
  - generate_point_clouds.py: save the point clouds estimated by DROID-SLAM
  - normalize_pcd.py: normalize/standarize the estimated and ground truth point clouds
  - point_clouds_evaluations.py: evaluate the point clouds estimated by DROID-SLAM
  - pcd_visualization.py: visualize the estimated point clouds
 
## 2. code for running inference using DUSt3R are saved in folder `dust3r`:
- The repo for `dust3r` has the following structures:
```md
├── main/ # main code
│   ├── inference_on_co3d_dust3r.py
│   └── inference_on_egoexo_4D_dust3r.py /# main inferece code on CO3D and Ego-Exo 4D using DUSt3R
├── utils/
│   ├── get_pose_dust3r.py
│   ├── trajectory_evaluation_dust3r.py
│   ├── point_clouds_evaluations_dust3r.py
└── └── ego_exo4d_mask.py/ # utils functions for running inference using DUSt3R
```
- Explanation for code:
  - inference_on_co3d_dust3r: inference code on CO3D dataset using DUSt3R
  - inference_on_egoexo_4D_dust3r.py: inference code on Ego-Exo 4D dataset using DUSt3R
  - get_pose_dust3r.py: save the camera poses estimated by DUSt3R and access the ground truth camera poses
  - trajectory_evaluation_dust3r.py: evaluate the camera poses estimated by DUSt3R
  - point_clouds_evaluations.py: evaluate the point clouds estimated by DUSt3R
  - ego_exo4d_mask.py: extract frames and masks information for Ego-Exo 4D
 
## 3. code for running inference using DUSt3R are saved in folder `COLMAP`:
- The repo for `COLMAP` has the following structures:
```md
├── main/ # main code
│   ├── colmap_co3d_large.sh
│   ├── colmap_convert_single_co3d.sh
│   ├── colmap_convert_large_scale.sh
│   └── inference_on_co3d_colmap.py /# main inferece code on CO3D using COLMAP
├── colmap_converter/
│   ├── __init__.py
│   ├── __main__.py
│   ├── colmap_utils.py
│   ├── frames_utils.py
└── └── metadata_utils.py /# Module for convert estimation from COLMAP into desired format and save into json file
├── utils/
│   ├── trajectory_evaluation_colmap.py
│   ├── point_clouds_evaluations_colmap.py
│   ├── select_videos.py
│   ├── select_objects.py
│   ├── frames_utils.py
└── └── point_clouds_evaluations_colmap.py/ # utils functions for running inference using COLMAP
```
- Explanation for code:
  - inference_on_co3d_colmap: inference code on CO3D dataset using COLMAP
  - colmap_co3d_large.sh: run large-sclae inference on CO3D using COLMAP
  - colmap_convert_single_co3d.sh: convert estimation for single CO3D video from COLMAP into derised format and save in json file
  - colmap_convert_large_scale.sh: convert estimation for large_scale CO3D videos from COLMAP into derised format and save in json file
  - colmap_converter: Module for convert estimation from COLMAP into desired format and save into json file
  - trajectory_evaluation_colmap.py: evaluate the camera poses estimated by COLMAP
  - point_clouds_evaluations_colmap.py: evaluate the point clouds estimated by COLMAP
  - select_videos.py & select_objects.py: select objects and videos, save them into separate folder for running COLMAP
  -  frames_utils.py: selected frames preprocess 

# :file_folder: Data & Storage
### Accessing Datasets
1. CO3D dataset can be accessed via `cd /datasets01/co3dv2`.
2. Ego-Exo 4D dataset can be accessed as follows:
   1. Via FAIR Cluster: 'cd /datasets01/egoexo4d'
   2. Ground Truth camera poses can be accessed via:
      ```
      /checkpoint/xiaodongwang/flow/EgoExo4D/existing_frames/{}/true_poses_on_camera.pt
      ```


# Key Referece
1. DROID-SLAM: https://arxiv.org/abs/2108.10869
2. DUSt3R: https://arxiv.org/abs/2312.14132
3. MASt3R: https://arxiv.org/abs/2406.09756
4. COLMAP: https://colmap.github.io/
5. Density-aware Chamfer Distance: https://proceedings.neurips.cc/paper/2021/file/f3bd5ad57c8389a8a1a541a76be463bf-Paper.pdf
