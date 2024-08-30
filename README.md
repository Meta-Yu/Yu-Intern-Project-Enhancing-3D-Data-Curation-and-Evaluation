# :plate_with_cutlery: A Getting Started Recipe 
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.11+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

> a recipe to jumpstart your exploratory research at FAIR :earth_africa:

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
### Accessing Research Datasets
FAIR hosts the majority of common research datasets on the FAIR Cluster, in a folder called datasets01 (`cd /datasets01`). Always check datasets01 before individually downloading a research dataset!



# :brain: Compute
There are several types of compute you have access to at FAIR, and this code is designed to allow you to quickly switch between them depending on the scale of your experiment. 

- **Devfair local GPUs**: each devfair has 2-4 GPUs which are shared among a few users of the devfair. This is great for testing code, or running small scale, quick experiments. However, all major experiments should be run on the cluster, which has built-in safety mechanisms for data processing and compute that prevent many job failures. In this codebase, you can run code on the local GPUs by setting mode to 'local'. 

- **FAIR Cluster**: the cluster is the massive set of GPUs which we use for pretty much all experiments. In this codebase, you can run code on the cluster by setting the mode to 'cluster'. You can specify the numer of GPUs or other parameters in the cluster config file (configs/mode/cluster.yaml). Partitions are groups of GPUs on the cluster that are designated for different teams or priority levels. Most FAIR users run their experiments on the general partition called devlab (the default for this repository).  

- **Have a question about compute?** You can look through the [FAIR Cluster Wiki](https://www.internalfb.com/intern/wiki/FAIR/Platforms/Clusters/FAIRClusters/), search the [FAIR Cluster Discussion & Support Workplace Group](https://fb.workplace.com/groups/FAIRClusterUsers/), or ask your managers!



# Thanks to
* Jack Urbaneck, Matthew Muckley, Pierre Gleize, Ashutosh Kumar, Megan Richards, Haider Al-Tahan, Narine Kokhlikyan, Ouail Kitouni, and Vivien Cabannes for contributions and feedback
* The CIFAR10 [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
) on which the training is based 
* [Hydra Lightning Template](https://github.com/ashleve/lightning-hydra-template) for inspiration on code organization
