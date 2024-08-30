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

The repo structures code around three compoenents common to most machine learning research: 1) Data, 2) Model, 3) Experiment Logic

## 1. code for running inference using DROID-SLAM are saved in folder `DROID-SLAM`:
-
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

# :file_folder: Data & Storage
### Accessing Research Datasets
FAIR hosts the majority of common research datasets on the FAIR Cluster, in a folder called datasets01 (`cd /datasets01`). Always check datasets01 before individually downloading a research dataset!

### Storing Your Stuff:
- **Storing Code**: all of your code should be stored in the 'projects' folder on your dev machine. This projects folder exists at 'private/home/{username}/projects/'. This project folder is synced between different dev machines, so if you switch dev machines, you can still access your code. 

- **Storing Models, Experiment Logs, or Datasets**: each user has a 'checkpoint' folder that has two main properties: 1) It has much more storage than your projects folder, and 2) it can be accessed by others in FAIR, making it easy to share models or results with your team. Your checkpoint folder is located at '/checkpoint/{username}/'. You can cd into it by running 'cd checkpoint/{username}'. 


# :brain: Compute
There are several types of compute you have access to at FAIR, and this code is designed to allow you to quickly switch between them depending on the scale of your experiment. 

- **Devfair local GPUs**: each devfair has 2-4 GPUs which are shared among a few users of the devfair. This is great for testing code, or running small scale, quick experiments. However, all major experiments should be run on the cluster, which has built-in safety mechanisms for data processing and compute that prevent many job failures. In this codebase, you can run code on the local GPUs by setting mode to 'local'. 

- **FAIR Cluster**: the cluster is the massive set of GPUs which we use for pretty much all experiments. In this codebase, you can run code on the cluster by setting the mode to 'cluster'. You can specify the numer of GPUs or other parameters in the cluster config file (configs/mode/cluster.yaml). Partitions are groups of GPUs on the cluster that are designated for different teams or priority levels. Most FAIR users run their experiments on the general partition called devlab (the default for this repository).  

- **Have a question about compute?** You can look through the [FAIR Cluster Wiki](https://www.internalfb.com/intern/wiki/FAIR/Platforms/Clusters/FAIRClusters/), search the [FAIR Cluster Discussion & Support Workplace Group](https://fb.workplace.com/groups/FAIRClusterUsers/), or ask your managers!

# :notebook: Jupyter Notebook
There are several ways to set up and run jupyter notebooks. The easiest way is to use VSCode. If you prefer to start a jupyter server from the terminal, and use a local browser to access the notebooks, it's relatively easy to do too. Below we provide instructions both for VSCode and the command line (devfair and learnfair).

### VSCode Setup
Let's assume that you have your notebooks under the 'notebooks' folder in the main project directory. Similar to the 'notebooks/example.ipynb' provided in this project. In order to execute the notebook:
1. open it with VSCode
2. run a cell or press 'Run All'. This will open 'Select Kernel' control.
3. select 'Python Environments > YOUR_CONDA_ENVIRONMENT'. In this case 'Python Environments > fair-recipe'. This will start the server in the background and run the cells without you having to start the server separately. 
You can run the steps described above using 'notebooks/example.ipynb' notebook.

### Jupyter Setup using devfair conda environment in terminal
Another way of setting up jupyter is to start the jupyter server from the terminal. Here are steps that we can follow for that setup:
1. activate your conda environment. In this case with the following command: `conda activate fair-recipe`
2. start jupyter server. An example command to start the server is the following: 
`jupyter notebook --port 8090`. This will start the server and generate a URL to access the notebooks.
   For example: http://localhost:8090/?token=YOUR_TOKEN
3. forward your local host’s port 8090 to the devfair’s port 8090 so that you can access the notebooks from the local browser. 
This can be done with the following command from the local terminal:
`et devfair:8080 -t "8090:8090" --jport 8080`
More about jupyter setup from the terminal can be found [here](https://www.internalfb.com/intern/wiki/FAIR/Platforms/Clusters/FAIRClusters/Persist_SSH_Connection_with_ET/#jupyter-notebooks)

### Jupyter Setup using learnfair conda environment in terminal
Below we describe how to use jupyter from learnfair.
Let's assume that 8888 port was used to forward local port 8888 to the devfair’s port 8888.
`et devfair:8080 --tunnel "1234:22,18888:8888" --jport 8080`
Here are steps that you can follow for jupyter setup:
1. activate your conda environment. In this case with the following command: `conda activate fair-recipe`
2. start jupyter server. An example command to start the server is the following:
 `jupyter notebook --no-browser --ip 0.0.0.0 --port 8090`
This will generate the following link: http://learnfairXXX:8090/?token=YOUR_TOKEN
3. on the devfair in a separate terminal, forward 8888 port to the jupyter server's port. In our case 8090.
`ssh -R 8888:learnfairXXX:8090 user@devfairYYY`
4. now you can access jupyter notebook from the local browser using the following URL:
http://localhost:8888/notebooks/notebooks/example.ipynb

### Git and Jupyter notebook
Since Jupyter notebooks' output cells get regenerated each time we execute them, we might want to trim their output before pushing the code to github. In order to trim the output cells we can add the following line in `.gitattributes` file:
`*.ipynb filter=strip-notebook-output`.

In some cases it might be useful to leave generated output cells so that the users don't have to rerun the cells to generate the output.

# :wrench: Other tips and tools

* Log experiments using [Weights and Biases](https://fb.workplace.com/groups/623655025018541) available at FAIR
* Snapshot code so you can develop before jobs are queued in SLURM, based on [Matt Le's Sync Code](https://fb.workplace.com/groups/airesearchinfrausers/posts/1774890499334188/?comment_id=1774892729333965&reply_comment_id=1775084782648093)
* Don't rely on Jupyter notebooks for code you'll need to reuse, instead write functions in modules (`.py` files) and call them in your notebook. Use [AutoReload](https://ipython.org/ipython-doc/3/config/extensions/autoreload.html) to make this seamless
* Hydra has autocompletion, run `eval "$(python main.py -sc install=bash)"` . Consult [this guide](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion/) for setup details
* Setup code formatting so code is readable (and no one quibbles over style). We recommend the [Black Formatter](https://github.com/psf/black) (used in this repo)
* Loading modules and environments in your bash_profile. 
  <details>
    <summary> Vivien C.'s Example</summary>

    ```bash
    export CONDAENV_HOME="/private/home/$USER/.conda/envs"

    # Load cuda modules and the fair-recipe conda env
    if [ -r $CONDAENV_HOME/fair-recipe ]; then
      dev() {
        module purge
        module load anaconda3/2023.3-1 cuda/11.7 cudnn/v8.4.1.50-cuda.11.6
        conda activate fair-recipe
      }
    fi
    ```
  </details>
* If you're stuck, consult the [FAIR Cluster Wiki](https://www.internalfb.com/intern/wiki/FAIR/Platforms/Clusters/FAIRClusters/). If you're still stuck, post a question in the [Workplace Group](https://fb.workplace.com/groups/airesearchinfrausers).
  <details>
    <summary> Multi-GPU Distributed Training Options</summary>
    
    * Native [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html): for those who are brave and don't want to rely on other libraries.
    * [Torch TNT](https://pytorch.org/tnt/stable/): light-weight trainer supported internally.
    * [Accelerate](https://github.com/huggingface/accelerate): less abstraction, lighter-weight trainer that doesn't replace standard loop
    * [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/): heavier weight with bells and whistles, but more abstraction.
    * Hugging Face Trainer: heavier weight, but popular trainer integrated with the many models available on HuggingFace.
    * Mosaic Composer: optimized trainer with fast data loading using FFCV and other bells and whistles.
  </details>

# Want to contribute?

 We welcome [Pull Requests](https://github.com/fairinternal/fair-getting-started-recipe/pulls) with improvements or suggestions.
 If you want to flag an issue or propose an improvement, but dont' know how to realize it, create a [GitHub Issue](https://github.com/fairinternal/fair-getting-started-recipe/issues).


# Thanks to
* Jack Urbaneck, Matthew Muckley, Pierre Gleize, Ashutosh Kumar, Megan Richards, Haider Al-Tahan, Narine Kokhlikyan, Ouail Kitouni, and Vivien Cabannes for contributions and feedback
* The CIFAR10 [PyTorch Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
) on which the training is based 
* [Hydra Lightning Template](https://github.com/ashleve/lightning-hydra-template) for inspiration on code organization
