export CONDAENV_HOME="/private/home/$USER/.conda/envs"

# Environment aliases
if [ -r $CONDAENV_HOME/fair-recipe ]; then
	dev() {
		module purge
		module load anaconda3/2023.3-1 cuda/11.7 cudnn/v8.4.1.50-cuda.11.6
		conda activate fair-recipe
	}
fi
