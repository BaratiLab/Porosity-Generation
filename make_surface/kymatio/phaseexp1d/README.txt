
```Create environment phc from file
conda env create -f phc_environment_roberto.yml

In case your cuda is not in bash env,
```INCLUDE it
export CUDA_INC_DIR=/usr/local/cuda/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin/

In case you have already phc,
```Rename conda environment:

conda create --name new_name --clone old_name
conda remove --name old_name --all # or its alias: `conda env remove --name old_name`


