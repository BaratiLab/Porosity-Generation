# Create the virtualenv
#conda create --name phc python=3.5

# Activate it
#source activate phc

# Install helpful packages
#conda install ipython numpy scipy scikit-learn jupyter matplotlib nose h5py pandas joblib sympy tqdm cython numba statsmodels

# librosa
#pip install librosa

# cuda toolkit
#conda install cudatoolkit

# install ffmpeg and other stuff (to read and write audio)
#sudo apt-get install libav-tools

# CuPy and nvrtc for the complex modulus
export CUDA_INC_DIR=/usr/local/cuda/include
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin/

#pip install scikit-cuda
pip install cupy
pip install pynvrtc

# install pytorch
#conda install pytorch torchvision cuda91 -c pytorch
conda install pytorch=0.4.1 cuda91 -c pytorch


# deactivate it
source deactivate phc

