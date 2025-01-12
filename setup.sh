apt-get update
apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
apt-get install -y patchelf
pip install -U 'mujoco-py<2.2,>=2.1'
pip install -U torch
pip install -U pandas
pip install -U matplotlib
pip install -U gym==0.23.1
pip install -U gym[mujoco]
pip install "cython<3"
pip install d3rlpy==1.1.1
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
export D4RL_SUPPRESS_IMPORT_ERROR=1
pip install wandb
pip install protobuf==3.20.*
pip install UtilsRL
pip install scikit-learn
pip install -U gym==0.23.1