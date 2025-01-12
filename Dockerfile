FROM ubuntu:20.04
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y gnupg2
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get update && apt-get install -y build-essential
RUN apt-get remove -y --auto-remove python3.8
RUN apt-get install -y python3.10
RUN ln -s python3.10 /usr/bin/python3
RUN apt-get install -y curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN apt-get install -y python3.10-distutils
RUN apt-get install -y python3.10-dev
RUN apt-get update
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
RUN apt-get install -y patchelf
RUN python3 -m pip install -U 'mujoco-py<2.2,>=2.1'
RUN python3 -m pip install -U torch
RUN python3 -m pip install -U pandas
RUN python3 -m pip install -U matplotlib
RUN python3 -m pip install -U gym==0.23.1
RUN python3 -m pip install -U gym[mujoco]
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz
RUN mkdir /root/.mujoco
RUN tar -xf mujoco210.tar.gz --directory /root/.mujoco
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
RUN python3 -m pip install "cython<3"
RUN python3 -m pip install d3rlpy==1.1.1
RUN python3 -m pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
ENV D4RL_SUPPRESS_IMPORT_ERROR=1
RUN python3 -m pip install wandb
RUN python3 -m pip install protobuf==3.20.*
RUN python3 -m pip install UtilsRL
RUN python3 -m pip install scikit-learn
RUN python3 -m pip install -U gym==0.23.1