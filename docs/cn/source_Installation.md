#### 2.使用 源码 安装

##### 环境需求

Ubuntu 22.04


```
sudo apt-get -y update
sudo apt-get -y install git
git clone https://github.com/IEIAuto/AutoDRRT.git
# Taken from https://docs.ros.org/en/humble/Installation/Ubuntu-Development-Setup.html
sudo apt update && sudo apt install -y \
  python3-flake8-docstrings \
  python3-pip \
  python3-pytest-cov \
  ros-dev-tools

sudo apt install -y \
  python3-flake8-blind-except \
  python3-flake8-builtins \
  python3-flake8-class-newline \
  python3-flake8-comprehensions \
  python3-flake8-deprecated \
  python3-flake8-import-order \
  python3-flake8-quotes \
  python3-pytest-repeat \
  python3-pytest-rerunfailures

# Initialize rosdep
sudo rosdep init
rosdep update

wget -O /tmp/amd64.env https://raw.githubusercontent.com/autowarefoundation/autoware/main/amd64.env && source /tmp/amd64.env

# For details: https://docs.ros.org/en/humble/How-To-Guides/Working-with-multiple-RMW-implementations.html
sudo apt update
rmw_implementation_dashed=$(eval sed -e "s/_/-/g" <<< "${rmw_implementation}")
sudo apt install ros-${rosdistro}-${rmw_implementation_dashed}

# (Optional) You set the default RMW implementation in the ~/.bashrc file.
echo '' >> ~/.bashrc && echo "export RMW_IMPLEMENTATION=${rmw_implementation}" >> ~/.bashrc

wget -O /tmp/amd64.env https://raw.githubusercontent.com/autowarefoundation/autoware/main/amd64.env && source /tmp/amd64.env

# Taken from https://github.com/astuff/pacmod3#installation
sudo apt install apt-transport-https
sudo sh -c 'echo "deb [trusted=yes] https://s3.amazonaws.com/autonomoustuff-repo/ $(lsb_release -sc) main" > /etc/apt/sources.list.d/autonomoustuff-public.list'
sudo apt update
sudo apt install ros-${rosdistro}-pacmod3

# Install gdown to download files from CMakeLists.txt
pip3 install gdown

sudo apt install geographiclib-tools

# Add EGM2008 geoid grid to geographiclib
sudo geographiclib-get-geoids egm2008-1

clang_format_version=16.0.0
pip3 install pre-commit clang-format==${clang_format_version}

sudo apt install golang

wget -O /tmp/amd64.env https://raw.githubusercontent.com/autowarefoundation/autoware/main/amd64.env && source /tmp/amd64.env

# Modified from: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

# A temporary workaround for Ubuntu 22.04 with the ubuntu2004 repository
if [[ "$(uname -m)" == "x86_64" ]]; then
  liburcu6_url=http://archive.ubuntu.com/ubuntu
else
  liburcu6_url=http://ports.ubuntu.com/ubuntu-ports
fi
sudo echo "deb $liburcu6_url focal main restricted" > /etc/apt/sources.list.d/focal.list

# TODO: Use 22.04 in https://github.com/autowarefoundation/autoware/pull/3084. Currently, 20.04 is intentionally used.
os=ubuntu2004
wget https://developer.download.nvidia.com/compute/cuda/repos/$os/$(uname -m)/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
cuda_version_dashed=$(eval sed -e "s/[.]/-/g" <<< "${cuda_version}")
sudo apt-get -y install cuda-${cuda_version_dashed}

# Taken from: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc

wget -O /tmp/amd64.env https://raw.githubusercontent.com/autowarefoundation/autoware/main/amd64.env && source /tmp/amd64.env

# Taken from: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing

sudo apt-get install libcudnn8=${cudnn_version} libcudnn8-dev=${cudnn_version}
sudo apt-mark hold libcudnn8 libcudnn8-dev

sudo apt-get install libnvinfer8=${tensorrt_version} libnvonnxparsers8=${tensorrt_version} libnvparsers8=${tensorrt_version} libnvinfer-plugin8=${tensorrt_version} libnvinfer-dev=${tensorrt_version} libnvonnxparsers-dev=${tensorrt_version} libnvparsers-dev=${tensorrt_version} libnvinfer-plugin-dev=${tensorrt_version}
sudo apt-mark hold libnvinfer8 libnvonnxparsers8 libnvparsers8 libnvinfer-plugin8 libnvinfer-dev libnvonnxparsers-dev libnvparsers-dev libnvinfer-plugin-dev

colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```
