#### 1.使用 DOCKER 安装

Docker可以确保项目中的所有开发者拥有一致的开发环境。推荐初学者、临时用户以及不熟悉Ubuntu使用。
```
## 1安装docker-engine
# Taken from: https://docs.docker.com/engine/install/ubuntu/
# And: https://docs.docker.com/engine/install/linux-postinstall/

# Uninstall old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install using the repository
sudo apt-get update

sudo apt-get install \
ca-certificates \
curl \
gnupg \
lsb-release

# Add Docker’s official GPG key:
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Use the following command to set up the repository:
echo \
"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Verify that Docker Engine is installed correctly by running the hello-world image.
sudo docker run hello-world
# Note: This command downloads a test image and runs it in a container. When the container runs, it prints a message and exits.


#以下步骤为可选，docker加入用户组后，执行docker命令前不需要添加sudo
1. 创建docker用户组

sudo groupadd docker

2. 添加当前用户加入docker用户组

sudo usermod -aG docker ${USER}

3. 重启docker服务

sudo systemctl restart docker

4. 切换或者退出当前账户再从新登入

docker ps

## 2安装 Nvidia Container Toolkit
# Taken from https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit

# Setup the package repository and the GPG key:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add - \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install the nvidia-docker2 package (and dependencies) after updating the package listing:
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart the Docker daemon to complete the installation after setting the default runtime:
sudo systemctl restart docker

##3开启容器调用显示器权限

#此步骤需要每次开机、重启后执行，终端输入

xhost +

##4下载镜像

#arm使用镜像
docker pull oncepursuit/autodrrt:humble_orin_cuda_20.04_release

#x86使用镜像
docker pull ghcr.io/autodrrtfoundation/autodrrt-universe:latest-cuda

##5载入镜像

docker load -i ghcr.io/autodrrtfoundation/autodrrt-universe:latest-cuda

##6克隆代码(假设目录位置为 ${workspace})

git clone https://github.com/IEIAuto/AutoDRRT.git

##7.1启动容器(orin)
docker run -it \
--net=host \
--runtime nvidia \
-e DISPLAY=$DISPLAY \
-v  /home/:/home/ \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v /dev/shm:/dev/shm \
-w ${workspace} \
--name=autodrrt \
oncepursuit/autodrrt:humble_orin_cuda_20.04_release
#7.2 启动容器(x86)
docker run -it \
--net=host \
--gpus all \
-e DISPLAY=$DISPLAY \
-v  /home/:/home/ \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-w ${workspace} \
-v /dev/shm:/dev/shm \
--name=autodrrt \
ghcr.io/autodrrtfoundation/autodrrt-universe:latest-cuda

##8进入容器

docker start autodrrt && docker exec -it autodrrt /bin/bash

##9在容器中编译

./env_set.sh && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=Off 
```
