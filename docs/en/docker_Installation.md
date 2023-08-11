#### 1. Installation using DOCKER

Docker ensures that all developers in the project have a consistent development environment. It is recommended for beginners, temporary users, and those unfamiliar with Ubuntu.

```shell
## 1. Install docker-engine
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

# The following steps are optional; after adding docker to the user group, the 'sudo' prefix is not needed before docker commands
# 1. Create a docker user group
sudo groupadd docker

# 2. Add the current user to the docker user group
sudo usermod -aG docker ${USER}

# 3. Restart the docker service
sudo systemctl restart docker

# 4. Switch or log out of the current account and log back in
docker ps

## 2. Install Nvidia Container Toolkit
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


## 3. Grant Container Access to Display

# This step needs to be executed after each boot or restart; enter the following command in the terminal
xhost +

## 4. Download Images

# For ARM architecture
docker pull oncepursuit/autodrrt:humble_orin_cuda_20.04_release

# For x86 architecture
docker pull ghcr.io/autodrrtfoundation/autodrrt-universe:latest-cuda

## 5. Load Images

docker load -i ghcr.io/autodrrtfoundation/autodrrt-universe:latest-cuda

## 6. Clone the Code (Assuming the directory location is ${workspace})

git clone https://github.com/IEIAuto/AutoDRRT.git

## 7.1 Start the Container (for ORIN/arm64)

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

## 7.2 Start the Container (for x86)

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


## 8. Enter the Container

docker start autodrrt && docker exec -it autodrrt /bin/bash

## 9. Compile Inside the Container

./env_set.sh && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=Off 
