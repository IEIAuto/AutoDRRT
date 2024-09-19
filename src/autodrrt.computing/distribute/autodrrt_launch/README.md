# humble版本分布式环境配置

## 1.autoware分布式文件以及代码要求

### 1.1 项目代码要求

X86和四个orin的autoware代码目录一致，例如均位于/home/orin/autoware/目录下

X86和四个orin上均有autoware_map文件，且目录需一致

X86端拉取该项目代码，注意分支为humble，且拉取该代码至autoware项目的./src/目录下

将该项目的env.sh文件放置于各个orin的autoware目录下

各个设备设置DDS参数：
```
sudo sysctl -w net.core.rmem_max=2147483647
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl net.ipv4.ipfrag_time=3
sudo sysctl net.ipv4.ipfrag_high_thresh=134217728
```

注意：目前orin容器中有些小的问题：

```

RMW_IMPLEMENTATION=rmw_cyclonedds_cpp #修改~/.bahsrc文件
rm -rf /opt/ros/humble/install/rmw_fastrtps_cpp # 删除rmw_fastrtps_cpp
```


## 2.共享目录NFS设置

共享目录NFS设置原因：autoware软件产生的参数文件以临时文件格式存储于临时文件目录中，分布式程序需共享该临时文件目录，以实现共享参数的目标。官方默认目录为/tmp目录，但若共享该目录会产生一些系统bug，因此该步骤主要有两个配置工作：

### 2.1 设置该临时文件目录为共享目录（orin和X86共享目录为/home/orin/autoware/tmp）

注意：该步骤推荐在创建容器前进行配置（如果已经建好了容器，需docker restart重启容器）

X86端：
```
sudo mkdir -p /home/orin/autoware/tmp
sudo chmod -R 777 /home/orin/autoware/tmp
sudo vim /etc/exports # 添加 /home/orin/autoware/tmp *(rw,sync,all_squash,anongid=0,anonuid=0)
/etc/init.d/nfs-kernel-server start
sudo exportfs -rv
systemctl start nfs-kernel-server.service
```

orin端：192.168.3.2为X86的IP地址，按需修改。

```
sudo mkdir -p /home/orin/autoware/tmp
sudo chmod -R 777 /home/orin/autoware/tmp
sudo mount 192.168.3.2:/home/orin/autoware/tmp /home/orin/autoware/tmp
```

orin使用`df -h`可查看挂载结果。

orin重新启动后，该挂载会消失，需在orin端重新进行挂载命令。

### 2.2 修改临时文件目录为/home/orin/autoware/tmp

在X86和orin容器中（X86和orin每个容器都需要修改），修改临时文件目录(X86使用的是python3.10)
```
sudo vim /opt/ros/humble/install/launch_ros/lib/python3.8/site-packages/launch_ros/actions/node.py  #修改370行，加入dir='/home/orin/autoware_awsim/tmp/'

with NamedTemporaryFile(mode='w', prefix='launch_params_', delete=False, dir='/home/orin/autoware/tmp/') as h:

sudo vim /opt/ros/humble/install/launch_ros/lib/python3.8/site-packages/launch_ros/parameter_descriptions.py   #修改241行，加入dir='/home/orin/autoware_awsim/tmp/'

with open(param_file_path, 'r') as f, NamedTemporaryFile(mode='w', prefix='launch_params_', delete=False, dir='/home/orin/autoware/tmp/') as h
```

### 2.3 结果验证
X86容器和orin容器中进行验证：

X86容器内：`echo "I am writting on X86." > /home/orin/autoware/tmp/a.txt`

orin容器内：`cat /home/orin/autoware/tmp/a.txt`  #结果为I am writting on X86

`ls /home/orin/autoware/tmp`服务器和orin的文件内容相同


## 3.设置SSH免密登录

目标：从X86容器可以免密登录到各个orin容器中，实现以SSH协议发送启动命令。

注意：X86和每个orin都需设置一个端口号，推荐端口号设置为19010
### 3.1 X86端容器内：
```
sudo apt update
sudo apt install openssh-server

echo "Port 19010" >> /etc/ssh/sshd_config
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config
echo "MaxStartups 1000" >> /etc/ssh/sshd_config
echo "MaxSessions 1000" >> /etc/ssh/sshd_config
service ssh restart
passwd root  #密码设置为1

ssh-keygen -t rsa #连续回车
```

在X86容器外执行 ssh root@localhost -p 19010  19010为端口号，如果可以进入容器，则证明成功


### 3.2 orin容器内：
```
sudo apt update
sudo apt install openssh-server

sudo dpkg -i openssh-client_8.2p1-4_arm64.deb
sudo dpkg -i openssh-sftp-server_8.2p1-4_arm64.deb
sudo dpkg -i openssh-server_8.2p1-4_arm64.deb


echo "Port 19010" >> /etc/ssh/sshd_config  #注意修改端口号
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config
echo "MaxStartups 1000" >> /etc/ssh/sshd_config
echo "MaxSessions 1000" >> /etc/ssh/sshd_config
service ssh restart
passwd root  #密码设置为1

ssh-keygen -t rsa #连续回车

cd /root/.ssh/
ssh root@192.168.1.4 -p 19010 cat ~/.ssh/id_rsa.pub>> authorized_keys  #IP地址为X86的ip地址，端口号为X86的端口号

```

在orin容器外执行` ssh root@localhost -p 19010`端口号，如果可以进入容器，则证明成功

### 3.3
在X86端容器内执行` ssh root@192.168.3.5 -p 19010`, 如果可以进入到orin的容器，则证明ssh配置成功


## 4.X86和orin通信验证
```
ros2 run examples_rclcpp_minimal_service service_main #X86执行
ros2 run examples_rclcpp_minimal_client client_main  #orin执行
```


## 5.autoware分布式启动


X86端通过修改launch软件包配置文件，实现分布式节点配置

注：若远程启动lidar_centerpoint节点，则需修改如下文件：
```
vim autoware/src/universe/autoware.universe/launch/tier4_perception_launch/launch/perception.launch.xml      #12行修改为 default="centerpoint_tiny"
```
### 5.1 文件配置修改

配置文件修改方式：
```
vim  /home/orin/autoware/src/autoware-distributed-parallel-humble/src/launch/launch/param/node_config.param.yaml  

remote_info：目标orin的ip地址，用户名，端口号，以及目标启动节点名称
env: 目标orin的环境变量设置文件
flag: 若设置为False，表示不启动分布式，若为True，则表示启动分布式

```

### 5.2 编译launch文件
```
colcon build --continue-on-error --packages-select launch --cmake-args -DCMAKE_BUILD_TYPE=Release 
```
如果仿真出现没车模型的情况，则删除掉build/launch 和 install/launch两个文件夹

### 5.3 autoware启动


### 5.4 关闭进程
```
ps -ef | grep ros| grep -v grep | awk '{print $2}' | xargs kill -9
```

## 6.其他

### 6.1 AWSIM循环播放：loop_sim软件包

```
cd ./src/autoware-distributed-parallel-humble/
colcon build --packages-select loop_sim
source ./install/setup.bash
ros2 run loop_sim listener
```

### 6.2 杀死进程




