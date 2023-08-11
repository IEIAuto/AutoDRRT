# 自动驾驶计算框架AutoDRRT
AutoDRRT是一个自动驾驶框架，基于[Autoware开源框架](https://github.com/autowarefoundation/autoware/tree/main)开发，针对EIS400车载域控制器进行了针对性优化，提升了框架的实时性、可分布性及容错性。提供了一系列工具来方便用户更轻松地使用这些新增特性。该框架基于机器人操作系统2（ROS2）。它包含了从定位和目标检测到路径规划和控制的所有必要功能，并努力降低入门门槛，旨在让尽可能多的个人和组织参与到自动驾驶技术的开放创新中。
![avatar](./docs/imgs/Architecture_Diagram.png)

## 安装

### 运行平台

AutoDRRT的目标平台如下所列。在未来版本的AutoDRRT中可能会有所变化。

AutoDRRT基金会仅在下面列出的平台上提供支持，其他平台不予支持。

### 平台架构

amd64

arm64

### 推荐平台

EIS400

ORIN



### 最小系统要求

8核 CPU

16GB RAM

[可选] NVIDIA GPU (4GB RAM)

### 安装说明

#### 1.使用 DOCKER 安装

Docker可以确保项目中的所有开发者拥有一致的开发环境。推荐初学者、临时用户以及不熟悉Ubuntu使用。

[安装方法](./docs/cn/docker_Installation.md)

#### 2.使用 源码 安装

##### 环境需求

Ubuntu 22.04

[安装方法](./docs/cn/source_Installation.md)


### 使用说明

[使用说明](./docs/cn/tutorials.md)

### 沟通交流

[邮箱](AutoDRRT@ieisystem.com)