## Autoware Adaptation for the Horizon Journey 6 Platform
AutoDRRT has achieved deep native support for the domestic Horizon Robotics Journey 6 (J6) platform, establishing a complete link from the heterogeneous computing power of the underlying chip to the upper-layer general software stack for autonomous vehicle scenarios. It has completed full-stack native adaptation of ROS + Autoware.ai and ROS2 + Autoware.universe, becoming the industry's first open-source autonomous driving framework adapted to this platform. Based on the J6 domain controller + AutoDRRT, customers can directly implement "out-of-the-box" solutions and quickly verify them on the J6 domain controller. 

A complete cross-compilation environment has been built for the J6 platform, enabling rapid application migration and adaptation. The currently supported ROS and Autoware versions are as follows:

## 📋 Key Advantages

- **One-Stop Environment**: Pre-configured cross-compilation toolchain, significantly reducing setup complexity.
- **Multi-Version Compatibility**: Deeply integrated with mainstream autonomous driving software stacks.

## 📦 Supported Software Versions

The environment currently provides robust support for multiple generations of ROS and Autoware to meet diverse development needs:

### 1. ROS (Robot Operating System)

We support both ROS 1 and ROS 2 distributions, ensuring compatibility with various middleware requirements.

| Category | Distribution |
|----------|-------------|
| ROS 2 | Humble | 
| ROS 1 | Noetic |
| ROS 1 | Melodic|

### 2. Autoware

The two primary architectures of the Autoware project are fully supported:

- **Autoware.Universe**: The latest modular architecture based on ROS 2, optimized for advanced autonomous driving deployment and NPU acceleration on the J6 platform.
- **Autoware.AI**: The classic, battle-tested ROS 1-based stack. It serve as a reliable foundation for standard autonomous driving applications and reference implementations.


