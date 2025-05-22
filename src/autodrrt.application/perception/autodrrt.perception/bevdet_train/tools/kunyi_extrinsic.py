import numpy as np

def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    通过欧拉角 (roll, pitch, yaw) 生成旋转矩阵。
    :param roll: 绕 X 轴旋转的角度 (弧度)
    :param pitch: 绕 Y 轴旋转的角度 (弧度)
    :param yaw: 绕 Z 轴旋转的角度 (弧度)
    :return: 3x3 的旋转矩阵
    """
    # 绕 X 轴旋转
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    # 绕 Y 轴旋转
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    # 绕 Z 轴旋转
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 总旋转矩阵
    R = Rz @ Ry @ Rx
    return R

def generate_extrinsic_matrix(x, y, z, roll, pitch, yaw):
    """
    根据平移向量和欧拉角生成外参矩阵。
    :param x: 平移向量的 x 分量
    :param y: 平移向量的 y 分量
    :param z: 平移向量的 z 分量
    :param roll: 绕 X 轴旋转的角度 (弧度)
    :param pitch: 绕 Y 轴旋转的角度 (弧度)
    :param yaw: 绕 Z 轴旋转的角度 (弧度)
    :return: 4x4 的外参矩阵
    """
    # 生成旋转矩阵
    R = rotation_matrix_from_euler(roll, pitch, yaw)

    # 平移向量
    t = np.array([x, y, z]).reshape(3, 1)

    # 构造齐次变换矩阵
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = R
    extrinsic_matrix[:3, 3] = t.flatten()

    return extrinsic_matrix



## camera2ego计算，请根据昆易的xyz roll pitch yaw进行计算

# 下面是昆易右前相机到ego的外参矩阵，即frontright2ego

# # 前
# x, y, z = 3.9, 0.0, 0.9  # 平移向量
# roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(0)

# 左前
x, y, z = 2.1, 1.1, 0.9  # 平移向量
roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(130)

# 右前
# x, y, z = 2.1, -1.0, 0.9  # 平移向量
# roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(-130)


# 后
# x, y, z = -1.05, 0.0, 0.9  # 平移向量
# roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(-180)


# 左后
# x, y, z = 0.2, 1.0, 0.9  # 平移向量
# roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(45)


# # 右后
# x, y, z = 0.2, -1.0, 0.9  # 平移向量
# roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(-45)

# x, y, z = 2.1, -1.0, 0.9  # 平移向量
# roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(-130)  # 欧拉角 (单位: 弧度)
# 生成外参矩阵
camera2ego = generate_extrinsic_matrix(x, y, z, roll, pitch, yaw)
print("外参矩阵:\n", camera2ego)

# lidar2ego计算，根据昆易的lidar的xyz roll pitch yaw进行计算

# 下面是昆易雷达到ego的外参矩阵，即lidar2ego
x, y, z = 1.0, 0.0, 1.8  # 平移向量
roll, pitch, yaw = np.radians(0), np.radians(0), np.radians(0)  # 欧拉角 (单位: 弧度)
# 生成外参矩阵
lidar2ego = generate_extrinsic_matrix(x, y, z, roll, pitch, yaw)
print("外参矩阵:\n", lidar2ego)


# 求相机到lidar的外参矩阵，即camera2lidar = ego2lidar * camera2ego
ego2lidar = np.linalg.inv(lidar2ego)
camera2lidar = ego2lidar @ camera2ego
print("相机到雷达的外参矩阵:\n", camera2lidar)

# 
camera2image = np.array([
    [0, -1,  0, 0],
    [0,  0, -1, 0],
    [1,  0,  0, 0],
    [0,  0,  0, 1]
])


image2camera =  np.linalg.inv(camera2image)

image2ego = np.dot(camera2ego, image2camera)

image2lidar = np.dot(np.linalg.inv(lidar2ego), image2ego)

print("图片到雷达的外参矩阵:\n", image2lidar)

formatted_matrix = ", ".join(map(str, image2lidar.reshape(-1)))
print("图片到雷达的外参矩阵（逗号分隔）:")
print(formatted_matrix)