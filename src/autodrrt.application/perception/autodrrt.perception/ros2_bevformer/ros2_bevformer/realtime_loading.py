
import os
import mmcv
import numpy as np
import cv2
from cv_bridge import CvBridge
from PIL import Image as Im
import torchvision
import torch
import warnings
from abc import abstractmethod
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from typing import Any, Dict
from sensor_msgs.msg import PointCloud2, PointField
import struct
import math
import random
from pyquaternion import Quaternion
import pickle

bridge = CvBridge() 

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = '>' if is_bigendian else '<'

    offset = 0
    for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
        if offset < field.offset:
            fmt += 'x' * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt    += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    assert isinstance(cloud, PointCloud2), 'cloud is not a sensor_msgs.msg.PointCloud2'
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step

##################################################################################
#                                IMAGE CONVERT                                   #
##################################################################################

def load_images(f_msg, fr_msg, fl_msg, b_msg, br_msg, bl_msg):
    f_cv2 = bridge.imgmsg_to_cv2(f_msg, 'bgr8')
    fr_cv2 = bridge.imgmsg_to_cv2(fr_msg, 'bgr8')
    fl_cv2 = bridge.imgmsg_to_cv2(fl_msg, 'bgr8')
    b_cv2 = bridge.imgmsg_to_cv2(b_msg, 'bgr8')
    br_cv2 = bridge.imgmsg_to_cv2(br_msg, 'bgr8')
    bl_cv2 = bridge.imgmsg_to_cv2(bl_msg, 'bgr8')
        
    color_coverted_f = cv2.cvtColor(f_cv2, cv2.COLOR_BGR2RGB)
    color_coverted_fr = cv2.cvtColor(fr_cv2, cv2.COLOR_BGR2RGB)
    color_coverted_fl = cv2.cvtColor(fl_cv2, cv2.COLOR_BGR2RGB)
    color_coverted_b = cv2.cvtColor(b_cv2, cv2.COLOR_BGR2RGB)
    color_coverted_br = cv2.cvtColor(br_cv2, cv2.COLOR_BGR2RGB)
    color_coverted_bl = cv2.cvtColor(bl_cv2, cv2.COLOR_BGR2RGB)

    pil_image_f=Im.fromarray(color_coverted_f)
    pil_image_fr=Im.fromarray(color_coverted_fr)
    pil_image_fl=Im.fromarray(color_coverted_fl)
    pil_image_b=Im.fromarray(color_coverted_b)
    pil_image_br=Im.fromarray(color_coverted_br)
    pil_image_bl=Im.fromarray(color_coverted_bl)

    pil_img_ls=[pil_image_f, pil_image_fr, pil_image_fl, pil_image_b, pil_image_br, pil_image_bl]

    # results=OrderedDict()
    results=dict()
    results["img"] = pil_img_ls

    return results

def sample_augmentation(results):
    is_train=False
    rand_flip=False
    W, H = [1600, 900]
    fH, fW = [256, 704]
    resize_lim=[0.48, 0.48]
    bot_pct_lim=[0.0, 0.0]
    rot_lim=[0.0, 0.0]
    
    if is_train:
        # resize = np.random.uniform(*resize_lim)
        # resize_dims = (int(W * resize), int(H * resize))
        # newW, newH = resize_dims
        # crop_h = int((1 - np.random.uniform(*bot_pct_lim)) * newH) - fH
        # crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        # crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        # flip = False
        # if rand_flip and np.random.choice([0, 1]):
        #     flip = True
        # rotate = np.random.uniform(rot_lim)
        print('USELESS')
    else:
        resize = np.mean(resize_lim)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.mean(bot_pct_lim)) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate = 0
    return resize, resize_dims, crop, flip, rotate

def img_transform(
    img, rotation, translation, resize, resize_dims, crop, flip, rotate
):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Im.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    rotation *= resize
    translation -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b
    theta = rotate / 180 * np.pi
    A = torch.Tensor(
        [
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)],
        ]
    )
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    rotation = A.matmul(rotation)
    translation = A.matmul(translation) + b

    return img, rotation, translation

def img_augmentation(data: Dict[str, Any]) -> Dict[str, Any]:
    imgs = data["img"]
    # print('=================IN CHECK=================')
    # print(data)
    # print('=================IN CHECK=================')
    new_imgs = []
    transforms = []
    for img in imgs:
        resize, resize_dims, crop, flip, rotate = sample_augmentation(data)
        post_rot = torch.eye(2)
        post_tran = torch.zeros(2)
        new_img, rotation, translation = img_transform(
            img,
            post_rot,
            post_tran,
            resize=resize,
            resize_dims=resize_dims,
            crop=crop,
            flip=flip,
            rotate=rotate,
        )
        transform = torch.eye(4)
        transform[:2, :2] = rotation
        transform[:2, 3] = translation
        new_imgs.append(new_img)
        transforms.append(transform.numpy())
    data["img"] = new_imgs
    # update the calibration matrices
    data["img_aug_matrix"] = transforms
    # print('=================OUT CHECK=================')
    # print(data)
    # print('=================OUT CHECK=================')
    return data

def img_normalize(data):
    mean= [0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]

    compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
            )
    
    # print('=================IN CHECK=================')
    # print(data)
    # print('=================IN CHECK=================')
    #ORIGIGNAL WAY
    data["img"] = [compose(img) for img in data["img"]]
    data["img_norm_cfg"] = dict(mean=mean, std=std)

    #MY WAY
    img0_tensor=data["img"][0]
    img1_tensor=data["img"][1]
    img2_tensor=data["img"][2]
    img3_tensor=data["img"][3]
    img4_tensor=data["img"][4]
    img5_tensor=data["img"][5]

    img_stack_tensor=torch.stack((img0_tensor, img1_tensor, img2_tensor, img3_tensor, img4_tensor, img5_tensor))
    format_ls=[]
    format_ls.append(img_stack_tensor)
    data["img"]=format_ls
    data["img"]==img_stack_tensor

    # print('=================OUT CHECK=================')
    # print(data)
    # print('=================OUT CHECK=================')

    return data

def collect3d_for_img(results):
    """Call function to collect keys in results. The keys in ``meta_keys``
    will be converted to :obj:`mmcv.DataContainer`.

    Args:
        results (dict): Result dict contains the data to collect.

    Returns:
        dict: The result dict contains the following keys
            - keys in ``self.keys``
            - ``metas``
    """
    data = {}

    keys=['img', 'points']
    meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix']
    # meta_lis_keys=('filename', 'timestamp', 'ori_shape', 'img_shape', 'lidar2image', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'token', 'pcd_scale_factor', 'pcd_rotation', 'lidar_path', 'transformation_3d_flow')
    for key in keys:
        if key not in meta_keys:
            # data[key] = results[key]
            if key=='img':
                data['img']=results['img']
            # elif key=='points':
            #     data['points']=results['points']
    # print('=================OUT CHECK=================')
    # print(data)
    # print('=================OUT CHECK=================')
    return data

##################################################################################
#                                LIDAR CONVERT                                   #
##################################################################################

class BasePoints:
    """Base class for Points.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that
            # does not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, points_dim)).to(
                dtype=torch.float32, device=device
            )
        assert tensor.dim() == 2 and tensor.size(-1) == points_dim, tensor.size()

        self.tensor = tensor
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        # after modification, axis=2 corresponds to z
        self.rotation_axis = 2

    @property
    def coord(self):
        """torch.Tensor: Coordinates of each point with size (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor):
        """Set the coordinates of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        self.tensor[:, :3] = tensor

    @property
    def height(self):
        """torch.Tensor: A vector with height of each point."""
        if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["height"]]
        else:
            return None

    @height.setter
    def height(self, tensor):
        """Set the height of each point."""
        try:
            tensor = tensor.reshape(self.shape[0])
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "height" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["height"]] = tensor
        else:
            # add height attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor.unsqueeze(1)], dim=1)
            self.attribute_dims.update(dict(height=attr_dim))
            self.points_dim += 1

    @property
    def color(self):
        """torch.Tensor: A vector with color of each point."""
        if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
            return self.tensor[:, self.attribute_dims["color"]]
        else:
            return None

    @color.setter
    def color(self, tensor):
        """Set the color of each point."""
        try:
            tensor = tensor.reshape(self.shape[0], 3)
        except (RuntimeError, ValueError):  # for torch.Tensor and np.ndarray
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if tensor.max() >= 256 or tensor.min() < 0:
            warnings.warn("point got color value beyond [0, 255]")
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        if self.attribute_dims is not None and "color" in self.attribute_dims.keys():
            self.tensor[:, self.attribute_dims["color"]] = tensor
        else:
            # add color attribute
            if self.attribute_dims is None:
                self.attribute_dims = dict()
            attr_dim = self.shape[1]
            self.tensor = torch.cat([self.tensor, tensor], dim=1)
            self.attribute_dims.update(
                dict(color=[attr_dim, attr_dim + 1, attr_dim + 2])
            )
            self.points_dim += 3

    @property
    def shape(self):
        """torch.Shape: Shape of points."""
        return self.tensor.shape

    def shuffle(self):
        """Shuffle the points.

        Returns:
            torch.Tensor: The shuffled index.
        """
        idx = torch.randperm(self.__len__(), device=self.tensor.device)
        self.tensor = self.tensor[idx]
        return idx

    def rotate(self, rotation, axis=None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation (float, np.ndarray, torch.Tensor): Rotation matrix
                or angle.
            axis (int): Axis to rotate at. Defaults to None.
        """
        if not isinstance(rotation, torch.Tensor):
            rotation = self.tensor.new_tensor(rotation)
        assert (
            rotation.shape == torch.Size([3, 3]) or rotation.numel() == 1
        ), f"invalid rotation shape {rotation.shape}"

        if axis is None:
            axis = self.rotation_axis

        if rotation.numel() == 1:
            rot_sin = torch.sin(rotation)
            rot_cos = torch.cos(rotation)
            if axis == 1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]]
                )
            elif axis == 2 or axis == -1:
                rot_mat_T = rotation.new_tensor(
                    [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]]
                )
            elif axis == 0:
                rot_mat_T = rotation.new_tensor(
                    [[0, rot_cos, -rot_sin], [0, rot_sin, rot_cos], [1, 0, 0]]
                )
            else:
                raise ValueError("axis should in range")
            rot_mat_T = rot_mat_T.T
        elif rotation.numel() == 9:
            rot_mat_T = rotation
        else:
            raise NotImplementedError
        self.tensor[:, :3] = self.tensor[:, :3] @ rot_mat_T

        return rot_mat_T

    @abstractmethod
    def flip(self, bev_direction="horizontal"):
        """Flip the points in BEV along given BEV direction."""
        pass

    def translate(self, trans_vector):
        """Translate points with the given translation vector.

        Args:
            trans_vector (np.ndarray, torch.Tensor): Translation
                vector of size 3 or nx3.
        """
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        trans_vector = trans_vector.squeeze(0)
        if trans_vector.dim() == 1:
            assert trans_vector.shape[0] == 3
        elif trans_vector.dim() == 2:
            assert (
                trans_vector.shape[0] == self.tensor.shape[0]
                and trans_vector.shape[1] == 3
            )
        else:
            raise NotImplementedError(
                f"Unsupported translation vector of shape {trans_vector.shape}"
            )
        self.tensor[:, :3] += trans_vector

    def in_range_3d(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                (x_min, y_min, z_min, x_max, y_max, z_max)

        Note:
            In the original implementation of SECOND, checking whether
            a box in the range checks whether the points are in a convex
            polygon, we try to reduce the burden for simpler cases.

        Returns:
            torch.Tensor: A binary vector indicating whether each point is \
                inside the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 2] > point_range[2])
            & (self.tensor[:, 0] < point_range[3])
            & (self.tensor[:, 1] < point_range[4])
            & (self.tensor[:, 2] < point_range[5])
        )
        return in_range_flags

    @abstractmethod
    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside \
                the reference range.
        """
        pass

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`CoordMode`): The target Box mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted box of the same type \
                in the `dst` mode.
        """
        pass

    def scale(self, scale_factor):
        """Scale the points with horizontal and vertical scaling factors.

        Args:
            scale_factors (float): Scale factors to scale the points.
        """
        self.tensor[:, :3] *= scale_factor

    def __getitem__(self, item):
        """
        Note:
            The following usage are allowed:
            1. `new_points = points[3]`:
                return a `Points` that contains only one point.
            2. `new_points = points[2:10]`:
                return a slice of points.
            3. `new_points = points[vector]`:
                where vector is a torch.BoolTensor with `length = len(points)`.
                Nonzero elements in the vector will be selected.
            4. `new_points = points[3:11, vector]`:
                return a slice of points and attribute dims.
            5. `new_points = points[4:12, 2]`:
                return a slice of points with single attribute.
            Note that the returned Points might share storage with this Points,
            subject to Pytorch's indexing semantics.

        Returns:
            :obj:`BasePoints`: A new object of  \
                :class:`BasePoints` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(
                self.tensor[item].view(1, -1),
                points_dim=self.points_dim,
                attribute_dims=self.attribute_dims,
            )
        elif isinstance(item, tuple) and len(item) == 2:
            if isinstance(item[1], slice):
                start = 0 if item[1].start is None else item[1].start
                stop = self.tensor.shape[1] if item[1].stop is None else item[1].stop
                step = 1 if item[1].step is None else item[1].step
                item = list(item)
                item[1] = list(range(start, stop, step))
                item = tuple(item)
            elif isinstance(item[1], int):
                item = list(item)
                item[1] = [item[1]]
                item = tuple(item)
            p = self.tensor[item[0], item[1]]

            keep_dims = list(
                set(item[1]).intersection(set(range(3, self.tensor.shape[1])))
            )
            if self.attribute_dims is not None:
                attribute_dims = self.attribute_dims.copy()
                for key in self.attribute_dims.keys():
                    cur_attribute_dims = attribute_dims[key]
                    if isinstance(cur_attribute_dims, int):
                        cur_attribute_dims = [cur_attribute_dims]
                    intersect_attr = list(
                        set(cur_attribute_dims).intersection(set(keep_dims))
                    )
                    if len(intersect_attr) == 1:
                        attribute_dims[key] = intersect_attr[0]
                    elif len(intersect_attr) > 1:
                        attribute_dims[key] = intersect_attr
                    else:
                        attribute_dims.pop(key)
            else:
                attribute_dims = None
        elif isinstance(item, (slice, np.ndarray, torch.Tensor)):
            p = self.tensor[item]
            attribute_dims = self.attribute_dims
        else:
            raise NotImplementedError(f"Invalid slice {item}!")

        assert (
            p.dim() == 2
        ), f"Indexing on Points with {item} failed to return a matrix!"
        return original_type(p, points_dim=p.shape[1], attribute_dims=attribute_dims)

    def __len__(self):
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, points_list):
        """Concatenate a list of Points into a single Points.

        Args:
            points_list (list[:obj:`BasePoints`]): List of points.

        Returns:
            :obj:`BasePoints`: The concatenated Points.
        """
        assert isinstance(points_list, (list, tuple))
        if len(points_list) == 0:
            return cls(torch.empty(0))
        assert all(isinstance(points, cls) for points in points_list)

        # use torch.cat (v.s. layers.cat)
        # so the returned points never share storage with input
        cat_points = cls(
            torch.cat([p.tensor for p in points_list], dim=0),
            points_dim=points_list[0].tensor.shape[1],
            attribute_dims=points_list[0].attribute_dims,
        )
        return cat_points

    def to(self, device):
        """Convert current points to a specific device.

        Args:
            device (str | :obj:`torch.device`): The name of the device.

        Returns:
            :obj:`BasePoints`: A new boxes object on the \
                specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims,
        )

    def clone(self):
        """Clone the Points.

        Returns:
            :obj:`BasePoints`: Box object with the same properties \
                as self.
        """
        original_type = type(self)
        return original_type(
            self.tensor.clone(),
            points_dim=self.points_dim,
            attribute_dims=self.attribute_dims,
        )

    @property
    def device(self):
        """str: The device of the points are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a point as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A point of shape (4,).
        """
        yield from self.tensor

    def new_point(self, data):
        """Create a new point object with data.

        The new point and its tensor has the similar properties \
            as self and self.tensor, respectively.

        Args:
            data (torch.Tensor | numpy.array | list): Data to be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``, \
                the object's other properties are similar to ``self``.
        """
        new_tensor = (
            self.tensor.new_tensor(data)
            if not isinstance(data, torch.Tensor)
            else data.to(self.device)
        )
        original_type = type(self)
        return original_type(
            new_tensor, points_dim=self.points_dim, attribute_dims=self.attribute_dims
        )

class LiDARPoints(BasePoints):
    """Points of instances in LIDAR coordinates.

    Args:
        tensor (torch.Tensor | np.ndarray | list): a N x points_dim matrix.
        points_dim (int): Number of the dimension of a point.
            Each row is (x, y, z). Default to 3.
        attribute_dims (dict): Dictionary to indicate the meaning of extra
            dimension. Default to None.

    Attributes:
        tensor (torch.Tensor): Float matrix of N x points_dim.
        points_dim (int): Integer indicating the dimension of a point.
            Each row is (x, y, z, ...).
        attribute_dims (bool): Dictionary to indicate the meaning of extra
            dimension. Default to None.
        rotation_axis (int): Default rotation axis for points rotation.
    """

    def __init__(self, tensor, points_dim=3, attribute_dims=None):
        super(LiDARPoints, self).__init__(
            tensor, points_dim=points_dim, attribute_dims=attribute_dims
        )
        self.rotation_axis = 2

    def flip(self, bev_direction="horizontal"):
        """Flip the boxes in BEV along given BEV direction."""
        if bev_direction == "horizontal":
            self.tensor[:, 1] = -self.tensor[:, 1]
        elif bev_direction == "vertical":
            self.tensor[:, 0] = -self.tensor[:, 0]

    def in_range_bev(self, point_range):
        """Check whether the points are in the given range.

        Args:
            point_range (list | torch.Tensor): The range of point
                in order of (x_min, y_min, x_max, y_max).

        Returns:
            torch.Tensor: Indicating whether each point is inside \
                the reference range.
        """
        in_range_flags = (
            (self.tensor[:, 0] > point_range[0])
            & (self.tensor[:, 1] > point_range[1])
            & (self.tensor[:, 0] < point_range[2])
            & (self.tensor[:, 1] < point_range[3])
        )
        return in_range_flags

    def convert_to(self, dst, rt_mat=None):
        """Convert self to ``dst`` mode.

        Args:
            dst (:obj:`CoordMode`): The target Point mode.
            rt_mat (np.ndarray | torch.Tensor): The rotation and translation
                matrix between different coordinates. Defaults to None.
                The conversion from `src` coordinates to `dst` coordinates
                usually comes along the change of sensors, e.g., from camera
                to LiDAR. This requires a transformation matrix.

        Returns:
            :obj:`BasePoints`: The converted point of the same type \
                in the `dst` mode.
        """
        from mmdet3d.core.bbox import Coord3DMode

        return Coord3DMode.convert_point(point=self, src=Coord3DMode.LIDAR, dst=dst, rt_mat=rt_mat)

def load_augmented_point_cloud(path, virtual=False, reduce_beams=32):
    # NOTE: following Tianwei's implementation, it is hard coded for nuScenes
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    # NOTE: path definition different from Tianwei's implementation.
    tokens = path.split("/")
    vp_dir = "_VIRTUAL" if reduce_beams == 32 else f"_VIRTUAL_{reduce_beams}BEAMS"
    seg_path = os.path.join(
        *tokens[:-3],
        "virtual_points",
        tokens[-3],
        tokens[-2] + vp_dir,
        tokens[-1] + ".pkl.npy",
    )
    assert os.path.exists(seg_path)
    data_dict = np.load(seg_path, allow_pickle=True).item()

    virtual_points1 = data_dict["real_points"]
    # NOTE: add zero reflectance to virtual points instead of removing them from real points
    virtual_points2 = np.concatenate(
        [
            data_dict["virtual_points"][:, :3],
            np.zeros([data_dict["virtual_points"].shape[0], 1]),
            data_dict["virtual_points"][:, 3:],
        ],
        axis=-1,
    )

    points = np.concatenate(
        [
            points,
            np.ones([points.shape[0], virtual_points1.shape[1] - points.shape[1] + 1]),
        ],
        axis=1,
    )
    virtual_points1 = np.concatenate(
        [virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1
    )
    # note: this part is different from Tianwei's implementation, we don't have duplicate foreground real points.
    if len(data_dict["real_points_indice"]) > 0:
        points[data_dict["real_points_indice"]] = virtual_points1
    if virtual:
        virtual_points2 = np.concatenate(
            [virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1
        )
        points = np.concatenate([points, virtual_points2], axis=0).astype(np.float32)
    return points

def _load_points(lidar_path):
    """Private function to load point clouds data.

    Args:
        lidar_path (str): Filename of point clouds data.

    Returns:
        np.ndarray: An array containing point clouds data.
    """
    load_augmented=None
    reduce_beams=32

    mmcv.check_file_exist(lidar_path)
    if load_augmented:
        assert load_augmented in ["pointpainting", "mvp"]
        virtual = load_augmented == "mvp"
        points = load_augmented_point_cloud(
            lidar_path, virtual=virtual, reduce_beams=reduce_beams
        )
    elif lidar_path.endswith(".npy"):
        points = np.load(lidar_path)
    else:
        points = np.fromfile(lidar_path, dtype=np.float32)

    return points

def reduce_LiDAR_beams(pts, reduce_beams_to=32):
    # print(pts.size())
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    sine_theta = pts[:, 2] / radius
    # [-pi/2, pi/2]
    theta = torch.asin(sine_theta)
    phi = torch.atan2(pts[:, 1], pts[:, 0])

    top_ang = 0.1862
    down_ang = -0.5353

    beam_range = torch.zeros(32)
    beam_range[0] = top_ang
    beam_range[31] = down_ang

    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    # beam_range = [1, 0.18, 0.15, 0.13, 0.11, 0.085, 0.065, 0.03, 0.01, -0.01, -0.03, -0.055, -0.08, -0.105, -0.13, -0.155, -0.18, -0.205, -0.228, -0.251, -0.275,
    #                -0.295, -0.32, -0.34, -0.36, -0.38, -0.40, -0.425, -0.45, -0.47, -0.49, -0.52, -0.54]

    num_pts, _ = pts.size()
    mask = torch.zeros(num_pts)
    if reduce_beams_to == 16:
        for id in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    elif reduce_beams_to == 4:
        for id in [7, 9, 11, 13]:
            beam_mask = (theta < (beam_range[id - 1] - 0.012)) * (
                theta > (beam_range[id] - 0.012)
            )
            mask = mask + beam_mask
        mask = mask.bool()
    # [?] pick the 14th beam
    elif reduce_beams_to == 1:
        chosen_beam_id = 9
        mask = (theta < (beam_range[chosen_beam_id - 1] - 0.012)) * (
            theta > (beam_range[chosen_beam_id] - 0.012)
        )
    else:
        raise NotImplementedError
    # points = copy.copy(pts)
    points = pts[mask]
    # print(points.size())
    return points.numpy()

def get_points_type(points_type):
    """Get the class of points according to coordinate type.

    Args:
        points_type (str): The type of points coordinate.
            The valid value are "CAMERA", "LIDAR", or "DEPTH".

    Returns:
        class: Points type.
    """
    if points_type == "CAMERA":
        # points_cls = CameraPoints
        print('USELESS')
    elif points_type == "LIDAR":
        points_cls = LiDARPoints
    elif points_type == "DEPTH":
        print('USELESS')
        # points_cls = DepthPoints
    else:
        raise ValueError(
            'Only "points_type" of "CAMERA", "LIDAR", or "DEPTH"'
            f" are supported, got {points_type}"
        )

    return points_cls

def point_from_file(lidar_path):
    #lidar dimension for loading
    load_dim=5
    #lidar dimension for using
    use_dim=[0, 1, 2, 3, 4]
    #lidar channel
    reduce_beams=32
    #Whether to use shifted height. Defaults to False.
    shift_height=False
    #Whether to use color features. Defaults to False.
    use_color=False
    #coord_type
    coord_type='LIDAR'

    points = _load_points(lidar_path)
    # points = np.fromfile(lidar_path, dtype=np.float32) 사실상 윗줄은 이거랑 동일함
    # (173440,)으로 1행 [ , , , ,]의 형태
    # 포인트의 차원 [x, y, z, i ,t]를 옆으로 쭉 늘여붙여서 ndarray를 만들어주면 되는 것
                    
    points = points.reshape(-1, load_dim)
    # TODO: make it more general
    if reduce_beams and reduce_beams < 32:
        points = reduce_LiDAR_beams(points, reduce_beams)
    points = points[:, use_dim]
    attribute_dims = None

    if shift_height:
        floor_height = np.percentile(points[:, 2], 0.99)
        height = points[:, 2] - floor_height
        points = np.concatenate(
            [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
        )
        attribute_dims = dict(height=3)

    if use_color:
        assert len(use_dim) >= 6
        if attribute_dims is None:
            attribute_dims = dict()
        attribute_dims.update(
            dict(
                color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]
            )
        )

    points_class = get_points_type(coord_type)
    points = points_class(
        points, points_dim=points.shape[-1], attribute_dims=attribute_dims
    )
    # data=OrderedDict()
    data=dict()
    data["points"] = points

    return data

def load_points(points):
    #lidar dimension for loading
    load_dim=5
    # load_dim=4    #for normal PointCloud dimension
    #lidar dimension for using
    use_dim=[0, 1, 2, 3, 4]
    # use_dim=[0, 1, 2, 3] #for normal PointCloud dimension
    #lidar channel
    reduce_beams=32
    #Whether to use shifted height. Defaults to False.
    shift_height=False
    #Whether to use color features. Defaults to False.
    use_color=False
    #coord_type
    coord_type='LIDAR'

    points = points.reshape(-1, load_dim)
    # TODO: make it more general
    if reduce_beams and reduce_beams < 32:
        points = reduce_LiDAR_beams(points, reduce_beams)
    points = points[:, use_dim]
    attribute_dims = None

    if shift_height:
        floor_height = np.percentile(points[:, 2], 0.99)
        height = points[:, 2] - floor_height
        points = np.concatenate(
            [points[:, :3], np.expand_dims(height, 1), points[:, 3:]], 1
        )
        attribute_dims = dict(height=3)

    if use_color:
        assert len(use_dim) >= 6
        if attribute_dims is None:
            attribute_dims = dict()
        attribute_dims.update(
            dict(
                color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]
            )
        )

    points_class = get_points_type(coord_type)
    points = points_class(
        points, points_dim=points.shape[-1], attribute_dims=attribute_dims
    )
    # data=OrderedDict()
    data=dict()
    data["points"] = points

    return data

def global_rot_trans(data):
        is_train=None
        resize_lim=[0.48, 0.48]
        rot_lim=[0.0, 0.0]
        trans_lim=0.0
        # print('=================IN CHECK=================')
        # print(data)
        # print('=================IN CHECK=================')
        transform = np.eye(4).astype(np.float32)

        if is_train:
            scale = random.uniform(resize_lim)
            theta = random.uniform(rot_lim)
            translation = np.array([random.normal(0, trans_lim) for i in range(3)])
            rotation = np.eye(3)

            if "points" in data:
                data["points"].rotate(-theta)
                data["points"].translate(translation)
                data["points"].scale(scale)

            if "radar" in data:
                data["radar"].rotate(-theta)
                data["radar"].translate(translation)
                data["radar"].scale(scale)

            gt_boxes = data["gt_bboxes_3d"]
            rotation = rotation @ gt_boxes.rotate(theta).numpy()
            gt_boxes.translate(translation)
            gt_boxes.scale(scale)
            data["gt_bboxes_3d"] = gt_boxes

            transform[:3, :3] = rotation.T * scale
            transform[:3, 3] = translation * scale

        data["lidar_aug_matrix"] = transform
        # print('=================OUT CHECK=================')
        # print(data)
        # print('=================OUT CHECK=================')
        return data    

def point_range_filter(data):
    point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
    pcd_range=np.array(point_cloud_range, dtype=np.float32)
    points = data["points"]
    points_mask = points.in_range_3d(pcd_range)
    clean_points = points[points_mask]
    data["points"] = clean_points

    if "radar" in data:
        # radar = data["radar"]
        # # radar_mask = radar.in_range_3d(self.pcd_range)
        # radar_mask = radar.in_range_bev([-55.0, -55.0, 55.0, 55.0])
        # clean_radar = radar[radar_mask]
        # data["radar"] = clean_radar
        print('USELESS')
    # print('=================OUT CHECK=================')
    # print(data)
    # print('=================OUT CHECK=================')    
    return data

def collect3d_for_point(results):
    """Call function to collect keys in results. The keys in ``meta_keys``
    will be converted to :obj:`mmcv.DataContainer`.

    Args:
        results (dict): Result dict contains the data to collect.

    Returns:
        dict: The result dict contains the following keys
            - keys in ``self.keys``
            - ``metas``
    """
    data = {}
    keys=['img', 'points']
    meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix']
    # meta_lis_keys=('filename', 'timestamp', 'ori_shape', 'img_shape', 'lidar2image', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'token', 'pcd_scale_factor', 'pcd_rotation', 'lidar_path', 'transformation_3d_flow')
    for key in keys:
        if key not in meta_keys:
            # data[key] = results[key]
            if key=='points':
                data['points']=results['points']

            # elif key=='points':
            #     data['points']=results['points']
    return data

def collect3d(results):
    data = {}
    keys=['img', 'points']
    meta_keys=['camera_intrinsics', 'camera2ego', 'lidar2ego', 'lidar2camera', 'lidar2image', 'camera2lidar', 'img_aug_matrix', 'lidar_aug_matrix']
    meta_lis_keys=('filename', 'timestamp', 'ori_shape', 'img_shape', 'lidar2image', 'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'token', 'pcd_scale_factor', 'pcd_rotation', 'lidar_path', 'transformation_3d_flow')

    # print('=================IN CHECK=================')
    # print(results)
    # print('=================IN CHECK=================')
    
    for key in keys:
        if key not in meta_keys:
            data[key] = results[key]
    for key in meta_keys:
        if key in results:
            val = np.array(results[key])
            if isinstance(results[key], list):
                data[key] = DC(to_tensor(val), stack=True)
            else:
                data[key] = DC(to_tensor(val), stack=True, pad_dims=1)

    metas = {}
    for key in meta_lis_keys:
        if key in results:
            metas[key] = results[key]

    data["metas"] = DC(metas, cpu_only=True)
    # print('=================OUT CHECK=================')
    # print(data)
    # print('=================OUT CHECK=================')
    return data 


def default_format_bundle(results):
        if "points" in results:
            assert isinstance(results["points"], BasePoints)
            results["points"] = DC([[results["points"].tensor]])

        # if "radar" in results:
        #     results["radar"] = DC(results["radar"].tensor)

        # for key in ["voxels", "coors", "voxel_centers", "num_points"]:
        #     if key not in results:
        #         continue
        #     results[key] = DC(to_tensor(results[key]), stack=False)

        if "img" in results:
            
            results["img"] = DC([torch.stack(results["img"])], stack=True)

        for key in [
            "proposals",
            "gt_bboxes",
            "gt_bboxes_ignore",
            "gt_labels",
            "gt_labels_3d",
            "attr_labels",
            "centers2d",
            "depths",
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))

        return results

def gt_depth(data):
    ##################################################################################
    #                                GET_DATA_INFO                                   #
    ##################################################################################
    open_path='/home/nvidia/BEVfusion/data/nuscenes/nuscenes_infos_test.pkl'
    with open(open_path, 'rb') as f:
        info = pickle.load(f)
            
    info=info['infos'][0]

    # lidar to ego transform
    lidar2ego = np.eye(4).astype(np.float32)
    lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    lidar2ego[:3, 3] = info["lidar2ego_translation"]
    data["lidar2ego"] = lidar2ego


    data["image_paths"] = []
    data["lidar2camera"] = []
    data["lidar2image"] = []
    data["camera2ego"] = []
    data["camera_intrinsics"] = []
    data["camera2lidar"] = []

    for _, camera_info in info["cams"].items():
        # data["image_paths"].append(camera_info["data_path"])

        # lidar to camera transform
        lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
        lidar2camera_t = (
            camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
        )
        lidar2camera_rt = np.eye(4).astype(np.float32)
        lidar2camera_rt[:3, :3] = lidar2camera_r.T
        lidar2camera_rt[3, :3] = -lidar2camera_t
        data["lidar2camera"].append(lidar2camera_rt.T)

        # camera intrinsics
        camera_intrinsics = np.eye(4).astype(np.float32)
        # print(camera_info)
        camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
        data["camera_intrinsics"].append(camera_intrinsics)

        # lidar to image transform
        lidar2image = camera_intrinsics @ lidar2camera_rt.T
        data["lidar2image"].append(lidar2image)

        # camera to ego transform
        camera2ego = np.eye(4).astype(np.float32)
        camera2ego[:3, :3] = Quaternion(
            camera_info["sensor2ego_rotation"]
        ).rotation_matrix
        camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
        data["camera2ego"].append(camera2ego)

        # camera to lidar transform
        camera2lidar = np.eye(4).astype(np.float32)
        camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
        camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
        data["camera2lidar"].append(camera2lidar)

    ##################################################################################
    #                                GET_DATA_INFO                                   #
    ##################################################################################

    # keyframe_only=True
    keyframe_only=False

    format_points=data['points'].data[0][0]
    format_img=data['img'].data[0].squeeze()

    # sensor2ego = data['camera2ego'].data
    # cam_intrinsic = data['camera_intrinsics'].data 
    # img_aug_matrix = data['img_aug_matrix'].data 
    # bev_aug_matrix = data['lidar_aug_matrix'].data
    # lidar2ego = data['lidar2ego'].data 
    # camera2lidar = data['camera2lidar'].data
    # lidar2image = data['lidar2image'].data

    sensor2ego = data['camera2ego']
    cam_intrinsic = data['camera_intrinsics']
    img_aug_matrix = data['img_aug_matrix']
    bev_aug_matrix = data['lidar_aug_matrix']
    lidar2ego = data['lidar2ego']
    camera2lidar = data['camera2lidar']
    lidar2image = data['lidar2image']

    # rots = sensor2ego[..., :3, :3]
    # trans = sensor2ego[..., :3, 3]
    # intrins = cam_intrinsic[..., :3, :3]
    # post_rots = img_aug_matrix[..., :3, :3]
    # post_trans = img_aug_matrix[..., :3, 3]
    # lidar2ego_rots = lidar2ego[..., :3, :3]
    # lidar2ego_trans = lidar2ego[..., :3, 3]
    # camera2lidar_rots = camera2lidar[..., :3, :3]
    # camera2lidar_trans = camera2lidar[..., :3, 3]

    # points = data['points'].data 
    # img = data['img'].data
    points=format_points

    img=format_img

    if keyframe_only:
        points = points[points[:, 4] == 0]
        
    # print('====================================')
    # print(points)
    # print('====================================')
    batch_size = len(points)
    depth = torch.zeros(img.shape[0], *img.shape[-2:]) #.to(points[0].device)

    # for b in range(batch_size):
    cur_coords = points[:, :3]

    # inverse aug
    cur_coords -= bev_aug_matrix.data[:3, 3]

    cur_coords = torch.inverse(bev_aug_matrix.data[:3, :3]).matmul(
        cur_coords.transpose(1, 0)
    )

    format_lidar2image=np.array( [ lidar2image[0], lidar2image[1],lidar2image[2],lidar2image[3],lidar2image[4],lidar2image[5] ] )
    lidar2image = torch.from_numpy(format_lidar2image)

    # lidar2image
    cur_coords = lidar2image[:, :3, :3].matmul(cur_coords)
    cur_coords += lidar2image[:, :3, 3].reshape(-1, 3, 1)

    # get 2d coords
    dist = cur_coords[:, 2, :]
    cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
    cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

    # imgaug
    cur_coords = img_aug_matrix.data[:, :3, :3].matmul(cur_coords)
    cur_coords += img_aug_matrix.data[:, :3, 3].reshape(-1, 3, 1)
    cur_coords = cur_coords[:, :2, :].transpose(1, 2)

    # normalize coords for grid sample
    cur_coords = cur_coords[..., [1, 0]]

    on_img = (
        (cur_coords[..., 0] < img.shape[2])
        & (cur_coords[..., 0] >= 0)
        & (cur_coords[..., 1] < img.shape[3])
        & (cur_coords[..., 1] >= 0)
    )
    for c in range(on_img.shape[0]):
        masked_coords = cur_coords[c, on_img[c]].long()
        masked_dist = dist[c, on_img[c]]
        depth[c, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

    data['depths'] = depth 
    # print('=================OUT CHECK=================')
    # print(depth)
    # print('=================OUT CHECK=================')
    return data