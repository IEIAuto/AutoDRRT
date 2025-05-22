from typing import List, Optional, Tuple
import numpy as np
import matplotlib 
matplotlib.use('Agg')
import cv2
from math import sin, cos, radians


import copy

import torch

# OBJECT_PALETTE = (255, 158, 0)
OBJECT_PALETTE = (0, 255, 0)


def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points by angles according to axis.

    Args:
        points (torch.Tensor): Points of shape (N, M, 3).
        angles (torch.Tensor): Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. x,y,z Defaults to 0. 

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will \
            raise value error.

    Returns:
        torch.Tensor: Rotated points in shape (N, M, 3)
    """
    # print("================")
    # print(angles)
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        # rot_mat_T = torch.stack(
        #     [
        #         torch.stack([rot_cos, -rot_sin, zeros]),
        #         torch.stack([rot_sin, rot_cos, zeros]),
        #         torch.stack([zeros, zeros, ones]),
        #     ]
        # )
        rot_mat_T = torch.stack(
            [
                torch.stack([rot_cos, rot_sin, zeros]),  # 改为 rot_sin
                torch.stack([-rot_sin, rot_cos, zeros]),  # 改为 -rot_sin
                torch.stack([zeros, zeros, ones]),
            ]
        )
        # print(rot_mat_T)
    elif axis == 0:
        rot_mat_T = torch.stack(
            [
                torch.stack([zeros, rot_cos, -rot_sin]),
                torch.stack([zeros, rot_sin, rot_cos]),
                torch.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")

    return torch.einsum("aij,jka->aik", (points, rot_mat_T))

def tocorners(tensor):
    """torch.Tensor: Coordinates of corners of all the boxes
    in shape (N, 8, 3).

    Convert the boxes to corners in clockwise order, in form of
    ``(x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0)``

    .. code-block:: none

                                        up z
                        front x           ^
                                /            |
                            /             |
                (x1, y0, z1) + -----------  + (x1, y1, z1)
                            /|            / |
                            / |           /  |
            (x0, y0, z1) + ----------- +   + (x1, y1, z0)
                        |  /      .   |  /
                        | / origin    | /
        left y<-------- + ----------- + (x0, y1, z0)
            (x0, y0, z0)
    """
    # TODO: rotation_3d_in_axis function do not support
    #  empty tensor currently.
    dims = tensor[:, 3:6]
    corners_norm = torch.from_numpy(
        np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    ).to(device=dims.device, dtype=dims.dtype)

    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0.5])
    corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    # rotate around z axis
    corners = rotation_3d_in_axis(corners, tensor[:, 6], axis=2)
    # print("============")
    # print(corners)
    corners += tensor[:, :3].view(-1, 1, 3)
    return corners


def visualize_camera(
    image: np.ndarray,
    *,
    bboxes= np.ndarray,
    transform: Optional[np.ndarray] = None,
    thickness: float = 1,
) -> None:
    """
    bboxes : np.ndarray  -- > torch.Tensor
    """
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    bboxes  = torch.from_numpy(bboxes)  #-- > torch.Tensor
    if bboxes is not None and len(bboxes) > 0:
        corners = tocorners(bboxes)
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int_),
                    coords[index, end].astype(np.int_),
                    OBJECT_PALETTE,
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas_RGB = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    # canvas_RGB = cv2.resize(canvas_RGB,(1920, 1080))

    return canvas_RGB

