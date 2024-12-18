import torch
import torch.nn.functional as F

from utils.rotation_conversions import axis_angle_to_matrix
from utils.rotation_conversions import matrix_to_axis_angle
from utils.rotation_conversions import matrix_to_quaternion
from utils.rotation_conversions import matrix_to_rotation_6d
from utils.rotation_conversions import quaternion_to_matrix
from utils.rotation_conversions import rotation_6d_to_matrix


def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat


def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat


def rotation_6d_to_matrix_from_source(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    # print("a1: ", a1[1, 1, 1, 1], a1[1, 1, 12, 1], a1[1, 1, 22, 1])
    b1 = F.normalize(a1, dim=-1)
    # print("b1: ", b1[1, 1, 1, 1], b1[1, 1, 12, 1], b1[1, 1, 22, 1])
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    # print("b2: ", b2[1, 1, 1, 1], b2[1, 1, 12, 1], b2[1, 1, 22, 1])
    b2 = F.normalize(b2, dim=-1)
    # print("b2 after norm: ", b2[1, 1, 1, 1], b2[1, 1, 12, 1], b2[1, 1, 22, 1])
    b3 = torch.cross(b1, b2, dim=-1)
    # print("b3: ", b3[1, 1, 1, 1], b3[1, 1, 12, 1], b3[1, 1, 22, 1])
    return torch.stack((b1, b2, b3), dim=-2)

def quaternion_to_axis_angle_from_source(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def ax_from_6v(q):
    # 6 dof to 3d rotation matrix
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix_from_source(q)
    # print("mat: ", mat[1, 1, 1, 1, 1], mat[1, 1, 12, 1, 1], mat[1, 1, 22, 1, 1])
    # quaternion = matrix_to_quaternion(mat)
    # print("quaternion:", quaternion[1, 1, 1, 1], quaternion[1, 1, 12, 1], quaternion[1, 1, 22, 1])
    # ax = quaternion_to_axis_angle_from_source(quaternion)
    ax = matrix_to_axis_angle(mat)
    return ax


def quat_slerp(x, y, a):
    """
    Performs spherical linear interpolation (SLERP) between x and y, with proportion a

    :param x: quaternion tensor (N, S, J, 4)
    :param y: quaternion tensor (N, S, J, 4)
    :param a: interpolation weight (S, )
    :return: tensor of interpolation results
    """

    # dot product to get cosine
    len = torch.sum(x * y, axis=-1)

    neg = len < 0.0
    len[neg] = -len[neg]
    y[neg] = -y[neg]

    a = torch.zeros_like(x[..., 0]) + a

    amount0 = torch.zeros_like(a)
    amount1 = torch.zeros_like(a)

    # if sine too small, use linear interpolation
    linear = (1.0 - len) < 0.01
    omegas = torch.arccos(len[~linear])
    sinoms = torch.sin(omegas)

    amount0[linear] = 1.0 - a[linear]
    amount0[~linear] = torch.sin((1.0 - a[~linear]) * omegas) / sinoms

    amount1[linear] = a[linear]
    amount1[~linear] = torch.sin(a[~linear] * omegas) / sinoms

    # reshape
    amount0 = amount0[..., None]
    amount1 = amount1[..., None]

    res = amount0 * x + amount1 * y

    return res


if __name__ == '__main__':
    x = torch.randn(100, 1, 25, 4)
    y = torch.randn(100, 1, 25, 4)
    a = torch.tensor([0.5])[None, :, None]

    z = quat_slerp(x, y, a)
    print(f'x: {x[0, 0]}')
    print(f'y: {y[0, 0]}')
    print(f'z: {z[0, 0]}')