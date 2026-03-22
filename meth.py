import torch
import torch.nn.functional as F

def compute_jcd(sequence: torch.Tensor) -> torch.Tensor:
    """
    Compute joint-to-joint distances (JCD) for a frame sequence.
    sequence: [T, C]  (flattened joints: J*C)
    Returns: [T, num_jcd_features]  (upper-triangle distances)
    """
    T, C = sequence.shape
    C = (C // 3) * 3
    sequence = sequence[:, :C]
   
    J = C // 3  # assume 3 coords per joint
    joints = sequence.view(T, J, 3)                 # [T, J, 3]
    diff = joints[:, :, None, :] - joints[:, None, :, :]  # [T, J, J, 3]
    dist = torch.norm(diff, dim=-1)                # [T, J, J]
    triu_idx = torch.triu_indices(J, J, offset=1)
    jcd = dist[:, triu_idx[0], triu_idx[1]]        # [T, J*(J-1)/2]
    return jcd

def compute_td(sequence: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """
    Compute temporal differences (TD) of a sequence.
    sequence: [T, C]
    stride: how many frames to skip
    Returns: [T, C] with zero padding at start
    """
    td = torch.zeros_like(sequence)
    if sequence.shape[0] > stride:
        td[stride:] = sequence[stride:] - sequence[:-stride]
    return td


def compute_acc(diff: torch.Tensor) -> torch.Tensor:
    if diff is None or diff.numel() == 0:
        return torch.zeros_like(diff)
    acc = diff[1:] - diff[:-1]
    acc = F.pad(acc, (0, 0, 1, 0))
    return acc