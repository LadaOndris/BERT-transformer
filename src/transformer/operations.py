import numpy as np
import torch


def softmax(tensor: torch.Tensor, dim=-1):
    max_values, indices = torch.max(tensor, dim=dim, keepdim=True)
    exp = torch.exp(tensor - max_values)
    exp_sum = torch.sum(exp, dim=dim, keepdim=True)
    return torch.divide(exp, exp_sum)


def positional_encoding(max_position_encoding, d_model):
    positions = np.arange(max_position_encoding)[:, np.newaxis]  # (max_position_encoding, 1)
    dimensions = np.arange(d_model)[np.newaxis, :]  # (1, d_model)
    # (max_position_encoding, d_model)
    angles = positions / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))

    # Apply sin to even indices
    angles[0::2] = np.sin(angles[0::2])
    # Apply cos to odd indices
    angles[1::2] = np.cos(angles[1::2])

    encoding = angles[np.newaxis, ...]  # (1, max_position_encoding, d_model)

    return torch.from_numpy(encoding)


def create_padding_mask(x):
    """
    Creates a mask to denote padding locations.
    The mask contains value 1.0 if the value in the given
    sequence is zero. Other values are set to 0.

    :param x: A sequence of shape (batch_size, seq_len)
    :return: A padding mask of shape (batch_size, 1, 1, seq_len)
    """
    mask = (x == 0.).type(x.dtype)
    return mask[:, None, None, :]
