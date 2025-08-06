import torch
import numpy as np

def APE(V_pred, V_trgt, frame_idx):
    V_pred = V_pred - V_pred[:, :, :, 0:1, :]
    V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()
    return err * scale


def JPE(V_pred, V_trgt, frame_idx):
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2), dim=1).cpu().data.numpy().mean()
    return err * scale


# def ADE(V_pred, V_trgt, frame_idx):
#     scale = 1000
#     err = np.arange(len(frame_idx), dtype=np.float_)
#     for idx in range(len(frame_idx)):
#         err[idx] = torch.linalg.norm(V_trgt[:, :, :frame_idx[idx], :1, :] - V_pred[:, :, :frame_idx[idx], :1, :], dim=-1).mean(1).mean()
#     return err * scale


def FDE(V_pred,V_trgt, frame_idx):
    scale = 1000
    err = np.arange(len(frame_idx), dtype=np.float_)
    for idx in range(len(frame_idx)):
        err[idx] = torch.linalg.norm(V_trgt[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :] - V_pred[:, :, frame_idx[idx]-1:frame_idx[idx], : 1, :], dim=-1).mean(1).mean()
    return err * scale


def MPJPE(GT, pred, select_frames=[4, 9, 10, 19, 20]):
    '''Calculate the MPJPE at selected timestamps.

    Args:
        GT: [B, T, J, 3], np.array, ground-truth pose position in world coordinate system (meter).
        pred: [B, T, J, 3], np.array, predicted pose position.

    Returns:
        errorPose: [T], MPJPE at selected timestamps.
    '''

    errorPose = np.power(GT - pred, 2)
    # B, T, J, 3
    errorPose = np.sum(errorPose, -1)
    errorPose = np.sqrt(errorPose)
    # B, T, J
    errorPose = errorPose.sum(axis=-1) / pred.shape[2]
    # B, T
    errorPose = errorPose.sum(axis=0)
    # T
    return errorPose[select_frames]
