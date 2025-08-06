import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
from Model.short_model import AsymFormer
from Model.util import get_adj
from metrics import FDE, JPE, APE
from tqdm import tqdm
from data_short import Data
from detector.detector import InteractionMatrixGenerator

def batch_denormalization(data, para):
    '''
    :data: [B, T, N, J, 6] or [B, T, J, 3]
    :para: [B, 3]
    '''
    if data.shape[2]==2:
        data[..., :3] += para[:, None, None, None, :]
    else:
        data += para[:, None, None, :]
    return data


# mocap_umpm/mupots/3dpw/mix1/mix2
test_device = 'cuda'
test_dataset = Data(dataset='mix2', mode=1, device=test_device, transform=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False)
device = 'cuda'

model = AsymFormer(N=3, J=15, in_joint_size=50, in_relation_size=52,
                                   feat_size=512, out_joint_size=75, out_relation_size=75,
                                   num_heads=16, depth=8).to(device)

print('Data download completed')
model.load_state_dict(torch.load('./train_model/Train_short_16_0.001_833/epoch_70.model', map_location=device))

edges = np.array(
    [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [7, 9],
     [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]])

frame_idx = [5, 10, 15, 20, 25]
n = 0
rc = 1

ape_err_total = np.arange(len(frame_idx), dtype=np.float_)
jpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
fde_err_total = np.arange(len(frame_idx), dtype=np.float_)
prediction_all =[]
with torch.no_grad():
    model.eval()
    print('Validating Processing:')
    for _, test_data in tqdm(enumerate(test_dataloader, 0)):
        n = n + 1
        input_total_original = test_data

        batch_size = input_total_original.shape[0]  # 获取批次大小
        n_person = input_total_original.shape[1]  # 获取人数

        input_total_original = input_total_original.permute(0, 2, 1, 3).contiguous().view(batch_size, 75, n_person, 15,
                                                                                          6)  # B,T,N,J,D

        # 关节的固有连接
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12],
             [12, 13], [13, 14]])
        adj = get_adj(n_person, 15 * 3, edges)
        adj = adj.unsqueeze(0).unsqueeze(-1)

        # 个体间的交互连接
        #selected_joints = [1, 2, 3, 4, 5, 6, 12, 13, 14, 9, 10, 11]  # 选取12个关节
        num_selected = 15  # 12
        num_frames = 50
        dim = num_frames * num_selected * 3 * 2  # 50 * 12 * 3
        model_path = 'detector/short_interaction_detector.pth'

        generator = InteractionMatrixGenerator(model_path=model_path, input_size=dim, hidden_size=dim, output_size=1)
        generator = generator.to(test_device)
        inout = input_total_original[:, :50, :, :, :3].float()
        interaction_matrix = generator.predict_interactions(inout[:, :, :, :, :])

        block = torch.ones((15 * 3, 15 * 3), device=interaction_matrix.device)
        expanded_matrix = torch.kron(interaction_matrix, block)
        inter_matrix = interaction_matrix.cuda()
        conn = expanded_matrix.float().cuda().unsqueeze(-1)

        input_total_original = input_total_original.float().cuda()
        input_total = input_total_original.clone()

        batch_size = input_total.shape[0]  # 获取批次大小
        n_person = input_total.shape[2]  # 获取人数

        T = 75
        input_total[..., [1, 2]] = input_total[..., [2, 1]]
        input_total[..., [4, 5]] = input_total[..., [5, 4]]

        if rc:
            camera_vel = input_total[:, 1:50, :, :, 3:].mean(dim=(1, 2, 3))  # B, 3
            input_total[..., 3:] -= camera_vel[:, None, None, None]  # 减去相机速度
            input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:].cumsum(dim=1)  # 计算相对位置

        input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 75, 6)
        # B, NxJ, T, 6

        input_joint = input_total[:, :, :50]

        pos = input_total[:, :, :50, :3]
        pos = pos.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, 50)

        pos_i = pos.unsqueeze(-2)
        pos_j = pos.unsqueeze(-3)
        pos_rel = pos_i - pos_j
        dis = torch.pow(pos_rel, 2)
        dis = torch.sqrt(dis)
        exp_dis = torch.exp(-dis)
        input_relation = torch.cat((exp_dis, adj.repeat(batch_size, 1, 1, 1), conn), dim=-1)

        pred_vel = model.predict(input_joint[..., :3], input_joint[..., 3:], input_relation, inter_matrix)
        pred_vel = pred_vel[:, :, 50:]

        pred_vel = pred_vel.permute(0, 2, 1, 3)

        if rc:
            pred_vel = pred_vel + camera_vel[:, None, None]

        pred_vel[..., [1, 2]] = pred_vel[..., [2, 1]]

        motion_gt = input_total_original[..., :3].view(batch_size, T, -1, 3)
        motion_pred = (pred_vel.cumsum(dim=1) + motion_gt[:, 49:50])
        # motion_pred = pred_vel

        motion_gt = motion_gt[:, 49:74, :].view(batch_size, 25, n_person, 15, 3).permute(0, 2, 1, 3, 4)
        motion_pred = motion_pred.view(batch_size, 25, n_person, 15, 3).permute(0, 2, 1, 3, 4)

        prediction = motion_gt
        gt = motion_pred
        # ->test
        ape_err = APE(gt, prediction, frame_idx)
        jpe_err = JPE(gt, prediction, frame_idx)
        fde_err = FDE(gt, prediction, frame_idx)

        ape_err_total += ape_err
        jpe_err_total += jpe_err
        fde_err_total += fde_err

        prediction_all.append(prediction)
        if prediction_all:  # 确保列表非空
            prediction_all_tensor = torch.cat(prediction_all, dim=0).cpu().numpy()
        else:
            prediction_all_tensor = np.array([])  # 或者其他合适的默认值
    os.makedirs('./vis/', exist_ok=True)
    np.save('./vis/IPPJFormer_test_vis.npy', prediction_all_tensor) #B,N,T,J,D

    print("{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d} | {5:6d}".format("Lengths", 200, 400, 600, 800, 1000))
    print("=== JPE Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", jpe_err_total[0] / n,
                                                                                 jpe_err_total[1] / n,
                                                                                 jpe_err_total[2] / n,
                                                                                 jpe_err_total[3] / n,
                                                                                 jpe_err_total[4] / n))
    print("=== APE Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", ape_err_total[0] / n,
                                                                                 ape_err_total[1] / n,
                                                                                 ape_err_total[2] / n,
                                                                                 ape_err_total[3] / n,
                                                                                 ape_err_total[4] / n))
    print("=== FDE Test Error ===")
    print(
        "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}".format("Our", fde_err_total[0] / n,
                                                                                 fde_err_total[1] / n,
                                                                                 fde_err_total[2] / n,
                                                                                 fde_err_total[3] / n,
                                                                                 fde_err_total[4] / n))


