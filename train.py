import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.optim as optim
import numpy as np
import torch
import time
import random
from Model.short_model import AsymFormer
from Model.util import rotate_Y, get_adj, distance_loss, process_pred
from metrics import FDE, JPE, APE
from tqdm import tqdm
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


# train dataset
from data_short import Data
train_device = 'cuda'
dataset = Data(dataset='mocap_umpm', mode=0, device=train_device, transform=False)
batch_size = 16
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# test dataset
test_device = 'cuda'
test_dataset = Data(dataset='mocap_umpm', mode=1, device=test_device, transform=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False)

model = AsymFormer(N=3, J=15, in_joint_size=50, in_relation_size=52,
                                   feat_size=512, out_joint_size=75, out_relation_size=75,
                                   num_heads=16, depth=8).to(train_device)

lrate=0.001
print(">>> training params: {:.2f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))
# predictor P
optimizer = optim.AdamW(model.parameters(), lr=lrate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

# save model name
T_Model = f'Train_JRT_{batch_size}_{lrate}_temp'
B_Model = f'Train_JRT_{batch_size}_{lrate}_temp'

# log
log_dir = str(os.path.join(os.getcwd(), 'JRT_logs/'))
model_dir = T_Model + '/'
if not os.path.exists(log_dir + model_dir):
    os.makedirs(log_dir + model_dir)

loss_list = []
error_list = []

best_eval = 100
start_time = time.time()
best_epoch = 0
losses = []  # 初始化损失值列表
# 加载训练模型
# save_path = './train_model/Train_short_16_0.001_833/epoch_70.model'
# model.load_state_dict(torch.load(save_path, map_location=train_device))

# training
for epoch in range(100):
    rc = 1
    tf = 1
    steps = 0
    test_loss_list = []
    total_loss=0
    print(f'-------------------------Epoch:{epoch}-----------------------------')
    print('Time since start:{:.1f} minutes.'.format((time.time() - start_time) / 60.0))

    lrate = optimizer.param_groups[0]['lr']
    print('Training Processing:')
    all_mpjpe = np.zeros(5)
    count = 0


    for j, train_data in tqdm(enumerate(train_dataloader)):

        input_total = train_data  # 获取数据批次   B,N,T,JD

        batch_size = input_total.shape[0]  # 获取批次大小
        n_person = input_total.shape[1]   #获取人数

        input_total = input_total.permute(0, 2, 1, 3).contiguous().view(batch_size, 75, n_person, 15,6)   #B,T,N,J,D

        # 关节的固有连接
        edges = np.array(
            [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7],[7, 8], [7, 9],
             [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]])

        adj = get_adj(n_person, 15*3, edges)
        adj = adj.unsqueeze(0).unsqueeze(-1)
        # 个体间的交互连接
        #selected_joints = [1, 2, 3, 4, 5, 6, 12, 13, 14, 9, 10, 11]  # 选取12个关节
        num_selected = 15  # 12
        num_frames = 50
        dim = num_frames * num_selected * 3 * 2  # 25 * 12 * 3
        model_path = 'detector/short_interaction_detector.pth'

        generator = InteractionMatrixGenerator(model_path=model_path, input_size=dim, hidden_size=dim, output_size=1)
        generator = generator.to(train_device)

        inout = input_total[:, :50, :, :, :3].float()
        interaction_matrix = generator.predict_interactions(inout[:, :, :, :, :])
        block = torch.ones((15 * 3, 15 * 3), device=interaction_matrix.device)
        expanded_matrix = torch.kron(interaction_matrix, block)
        inter_matrix = interaction_matrix.cuda()
        conn = expanded_matrix.float().cuda().unsqueeze(-1)

        input_total = input_total.float().cuda()  # 将数据移动到GPU，并转换为浮点类型

        input_total[..., [1, 2]] = input_total[..., [2, 1]]  # 调整坐标轴顺序    B,T,N,J,D
        input_total[..., [4, 5]] = input_total[..., [5, 4]]  # 调整坐标轴顺序

        # 如果启用了相对坐标，则计算并减去相机速度
        if rc:
            camera_vel = input_total[:, 1:50, :, :, 3:].mean(dim=(1, 2, 3))  # B, 3
            input_total[..., 3:] -= camera_vel[:, None, None, None]  # 减去相机速度
            input_total[..., :3] = input_total[:, 0:1, :, :, :3] + input_total[..., 3:].cumsum(dim=1)  # 计算相对位置

        input_total = input_total.permute(0, 2, 3, 1, 4).contiguous().view(batch_size, -1, 75, 6)  # 调整数据维度  B, NxJ, T, 6


        if tf:
            angle = random.random() * 360  # 生成随机旋转角度
            # 随机旋转数据
            input_total = rotate_Y(input_total, angle)
            input_total *= (random.random() * 0.4 + 0.8)  # 随机缩放数据

        input_joint = input_total[:, :, :50]  # 获取关节特征
        #last_frame = input_joint[:, :, -1:]
        #repeated_last_frame = last_frame.repeat(1, 1, 10, 1)
        #input_joint = torch.cat((input_joint, repeated_last_frame), dim=2)
        # 计算关系特征
        pos = input_total[:, :, :50, :3]    # B,NJ,T,D
        pos = pos.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, 50)

        pos_i = pos.unsqueeze(-2)
        pos_j = pos.unsqueeze(-3)
        pos_rel = pos_i - pos_j
        dis = torch.pow(pos_rel, 2)
        dis = torch.sqrt(dis)
        exp_dis = torch.exp(-dis)

        exp_dis_in = exp_dis[:, :, :, :50]

        input_relation = torch.cat((exp_dis_in, adj.repeat(batch_size, 1, 1, 1), conn), dim=-1)
        # B, NJ, NJ, C


        # 执行模型的前向传播
        pred_vel= model(input_joint[..., :3],input_joint[..., 3:],input_relation, inter_matrix)

        recon_vel, pred_vel = process_pred(pred_vel)

        gt_vel = input_total[..., 3:]  # 获取真实速度
        # [B, NxJ, T=30, 3]
        gt_vel_x = gt_vel[:, :, :50]
        gt_vel_y = gt_vel[:, :, 49:74]

        # 计算损失
        loss_recon = distance_loss(recon_vel, gt_vel_x)
        loss_pred = distance_loss(pred_vel, gt_vel_y)
        loss = loss_pred * 50 + loss_recon * 20
        count += batch_size  # 更新样本计数器
        losses.append([loss.item()])  # 记录当前损失

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    scheduler.step()

    with open(os.path.join(log_dir + model_dir, 'log.txt'), 'a+') as log:
            log.write('Epoch: {} \n'.format(epoch))
            log.write('Lrate: {} \n'.format(lrate))
            log.write('Time since start:{:.1f} minutes.\n'.format((time.time() - start_time) / 60.0))
    save_path = os.path.join('train_model',f'{T_Model}', f'epoch_{epoch}.model')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

# Test
    # 加载模型
    model.load_state_dict(torch.load(save_path, map_location=test_device))

    frame_idx = [5, 10, 15, 20, 25]
    n = 0
    ape_err_total = np.arange(len(frame_idx), dtype=np.float_)
    jpe_err_total = np.arange(len(frame_idx), dtype=np.float_)
    fde_err_total = np.arange(len(frame_idx), dtype=np.float_)
    test_loss = 0
    loss_list1 = []
    loss_list2 = []
    loss_list3 = []
    all_mpjpe = np.zeros(5)
    count = 0
    with torch.no_grad():
        model.eval()
        print('Validating Processing:')
        for _, test_data in tqdm(enumerate(test_dataloader,0)):
            n = n+1
            input_total_original = test_data

            batch_size = input_total_original.shape[0]  # 获取批次大小
            n_person = input_total_original.shape[1]  # 获取人数

            input_total_original = input_total_original.permute(0, 2, 1, 3).contiguous().view(batch_size, 75, n_person, 15,6)  # B,T,N,J,D


            # 关节的固有连接
            edges = np.array([[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [7, 9],[9, 10], [10, 11], [7, 12], [12, 13], [13, 14]])
            adj = get_adj(n_person, 15*3, edges)
            adj = adj.unsqueeze(0).unsqueeze(-1)

            # 个体间的交互连接
            selected_joints = [1, 2, 3, 4, 5, 6, 12, 13, 14, 9, 10, 11]  # 选取12个关节
            num_selected = len(selected_joints)  # 12
            num_frames = 50
            dim = num_frames * num_selected * 3  # 50 * 12 * 3
            model_path = 'detector/interaction_detector.pth'

            generator = InteractionMatrixGenerator(model_path=model_path, input_size=dim, hidden_size=dim,output_size=1)
            generator = generator.to(test_device)
            inout = input_total_original[:, :50, :, selected_joints, :3].float()
            interaction_matrix = generator.predict_interactions(inout[:, ::2, :, :, :])


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
            #last_frame = input_joint[:, :, -1:]
            #repeated_last_frame = last_frame.repeat(1, 1, 10, 1)
            #input_joint = torch.cat((input_joint, repeated_last_frame), dim=2)

            pos = input_total[:, :, :50, :3]
            pos = pos.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, 50)

            pos_i = pos.unsqueeze(-2)
            pos_j = pos.unsqueeze(-3)
            pos_rel = pos_i - pos_j
            dis = torch.pow(pos_rel, 2)
            dis = torch.sqrt(dis)
            exp_dis = torch.exp(-dis)
            input_relation = torch.cat((exp_dis, adj.repeat(batch_size, 1, 1, 1), conn), dim=-1)

            pred_vel = model.predict(input_joint[..., :3],input_joint[..., 3:], input_relation, inter_matrix)
            pred_vel = pred_vel[:, :, 50:]

            pred_vel = pred_vel.permute(0, 2, 1, 3)

            if rc:
                pred_vel = pred_vel + camera_vel[:, None, None]

            pred_vel[..., [1, 2]] = pred_vel[..., [2, 1]]

            motion_gt = input_total_original[..., :3].view(batch_size, T, -1, 3)
            motion_pred = (pred_vel.cumsum(dim=1) + motion_gt[:, 49:50])
            # motion_pred = pred_vel

            motion_gt = motion_gt[:, 50:75, :].view(batch_size, 25, n_person, 15, 3).permute(0, 2, 1, 3, 4)
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

        # ->log
        with open(os.path.join(log_dir + model_dir, 'log.txt'), 'a+') as log:
            log.write(
                "{0: <16} | {1:6d} | {2:6d} | {3:6d} | {4:6d} | {5:6d}\n".format("Lengths", 200, 400, 600, 800, 1000))
            log.write("=== APE Test Error ===\n")
            log.write(
                "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}\n".format("Our", jpe_err_total[0] / n,
                                                                                           jpe_err_total[1] / n,
                                                                                           jpe_err_total[2] / n,
                                                                                           jpe_err_total[3] / n,
                                                                                           jpe_err_total[4] / n))
            log.write("=== APE Test Error ===\n")
            log.write(
                "{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}\n".format("Our", ape_err_total[0] / n,
                                                                                           ape_err_total[1] / n,
                                                                                           ape_err_total[2] / n,
                                                                                           ape_err_total[3] / n,
                                                                                           ape_err_total[4] / n))
            log.write("=== APE Test Error ===\n")
            log.write("{0: <16} | {1:6.0f} | {2:6.0f} | {3:6.0f} | {4:6.0f} | {5:6.0f}\n\n".format("Our",
                                                                                                   fde_err_total[0] / n,
                                                                                                   fde_err_total[1] / n,
                                                                                                   fde_err_total[2] / n,
                                                                                                   fde_err_total[3] / n,
                                                                                                   fde_err_total[4] / n))
