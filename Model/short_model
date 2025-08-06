import torch
import torch.nn as nn
import torch_dct as dct
# import torch.nn.functional as F



def compute_entropy(joint_features):
    """
    Args:
        joint_features: torch.Tensor,  (B, N, J, T, D)

    Returns:
        top_k_indices: torch.Tensor, 
    """
    B, N, J, T, D = joint_features.shape


    features_flat = joint_features.view(B, N, J, -1)  # shape: (B, N, J, T*D)


    prob_dist = F.softmax(features_flat, dim=-1)  #


    entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-8), dim=-1)  # 防止 log(0)

    return entropy  # shape: (B, N, J)

def select_joints_by_indices(data, top_k_indices):

    B, N, k = top_k_indices.shape
    J = 15  # 
    D = 3   # 
    _, _, C = data.shape

    data_reshaped = data.view(B, N, J, D, C)

    indices = top_k_indices.unsqueeze(-1).unsqueeze(-1)  # (B, N, k, 1, 1)
    indices = indices.expand(-1, -1, -1, D, C)  # (B, N, k, D, C)

    selected_data = torch.gather(data_reshaped, dim=2, index=indices)  # (B, N, k, D, C)

    selected_data = selected_data.view(B, N * k * D, C)

    return selected_data


def restore_selected_joints(selected_data, top_k_indices):
    """

    Args:
        selected_data: torch.Tensor,  (B, NKD, C)
        top_k_indices: torch.Tensor,  (B, N, K)
    Returns:
        restored_data: torch.Tensor,  (B, NJD, C)
    """
    B, N, k = top_k_indices.shape
    D = 3  # 
    J = 15  # 
    device = selected_data.device


    selected_data_reshaped = selected_data.view(B, N, k, D, -1)
    C = selected_data_reshaped.size(-1)


    restored_data = torch.zeros(B, N, J, D, C, device=device)


    expanded_indices = top_k_indices.unsqueeze(-1)  # (B, N, k, 1)
    expanded_indices = expanded_indices.unsqueeze(-1).expand(-1, -1, -1, D, C)  # (B, N, k, D, C)


    restored_data.scatter_(
        dim=2,  
        index=expanded_indices,
        src=selected_data_reshaped
    )

    return restored_data.view(B, N * J * D, C)



class MLP(nn.Module):
    def __init__(self, in_feat, out_feat, hid_feat=(256, 512), activation=None, dropout=-1):
        super(MLP, self).__init__()
        dims = (in_feat,) + hid_feat + (out_feat,)

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self.activation = activation if activation is not None else lambda x: x
        self.dropout = nn.Dropout(dropout) if dropout != -1 else lambda x: x

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.activation(x)
            x = self.dropout(x)
            x = self.layers[i](x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)




class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.J_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.R_conv = nn.Linear(dim, num_heads, bias=qkv_bias)
        # self.R_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, joint_feature, relation_feature, conn_feature, mask=None, ratio_i=0):
        #mask_ratio = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0., 0]
        B, N, C = joint_feature.shape
        H = self.num_heads
        HS = C // self.num_heads
        conn_feature = conn_feature.unsqueeze(1).repeat(1, H, 1, 1)
        J_qkv = self.J_qkv(joint_feature).reshape(B, N, 3, H, HS).permute(2, 0, 3, 1, 4)  # [3, B, #heads, N, C//#heads]
        # R_qkv = self.R_qk(relation_feature).reshape(B, N, N, 2, H, HS).permute(3, 0, 4, 1, 2, 5)  # [3, B, #heads, N, N, C//#heads]

        J_q, J_k, J_v = J_qkv[0], J_qkv[1], J_qkv[2]  # [B, #heads, N, C//#heads]
        # R_q, R_k = R_qkv[0], R_qkv[1]  # [B, #heads, N, N, C//#heads]

        attn_J = (J_q @ J_k.transpose(-2, -1))  # [B, #heads, N, N]
        attn_R_linear = self.R_conv(relation_feature).reshape(B, N, N, H).permute(0, 3, 1, 2)  # [B, #heads, N, N]
        # attn_R_qurt = (R_q.unsqueeze(-2) @ R_k.unsqueeze(-1)).squeeze()  # [B, #heads, N, N]
        attn = (attn_J + attn_R_linear) * conn_feature * self.scale  # [B, #heads, NJD, NJD]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        #m_r = torch.ones_like(attn) * mask_ratio[ratio_i]
        #attn = attn + torch.bernoulli(m_r) * -1e12

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, #heads, N, N]

        x = (attn @ J_v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention_k(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.J_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.R_conv = nn.Linear(dim, num_heads, bias=qkv_bias)
        # self.R_qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, joint_feature, k_conn, mask=None):

        B, N, C = joint_feature.shape
        H = self.num_heads
        HS = C // self.num_heads
        conn_feature = k_conn.unsqueeze(1).repeat(1,H,1, 1)
        J_qkv = self.J_qkv(joint_feature).reshape(B, N, 3, H, HS).permute(2, 0, 3, 1, 4)  # [3, B, #heads, N, C//#heads]
        # R_qkv = self.R_qk(relation_feature).reshape(B, N, N, 2, H, HS).permute(3, 0, 4, 1, 2, 5)  # [3, B, #heads, N, N, C//#heads]

        J_q, J_k, J_v = J_qkv[0], J_qkv[1], J_qkv[2]  # [B, #heads, N, C//#heads]
        # R_q, R_k = R_qkv[0], R_qkv[1]  # [B, #heads, N, N, C//#heads]

        attn_J = (J_q @ J_k.transpose(-2, -1))  # [B, #heads, N, N]
        # attn_R_linear = self.R_conv(relation_feature).reshape(B, N, N, H).permute(0, 3, 1, 2)  # [B, #heads, N, N]
        # attn_R_qurt = (R_q.unsqueeze(-2) @ R_k.unsqueeze(-1)).squeeze()  # [B, #heads, N, N]
        attn = attn_J * self.scale * conn_feature  # [B, #heads, NJD, NJD]

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, #heads, N, N]

        x = (attn @ J_v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm_attn1 = norm_layer(dim)
        self.norm_attn2 = norm_layer(dim)
        self.norm_joint = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_joint = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        #self.config = MOEConfig(dim, 4, 1)
        #self.MoEFFN = MoEFFN(self.config)

    def forward(self, joint_feature, relation_feature, conn_feature, mask=None, ratio_i=0):
        B, N, C = joint_feature.shape
        ## joint feature update through attention mechanism
        joint_feature = joint_feature + self.drop_path(
            self.attn(self.norm_attn1(joint_feature), self.norm_attn2(relation_feature), conn_feature, mask, ratio_i))
        joint_feature = joint_feature + self.drop_path(self.mlp_joint(self.norm_joint(joint_feature)))

        return joint_feature


class Block_k(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.attn = Attention_k(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.norm_attn1 = norm_layer(dim)
        self.norm_attn2 = norm_layer(dim)
        self.norm_joint = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_joint = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        #self.config = MOEConfig(dim, 4, 1)
        #self.MoEFFN = MoEFFN(self.config)

    def forward(self, joint_feature,k_conn, mask=None):
        B, N, C = joint_feature.shape
        ## joint feature update through attention mechanism
        joint_feature = joint_feature + self.drop_path(self.attn(self.norm_attn1(joint_feature),k_conn, mask))
        # Add & Norm

        joint_feature = joint_feature + self.drop_path(self.mlp_joint(self.norm_joint(joint_feature)))

        return joint_feature


class AsymFormer(nn.Module):
    def __init__(self, N=2, J=13, in_joint_size=16 * 6, in_relation_size=26, feat_size=128, out_joint_size=30 * 3,
                 out_relation_size=30, num_heads=8, depth=4, norm_layer=nn.LayerNorm):
        super().__init__()


        self.joint_encoder = MLP(in_joint_size, feat_size, (256, 256))
        self.relation_encoder = MLP(in_relation_size, feat_size, (256, 256))
        self.norm_layer = norm_layer(feat_size)

        # Generalized attention encoder block for k1 and k2
        self.attn_encoder = nn.ModuleList([
            Block(feat_size, num_heads, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        self.attn_encoder_k1 = nn.ModuleList([
            Block_k(feat_size, num_heads, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(1)
        ])

        self.attn_encoder_k2 = nn.ModuleList([
            Block_k(feat_size, num_heads, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(1)
        ])

        self.joint_decoder = MLP(feat_size, out_joint_size)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights for layers."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Custom weight initialization."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, n_joint, x_joint, x_relation, inter_matrix):
        B, NJ, T, D = x_joint.shape
        N = NJ // 15  # Number of individuals
        J = 15

        k1=4
        k2=8
        joint_entropy = compute_entropy(n_joint.reshape(B, N, J, T, D))

        block = torch.ones((J * 3, J * 3), device=inter_matrix.device)
        x_conn = torch.kron(inter_matrix, block)

        block = torch.ones((k1 * 3, k1 * 3), device=inter_matrix.device)
        k1_conn = torch.kron(inter_matrix, block)

        block = torch.ones((k2 * 3, k2 * 3), device=inter_matrix.device)
        k2_conn = torch.kron(inter_matrix, block)


        # Select top k1 and k2 joints based on entropy
        _, top_k1 = torch.topk(joint_entropy, k1, dim=2, largest=True, sorted=True)
        _, top_k2 = torch.topk(joint_entropy, k2, dim=2, largest=True, sorted=True)

        # Transform joint data to feature space
        x_joint = x_joint.permute(0, 1, 3, 2).contiguous().view(B, -1, 50)
        x_joint = dct.dct(x_joint)
        feat_x_joint = self.joint_encoder(x_joint)
        feat_x_relation = self.relation_encoder(x_relation)

        # Select top k1 joints and process with attention
        k1_joints = select_joints_by_indices(feat_x_joint, top_k1)
        for blk in self.attn_encoder_k1:
            k1_joints = blk(k1_joints, k1_conn)

        feat_x_joint = restore_selected_joints(k1_joints, top_k1) + feat_x_joint

        # Select top k2 joints and process with attention
        k2_joints = select_joints_by_indices(feat_x_joint, top_k2)
        for blk in self.attn_encoder_k2:
            k2_joints = blk(k2_joints, k2_conn)

        feat_x_joint = restore_selected_joints(k2_joints, top_k2) + feat_x_joint

        # Apply attention blocks
        for blk in self.attn_encoder:
            feat_x_joint = blk(feat_x_joint, feat_x_relation, x_conn)

        # Final prediction
        pred = self.joint_decoder(feat_x_joint).contiguous().view(B, -1, 75)
        pred = dct.idct(pred)
        pred = pred.reshape(B, NJ, 3, 75).permute(0, 1, 3, 2).contiguous()

        return pred

    def predict(self,n_joint, x_joint, x_relation, x_conn):
        return self.forward(n_joint,x_joint, x_relation, x_conn)

