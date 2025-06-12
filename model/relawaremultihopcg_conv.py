from helper import *
from model.message_passing import MessagePassing
import torch.nn as nn

class RelAwareMultiHopCompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_rels, num_bases, num_hops=2, act=lambda x: x, cache=True, params=None):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hops = num_hops  # 多跳的数量
        self.act = act
        self.device = None
        self.cache = cache

        # 检查并设置 b_norm 的默认值
        if not hasattr(self.p, 'b_norm'):
            self.p.b_norm = True

        self.w_loop = get_param((in_channels, out_channels))
        self.w_in = get_param((in_channels, out_channels))
        self.w_out = get_param((in_channels, out_channels))

        self.rel_basis = get_param((self.num_bases, in_channels))
        self.rel_wt = get_param((self.num_rels * 2, self.num_bases))
        self.w_rel = get_param((in_channels, out_channels))
        self.loop_rel = get_param((1, in_channels))

        self.drop = torch.nn.Dropout(self.p.dropout)
        self.bn = torch.nn.BatchNorm1d(out_channels)

        self.in_norm, self.out_norm = None, None
        self.in_index, self.out_index = None, None
        self.in_type, self.out_type = None, None
        self.loop_index, self.loop_type = None, None

        # 定义 MLP 层
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * 2),
            nn.ReLU(),
            nn.Linear(out_channels * 2, out_channels)
        )

        if self.p.bias:
            self.register_parameter('bias', Parameter(torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, edge_norm=None, rel_embed=None):
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.mm(self.rel_wt, self.rel_basis)
        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)

        num_edges = edge_index.size(1) // 2
        num_ent = x.size(0)

        if not self.cache or self.in_norm is None:
            self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:]
            self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:]

            self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device)
            self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, dtype=torch.long).to(self.device)

            self.in_norm = self.compute_norm(self.in_index, num_ent)
            self.out_norm = self.compute_norm(self.out_index, num_ent)

        # 多跳消息传递
        for _ in range(self.num_hops):
            in_res = self.propagate('add', self.in_index, x=x, edge_type=self.in_type, rel_embed=rel_embed,
                                    edge_norm=self.in_norm, mode='in')
            loop_res = self.propagate('add', self.loop_index, x=x, edge_type=self.loop_type, rel_embed=rel_embed,
                                      edge_norm=None, mode='loop')
            out_res = self.propagate('add', self.out_index, x=x, edge_type=self.out_type, rel_embed=rel_embed,
                                     edge_norm=self.out_norm, mode='out')
            x = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

            if self.p.bias:
                x = x + self.bias
            if self.p.b_norm:
                x = self.bn(x)

        # 通过 MLP 进行更新
        x = self.mlp(x)

        return self.act(x), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed):
        """实体-关系嵌入组合前的维度预处理"""
        # 确保输入维度一致
        if ent_embed.shape != rel_embed.shape:
            min_batch = min(ent_embed.shape[0], rel_embed.shape[0])
            min_dim = min(ent_embed.shape[1], rel_embed.shape[1])
            
            ent_embed = ent_embed[:min_batch, :min_dim]
            rel_embed = rel_embed[:min_batch, :min_dim]
            # print(f"[WARN] rel_transform shape mismatch: {ent_embed.shape} vs {rel_embed.shape}")
        
        # 确保维度为偶数（循环相关的最佳实践）
        if ent_embed.shape[1] % 2 != 0:
            ent_embed = F.pad(ent_embed, (0, 1))  # 补零至偶数维度
            rel_embed = F.pad(rel_embed, (0, 1))
            # print(f"[WARN] Padded to even dimension: {ent_embed.shape[1]}")
        
        if self.p.opn == 'corr':
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p.opn == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p.opn == 'mult':
            trans_embed = ent_embed * rel_embed
        else:
            raise NotImplementedError

        return trans_embed

    def message(self, x_j, edge_type, rel_embed, edge_norm, mode):
        weight = getattr(self, 'w_{}'.format(mode))
        rel_emb = torch.index_select(rel_embed, 0, edge_type)
        xj_rel = self.rel_transform(x_j, rel_emb)
        out = torch.mm(xj_rel, weight)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        edge_weight = torch.ones_like(row).float()
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_ent)
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0
        norm = deg_inv[row] * edge_weight * deg_inv[col]

        return norm

    def __repr__(self):
        return '{}({}, {}, num_rels={}, num_hops={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.num_rels, self.num_hops)