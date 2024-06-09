import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 判断电脑GPU可不可用，如果可用的话device就采用cuda()即调用GPU，不可用的话就采用cpu()即调用CPU


# 时间注意力
class TScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(TScaledDotProductAttention, self).__init__()
    # 可以添加各种网络层，例如添加self.conv1 = nn.Conv2d(3,10,3)  即：in_channels=3, out_channels=10, kernel_size=3

    def forward(self, Q, K, V):  # 定义向前传播
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # matmul tensor乘法,score是注意力分数
                                           # sorces=[B, h, N, T, T]
        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax，得到权重
        context = torch.matmul(attn,
                               V)  # [batch_size, n_heads, Nodes, T(Temporal), d_k]]
        return context


class TMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(TMultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.W_V = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, N, T, C]  T=12
        input_K: [batch_size, N, T, C]
        input_V: [batch_size, N, T, C]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        B, N, T, C = input_Q.shape
        # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = TScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, d_k]
        context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, d_k]
        context = context.reshape(B, N, T, self.heads * self.head_dim)  # [B, N, T, C]
        # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc_out(context)  # [batch_size, N, T ,embed_size]
        return output


# class FMultiHeadAttention(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(FMultiHeadAttention, self).__init__()
#
#         self.embed_size = embed_size
#         self.heads = heads
#         self.head_dim = embed_size // heads
#
#         assert (
#                 self.head_dim * heads == embed_size
#         ), "Embedding size needs to be divisible by heads"
#
#         self.W_V = nn.Linear(2*self.embed_size, 2*self.head_dim * self.heads, bias=False)
#         self.W_K = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
#         self.W_Q = nn.Linear(self.embed_size, self.head_dim * self.heads, bias=False)
#         self.fc_out = nn.Linear(2*heads * self.head_dim, 2*embed_size)
#
#     def forward(self, input_Q, input_K, input_V):
#         '''
#         input_Q: [batch_size, N, T, C]  T=12
#         input_K: [batch_size, N, T, C]
#         input_V: [batch_size, N, T, 2*C]
#         attn_mask: [batch_size, seq_len, seq_len]
#         '''
#         B, N, T, C = input_Q.shape
#         # [B, N, T, C] --> [B, N, T, h * d_k] --> [B, N, T, h, d_k] --> [B, h, N, T, d_k]
#         Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # Q: [B, h, N, T, d_k]
#         K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)  # K: [B, h, N, T, d_k]
#         V = self.W_V(input_V).view(B, N, T, self.heads, 2*self.head_dim).permute(0, 3, 1, 2, 4)  # V: [B, h, N, T, d_k]
#
#         # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]
#
#         # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
#         context = TScaledDotProductAttention()(Q, K, V)  # [B, h, N, T, 2*d_k]
#         context = context.permute(0, 2, 3, 1, 4)  # [B, N, T, h, 2*d_k]
#         context = context.reshape(B, N, T, 2*self.heads * self.head_dim)  # [B, N, T, 2*C]
#         # context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
#         output = self.fc_out(context)  # [batch_size, N, T ,2*embed_size]
#         return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_size):
        super(PoswiseFeedForwardNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, output):  # inputs: [batch_size, N, T, embed_size]
        residual = output
        output = self.fc(output)
        output = self.norm(output + residual)
        return output  # [batch_size, N, T, embed_size]


# class PositionalEncoding(nn.Module):
#     def __init__(self, embed_size, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pos_table = np.array([
#         [pos / np.power(10000, 2 * i / embed_size) for i in range(embed_size)]
#         if pos != 0 else np.zeros(embed_size) for pos in range(max_len)])
#         pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])  # 字嵌入维度为偶数时
#         pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])  # 字嵌入维度为奇数时
#         self.pos_table = torch.FloatTensor(pos_table)  # enc_inputs: [T, embed_size]
#
#     def forward(self, inputs):  # enc_inputs: [batch_size, N, T, embed_size]
#         inputs += self.pos_table[:inputs.size(2), :].to(device)
#         return self.dropout(inputs)


class TEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads):
        super(TEncoderLayer, self).__init__()
        # self.pos_emb = PositionalEncoding(embed_size)
        self.enc_self_attn = TMultiHeadAttention(embed_size, heads)  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(embed_size)  # 前馈神经网络

    def forward(self, inputs):     # enc_inputs:[batch_size, N, T, embed_size]
        # 输入三个enc_inputs分别与W_q,w_k,w_v相乘得到Q、K、V
        # outputs = self.pos_emb(inputs)
        # outputs = self.enc_self_attn(outputs, outputs, outputs)
        outputs = self.enc_self_attn(inputs, inputs, inputs)
        outputs = self.pos_ffn(outputs)  # enc_outputs:[batch_size, N, T, embed_size]

        return outputs  # attn: [batch_size, N, T, embed_size] # outputs: [batch_size, N, T, embed_size]


# 时间卷积
class Chomp1d(nn.Module):
    '''
    这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
    '''
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# 这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))  # 卷积
        self.chomp1 = Chomp1d(padding)   # 修改尺寸
        self.relu1 = nn.ReLU()           # relu
        self.dropout1 = nn.Dropout(dropout)   # dropout

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)  # 卷积+修改数据尺寸+relu+dropout+卷积+修改数据尺寸+relu+dropout
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  # 下采样，如果input不等于output，用一维卷积将其输入维度改为输出维度

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):   # # 对网络参数进行均值为0，标准差为0.01的初始化
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print(x.shape)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # 残差链接


class TemporalConvNet(nn.Module):
    '''
    param num_inputs: int， 输入通道数
    param num_channels: list，每层的hidden_channel数
    '''
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        # num_channels是个列表，其长度就是TemporalBlock的数量
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)  # 输入也可以是list,然后输入的时候用*来引用

    def forward(self, x):
        return self.network(x)


# GAT
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c, alpha=0.2):

        """
        graph attention layer
        :param in_c:
        :param out_c:
        :param alpha:
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_c, out_c)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_c, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.We = nn.Parameter(torch.empty(size=(out_c, 1)))

    def forward(self, features, adj, distance):
        """
        :param features: [B, N, T*H]
        :param adj: [N, N]
        :param distance: [N, N, H]
        :return: [B, N, out_features]
        """
        B, N = features.size(0), features.size(1)
        adj = adj + torch.eye(N, dtype=adj.dtype).to(device)  # A + I
        h = torch.matmul(features, self.W)  # [B,N,out_features]
        # [B, N, N, 2*out_features]
        a_input = torch.cat([h.repeat(1, 1, N).view(B, N*N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2*self.out_c)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [B,N,N,1] => [B,N,N]
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # True则e ,False则zero_vec # [B,N,N]
        cij = torch.matmul(distance, self.We).squeeze(2)
        # print(attention.shape)
        # print(cij.shape)
        attention = attention + cij
        attention = F.softmax(attention, dim=2)  # softmax [N,N]
        # attention = F.dropout(attention, 0.5)
        h_prime = torch.matmul(attention, h)  # [B,N,N] * [N, out_feature] => [B,N,out_feature]
        return h_prime

    def __repr__(self):
        return self.__class__.__name__+'(' + str(self.in_features) + '->' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        """
        super(GAT, self).__init__()
        self.attentions = nn.ModuleList([GraphAttentionLayer(12*in_c, hid_c) for _ in range(n_heads)])
        # self.conv1 = GraphAttentionLayer(in_c, hid_c)
        self.conv2 = GraphAttentionLayer(hid_c * n_heads, out_c)
        self.act = nn.ReLU()

    def forward(self, x, adj, distance):
        # data prepare
        adj = adj   # [N,N]
        x = x     # [B, N, T, H]
        B, N, T = x.size(0), x.size(1), x.size(2)
        # print(x.shape)
        x = x.reshape(B, N, -1)  # [B,N,T*H]
        # print('rrr', x.shape)
        # forward
        outputs = torch.cat([attention(x, adj, distance) for attention in self.attentions], dim=-1)  # [B,N,H]
        outputs = self.act(outputs)
        # outputs_1 = self.act(self.conv1(flow_x, adj))
        outputs_2 = self.act(self.conv2(outputs, adj, distance))   # [B,N,H]
        # outputs_2 = outputs_2.view(B, N, T, -1)

        return outputs_2.unsqueeze(2)  # [B,N,T,H]


class MiLearner(nn.Module):
    def __init__(self, n_hist, n_in, node_dim, dropout):
        super(MiLearner, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.nodes = nn.Sequential(nn.Conv2d(n_in, 64, kernel_size=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(64, node_dim, kernel_size=(1, n_hist))
        )

    def forward(self, inputs):
        """
        :param inputs: tensor, [B, N, T, H]
        # :param supports: tensor, [E, N, N]
        :return: tensor, [E, B, N, N]
        """
        x = inputs.permute(0, 3, 1, 2)  # b h n t
        B, N, H = x.size(0), x.size(2), x.size(1)

        nodes = self.nodes(x).transpose(1, 2)  # b n f t
        nodes = nodes.reshape(N, -1)

        self.dropout(nodes)

        m = nodes
        A_mi = torch.einsum('ud,vd->uv', [m, m])  # b n n

        # print(A_mi.shape)
        return A_mi


# class ls_fusion_module(nn.Module):
#     def __init__(self, embed_size, heads):
#         super(ls_fusion_module, self).__init__()
#         self.attention = FMultiHeadAttention(embed_size, heads)
#
#     def forward(self, h, t, y):
#         '''
#         h: [batch_size, N, T, C]
#         t: [batch_size, N, T, C]
#         y: [batch_size, N, T, 2*C]
#         '''
#         outputs = self.attention(h, t, y) # [b,n,t,2*h]
#         return outputs


class STLayer(nn.Module):
    def __init__(self, adj, embed_size, nodes, num_channels, heads, dropout=0):
        super(STLayer, self).__init__()
        self.nodes = nodes
        self.adj = adj

        self.tcn1 = TemporalConvNet(num_inputs=embed_size, num_channels=num_channels, kernel_size=3, dropout=dropout)
        self.tcn2 = TemporalConvNet(num_inputs=embed_size, num_channels=num_channels, kernel_size=2, dropout=dropout)
        self.tcngate = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()
        )
        self.gatgate = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()
        )

        self.TTransformer = TEncoderLayer(embed_size, heads)
        # self.fusion = ls_fusion_module(embed_size, heads)

        self.GAT = GAT(in_c=embed_size,  hid_c=embed_size, out_c=embed_size, n_heads=heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )

        self.feed_forward1 = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)

        self.norm5 = nn.LayerNorm(embed_size)
        self.norm6 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, embed_size),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, embed_size),
        )

    def forward(self, x, new_adj, distance):  #X[B, N, T, H]
        res_S = x
        x_tcn = x
        res_T = x

        B, N, T, H = x.shape


       ##################时间

        x_T = self.TTransformer(x)
        x_tcn = x_tcn.reshape(B, N*T, H).permute(0, 2, 1)  # [B, H, N*T]
        x_tcn = torch.tanh(self.tcn1(x_tcn).permute(0, 2, 1).reshape(B, N, T, H))*torch.sigmoid(self.tcn2(x_tcn).permute(0,2,1).reshape(B,N,T,H))
        x_T = x_T * self.tcngate(x_tcn)
        x_T = self.dropout(self.norm1(x_T + res_T))
        x_feed = self.feed_forward(x_T)
        x_T = self.dropout(self.norm2(x_feed + x_T))


        #################空间


        # x_S_adj = self.GAT(x, new_adj[0], distance)

        # x_S_adp = self.GAT(x, new_adj[1])
        # z = torch.sigmoid(self.fc1(x_S_adj) + self.fc2(x_S_adp))
        # x_gat = z * x_S_adj + (1-z) * x_S_adp
        x_gcn_F = torch.tanh(self.GAT(x, new_adj[0], distance)) * torch.sigmoid(self.GAT(x, new_adj[1], distance))
        x_gcn_D = self.GAT(x, new_adj[2], distance)
        g = torch.sigmoid(self.fc1(x_gcn_F) + self.fc2(x_gcn_D))
        x_gat = g * x_gcn_F + (1 - g) * x_gcn_D

        x_S = self.gatgate(x_gat)
        x_S = self.dropout(self.norm3(x_S+res_S))
        x_sfeed = self.feed_forward1(x_S)
        x_S = self.dropout(self.norm4(x_sfeed+x_S))

        return x_T, x_S


class NET(nn.Module):
    pos_embed: Parameter

    def __init__(self, adj, distance,embed_size, nodes, num_channels, heads, dropout=0):
        super(NET, self).__init__()
        self.nodes = nodes
        self.adj = adj
        self.distance = distance  # N*N
        self.ST1 = STLayer(adj, embed_size, nodes, num_channels, heads, dropout)
        self.ST2 = STLayer(adj, embed_size, nodes, num_channels, heads, dropout)
        self.ST3 = STLayer(adj, embed_size, nodes, num_channels, heads, dropout)
        self.ST4 = STLayer(adj, embed_size, nodes, num_channels, heads, dropout)

        self.conv1 = nn.Conv2d(1, embed_size, 1)
        self.conv2 = nn.Conv2d(12, 12, 1)
        self.conv3 = nn.Conv2d(embed_size, 1, 1)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)

        self.relu = nn.ReLU()

        self.pos_embed = nn.Parameter(torch.zeros(1, nodes, 12, embed_size), requires_grad=True)
        self.distance_embed = nn.Parameter(torch.zeros(nodes, nodes, embed_size), requires_grad=True)

        self.feed_forward1 = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

        self.feed_forward2 = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

        self.node_embeddings = nn.Parameter(torch.randn(self.nodes, embed_size), requires_grad=True)

        self.A_mi = MiLearner(n_hist=1, n_in=embed_size, node_dim=embed_size, dropout=dropout)

        self.fc1 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, embed_size),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, embed_size),
        )


    def forward(self, x):  # B T N

        x = x.permute(0, 2, 1).unsqueeze(1)  # B H N T
        input = self.conv1(x)
        input = input.permute(0, 2, 3, 1)  # B N T H
        input = input + self.pos_embed

        T_SKIP = 0
        S_SKIP = 0
        B, N, T, H = input.shape
        distance = self.distance # N N
        distance = distance.unsqueeze(0)
        distance = self.conv1(distance).permute(1, 2, 0)
        distance = distance + self.distance_embed
        # print('---------')
        # print(distance.shape)

        adp = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        A_mi = self.A_mi(input)
        A_mi = F.softmax(F.relu(A_mi))
        new_adj = self.adj + [adp] + [A_mi ]

        x_T, x_S = self.ST1(input, new_adj, distance)  # x_T[B, N, T, 2*H] x_S[B, N, T, H]
        T_SKIP = x_T + T_SKIP
        S_SKIP = x_S + S_SKIP
        x_T, x_S = self.ST2(input, new_adj, distance)
        T_SKIP = x_T + T_SKIP
        S_SKIP = x_S + S_SKIP
        x_T, x_S = self.ST3(input, new_adj, distance)
        T_SKIP = x_T + T_SKIP
        S_SKIP = x_S + S_SKIP
        x_T, x_S = self.ST4(input, new_adj, distance)
        T_SKIP = x_T + T_SKIP
        S_SKIP = x_S + S_SKIP

        # feed_T = self.feed_forward1(T_SKIP)
        # T_SKIP = self.norm1(feed_T+T_SKIP)
        #
        # feed_S = self.feed_forward2(S_SKIP)
        # S_SKIP = self.norm2(feed_S+S_SKIP)

        # feed_T = self.feed_forward1(self.norm1(T_SKIP))
        O_tt = self.fc1(T_SKIP)   # x_T[B, N, T, H]

        # feed_S = self.feed_forward2(self.norm2(S_SKIP))
        O_st = self.fc2(S_SKIP)   # x_S[B, N, T, H]

        gate = torch.sigmoid(O_tt + O_st)

        x = gate*O_tt + (1-gate)*O_st

        #####################################
        out = x.permute(0, 2, 1, 3)  # B T N C
        out = self.relu(self.conv2(out))
        out = out.permute(0, 3, 2, 1)  # B C N T
        out = self.conv3(out)
        out = out.squeeze(1)  # B N T
        # print(out.shape)
        return out.permute(0, 2, 1)