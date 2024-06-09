import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(TScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        B, n_heads, len1, len2, d_k = Q.shape
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn,
                               V)
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

        Q = self.W_Q(input_Q).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        K = self.W_K(input_K).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)
        V = self.W_V(input_V).view(B, N, T, self.heads, self.head_dim).permute(0, 3, 1, 2, 4)




        context = TScaledDotProductAttention()(Q, K, V)
        context = context.permute(0, 2, 3, 1, 4)
        context = context.reshape(B, N, T, self.heads * self.head_dim)

        output = self.fc_out(context)
        return output


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, embed_size):
        super(PoswiseFeedForwardNet, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, output):
        residual = output
        output = self.fc(output)
        output = self.norm(output + residual)
        return output

class TEncoderLayer(nn.Module):
    def __init__(self, embed_size, heads):
        super(TEncoderLayer, self).__init__()
        self.enc_self_attn = TMultiHeadAttention(embed_size, heads)
        self.pos_ffn = PoswiseFeedForwardNet(embed_size)

    def forward(self, inputs):
        outputs = self.enc_self_attn(inputs, inputs, inputs)
        outputs = self.pos_ffn(outputs)

        return outputs

class Chomp1d(nn.Module):
    '''
    '''
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    '''

    '''
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

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
        adj = adj + torch.eye(N, dtype=adj.dtype).to(device)
        h = torch.matmul(features, self.W)

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N*N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1, 2*self.out_c)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
        zero_vec = -1e12 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        cij = torch.matmul(distance, self.We).squeeze(2)
        attention = attention + cij
        attention = F.softmax(attention, dim=2)
        h_prime = torch.matmul(attention, h)
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
        self.conv2 = GraphAttentionLayer(hid_c * n_heads, out_c)
        self.act = nn.ReLU()

    def forward(self, x, adj, distance):
        adj = adj
        x = x
        B, N, T = x.size(0), x.size(1), x.size(2)

        x = x.reshape(B, N, -1)

        outputs = torch.cat([attention(x, adj, distance) for attention in self.attentions], dim=-1)
        outputs = self.act(outputs)
        outputs_2 = self.act(self.conv2(outputs, adj, distance))

        return outputs_2.unsqueeze(2)

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
        x = inputs.permute(0, 3, 1, 2)
        B, N, H = x.size(0), x.size(2), x.size(1)

        nodes = self.nodes(x).transpose(1, 2)
        nodes = nodes.reshape(N, -1)
        self.dropout(nodes)

        m = nodes

        # M_mean = np.mean(m, axis=1)
        # M = m - M_mean
        # M_norm = np.linalg.norm(M, axis=-1, keepdims=True)
        # M_norm[M_norm == 0] = 1
        # m = M / M_norm

        A_mi = torch.einsum('ud,vd->uv', [m, m])

        return A_mi


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

    def forward(self, x, new_adj, distance):
        res_S = x
        x_tcn = x
        res_T = x

        B, N, T, H = x.shape

        x_T = self.TTransformer(x)
        x_tcn = x_tcn.reshape(B, N*T, H).permute(0, 2, 1)
        x_tcn = torch.tanh(self.tcn1(x_tcn).permute(0, 2, 1).reshape(B, N, T, H))*torch.sigmoid(self.tcn2(x_tcn).permute(0,2,1).reshape(B,N,T,H))
        x_T = x_T * self.tcngate(x_tcn)
        x_T = self.dropout(self.norm1(x_T + res_T))
        x_feed = self.feed_forward(x_T)
        x_T = self.dropout(self.norm2(x_feed + x_T))


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
        self.distance = distance
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


    def forward(self, x):

        x = x.permute(0, 2, 1).unsqueeze(1)
        input = self.conv1(x)
        input = input.permute(0, 2, 3, 1)
        input = input + self.pos_embed

        T_SKIP = 0
        S_SKIP = 0
        B, N, T, H = input.shape
        distance = self.distance
        distance = distance.unsqueeze(0)
        distance = self.conv1(distance).permute(1, 2, 0)
        distance = distance + self.distance_embed

        adp = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        A_mi = self.A_mi(input)
        A_mi = F.softmax(F.relu(A_mi))
        new_adj = self.adj + [adp] + [A_mi ]

        x_T, x_S = self.ST1(input, new_adj, distance)
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

        O_tt = self.fc1(T_SKIP)

        O_st = self.fc2(S_SKIP)

        gate = torch.sigmoid(O_tt + O_st)

        x = gate*O_tt + (1-gate)*O_st

        #####################################
        out = x.permute(0, 2, 1, 3)
        out = self.relu(self.conv2(out))
        out = out.permute(0, 3, 2, 1)
        out = self.conv3(out)
        out = out.squeeze(1)
        return out.permute(0, 2, 1)