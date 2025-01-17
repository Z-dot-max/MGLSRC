import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Helper functions
def rescale_distance_matrix(w):  # For global
    constant_value = torch.tensor(1.0, dtype=torch.float32)
    return (constant_value + torch.exp(constant_value)) / (constant_value + torch.exp(constant_value - w))


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.))))


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)


def create_padding_mask(batch_data):
    return (batch_data == 0).float().unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]


# MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, q, k, v, mask, adjoin_matrix, dist_matrix):
        batch_size = q.size(0)
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled attention
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    dk = torch.tensor(k.size(-1), dtype=torch.float32)

    if dist_matrix is not None:
        matmul_qk = torch.relu(torch.matmul(q, k.transpose(-2, -1)))
        dist_matrix = rescale_distance_matrix(dist_matrix)
        scaled_attention_logits = (matmul_qk * dist_matrix) / torch.sqrt(dk)
    else:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    attention_weights = attention_weights.to(torch.float32)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


# FeedForward Network
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = nn.Linear(d_model, dff)
        self.dense2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = gelu(self.dense1(x))
        return self.dense2(x)


# EncoderLayer class
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, rate):
        super(EncoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model // 2, num_heads)
        self.mha2 = MultiHeadAttention(d_model // 2, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x, training, encoder_padding_mask, adjoin_matrix, dist_matrix):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x_l, attention_weights_local = self.mha1(x1, x1, x1, encoder_padding_mask, adjoin_matrix, None)
        x_g, attention_weights_global = self.mha2(x2, x2, x2, encoder_padding_mask, None, dist_matrix)
        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)
        return out2, attention_weights_local, attention_weights_global


# Encoder Model
class EncoderModelMotif(nn.Module):
    def __init__(self, num_layers, input_vocab_size, d_model, num_heads, dff, rate=0.1):
        super(EncoderModelMotif, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.dropout = nn.Dropout(rate)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)])

    def forward(self, x, training, atom_level_features, adjoin_matrix=None, dist_matrix=None):
        x = x.long()
        encoder_padding_mask = create_padding_mask(x)
        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.dropout(x)

        x_temp = x[:, 1:, :] + atom_level_features
        x = torch.cat([x[:, 0:1, :], x_temp], dim=1)

        attention_weights_list_local = []
        attention_weights_list_global = []

        for i in range(self.num_layers):
            x, attention_weights_local, attention_weights_global = self.encoder_layers[i](
                x, training, encoder_padding_mask, adjoin_matrix, dist_matrix
            )
            attention_weights_list_local.append(attention_weights_local)
            attention_weights_list_global.append(attention_weights_global)

        return x, attention_weights_list_local, attention_weights_list_global, encoder_padding_mask


# Co-Attention Layer
class CoAttentionLayer(nn.Module):
    def __init__(self, graph_feat_size, k):
        super(CoAttentionLayer, self).__init__()
        self.k = k
        self.graph_feat_size = graph_feat_size

        # 定义各个权重矩阵
        self.W_m = nn.Parameter(torch.Tensor(k, graph_feat_size))  # 交互项权重
        self.W_v = nn.Parameter(torch.Tensor(k, graph_feat_size))  # 值矩阵权重
        self.W_q = nn.Parameter(torch.Tensor(k, graph_feat_size))  # 查询矩阵权重
        self.W_h = nn.Parameter(torch.Tensor(1, k))  # 最后的权重，用于计算注意力

        # 初始化权重
        nn.init.xavier_uniform_(self.W_m)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_h)

    def forward(self, inputs):
        V_n, Q_n = inputs

        # 处理第一个 token
        V_0 = V_n[:, 0, :].unsqueeze(2)  # (batch_size, graph_feat_size, 1)
        Q_0 = Q_n[:, 0, :].unsqueeze(2)  # (batch_size, graph_feat_size, 1)

        # 处理剩余的部分
        V_r = V_n[:, 1:, :].transpose(1, 2)  # (batch_size, graph_feat_size, seq_len-1)
        Q_r = Q_n[:, 1:, :].transpose(1, 2)  # (batch_size, graph_feat_size, seq_len-1)

        # 计算交互项 M_0
        M_0 = V_0 * Q_0  # (batch_size, graph_feat_size, 1)

        # 计算 H_v 和 H_q
        H_v = torch.tanh(torch.matmul(self.W_v, V_r)) * torch.tanh(torch.matmul(self.W_m, M_0))
        H_q = torch.tanh(torch.matmul(self.W_q, Q_r)) * torch.tanh(torch.matmul(self.W_m, M_0))

        # 计算注意力分数 alpha_v 和 alpha_q
        alpha_v = F.softmax(torch.matmul(self.W_h, H_v), dim=-1)
        alpha_q = F.softmax(torch.matmul(self.W_h, H_q), dim=-1)

        # 最终的注意力向量
        vector_v = torch.matmul(alpha_v, V_r.transpose(1, 2))  # (batch_size, 1, graph_feat_size)
        vector_q = torch.matmul(alpha_q, Q_r.transpose(1, 2))  # (batch_size, 1, graph_feat_size)

        return vector_v.squeeze(1), vector_q.squeeze(1), alpha_v, alpha_q

class CrossAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, out_seq_reactant, out_seq_product):
        out_seq_reactant = out_seq_reactant.transpose(0, 1)  # (seq_len_reactant, batch_size, embed_size)
        out_seq_product = out_seq_product.transpose(0, 1)    # (seq_len_product, batch_size, embed_size)

        mask = (out_seq_product.sum(dim=-1) == 0).transpose(0, 1)

        # Step 2: Perform cross-attention (out_seq_reactant is query, out_seq_product is key, value)
        attention_output, attention_weights = self.attention(out_seq_reactant, out_seq_product, out_seq_product, key_padding_mask=mask)

        # Step 3: Apply the final linear transformation to the attention output
        output = self.fc_out(attention_output)

        # Step 4: Return the output (reshaped back to (batch_size, seq_len_reactant, embed_size))
        return output.transpose(0, 1), attention_weights