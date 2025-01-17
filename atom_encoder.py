import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)


def create_padding_mask_atom(batch_data):
    padding_mask = (batch_data.sum(dim=-1) == 0).float()
    return padding_mask.unsqueeze(1).unsqueeze(2)


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
        """Split the last dimension into (num_heads, depth)."""
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

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, adjoin_matrix, dist_matrix)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = scaled_attention.contiguous().view(batch_size, -1,
                                                              self.d_model)

        output = self.dense(concat_attention)

        return output, attention_weights


def scaled_dot_product_attention(q, k, v, mask, adjoin_matrix, dist_matrix):
    if dist_matrix is not None:
        matmul_qk = torch.relu(torch.matmul(q, k.transpose(-2, -1)))
        dist_matrix = rescale_distance_matrix(dist_matrix)
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_attention_logits = (matmul_qk * dist_matrix) / torch.sqrt(dk)
    else:
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        dk = torch.tensor(k.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    if adjoin_matrix is not None:
        scaled_attention_logits += adjoin_matrix

    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = nn.Linear(d_model, dff)
        self.dense2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = gelu(self.dense1(x))
        return self.dense2(x)


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

        x_l, attention_weights_local = self.mha1(x1, x1, x1, encoder_padding_mask, adjoin_matrix, None) # Done
        x_g, attention_weights_global = self.mha2(x2, x2, x2, encoder_padding_mask, None, dist_matrix)  # Done

        attn_output = torch.cat([x_l, x_g], dim=-1)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2, attention_weights_local, attention_weights_global


class EncoderModelAtom(nn.Module):
    def __init__(self, input_dim, num_layers, d_model, num_heads, dff, rate=0.1):
        super(EncoderModelAtom, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_dim, d_model)
        self.global_embedding = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(rate)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, rate)
            for _ in range(self.num_layers)])

    def forward(self, x, training, adjoin_matrix=None, dist_matrix=None, atom_match_matrix=None, sum_atoms=None):

        encoder_padding_mask = create_padding_mask_atom(x)

        if adjoin_matrix is not None:
            adjoin_matrix = adjoin_matrix.unsqueeze(1)
        if dist_matrix is not None:
            dist_matrix = dist_matrix.unsqueeze(1)

        x = self.embedding(x)
        x = self.dropout(x)

        attention_weights_list_local = []
        attention_weights_list_global = []

        for i in range(self.num_layers):
            x, attention_weights_local, attention_weights_global = self.encoder_layers[i](
                x, training, encoder_padding_mask, adjoin_matrix, dist_matrix)
            attention_weights_list_local.append(attention_weights_local)
            attention_weights_list_global.append(attention_weights_global)
        x = torch.matmul(atom_match_matrix, x) / sum_atoms
        x = self.global_embedding(x)
        return x, attention_weights_list_local, attention_weights_list_global, encoder_padding_mask
