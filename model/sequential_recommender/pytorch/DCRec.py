
import logging
import time
from collections import defaultdict,Counter
from copy import deepcopy
import scipy.sparse as sp
import random
import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model.base import AbstractRecommender
import dgl
from dgl.nn.pytorch import GraphConv
import math
from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from util.common.tool import batch_randint_choice
from util.common.tool import pad_sequences
from data import TimeOrderPairwiseSampler
# from data.data_iterator import DataIterator
from reckit import DataIterator


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output

class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class CLLayer(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.5):
        super().__init__()
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden, num_hidden)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def pair_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.sum(z1 * z2, dim=1)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def vanilla_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        pos_pairs = f(self.sim(z1, z2)).diag()
        neg_pairs = f(self.sim(z1, z2)).sum(1)
        return -torch.log(1e-8 + pos_pairs / neg_pairs)

    def vanilla_loss_overall(self, z1, z2, z_2_all):
        f = lambda x: torch.exp(x / self.tau)
        pos_pairs = f(self.pair_sim(z1, z2))
        neg_pairs = f(self.sim(z1, z_2_all)).sum(1)
        return -torch.log(pos_pairs / neg_pairs)

    def vanilla_loss_with_one_negative(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        pos_pairs = f(self.sim(z1, z2)).diag()
        neg_pairs = f(self.sim(z1, z2))
        rand_pairs = torch.randperm(neg_pairs.size(1))
        neg_pairs = neg_pairs[torch.arange(0, neg_pairs.size(0)), rand_pairs] + neg_pairs.diag()
        return -torch.log(pos_pairs / neg_pairs)

    def grace_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                   mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        # ret = ret.mean() if mean else ret.sum()

        return ret

    def push_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            f(torch.tensor(1)) / between_sim.diag()
        )


def cal_kl(target, input):
    ### log with sigmoid
    target = torch.sigmoid(target)
    input = torch.sigmoid(input)
    target = torch.log(target + 1e-8)
    input = torch.log(input + 1e-8)
    return F.kl_div(input, target, reduction='batchmean', log_target=True)

def cal_kl_1(target, input):
    target[target<1e-8] = 1e-8
    target = torch.log(target + 1e-8)
    input = torch.log_softmax(input + 1e-8, dim=0)
    return F.kl_div(input, target, reduction='batchmean', log_target=True)


def graph_dual_neighbor_readout(g: dgl.DGLGraph, aug_g: dgl.DGLGraph, node_ids, features):
    _, all_neighbors = g.out_edges(node_ids)
    all_nbr_num = g.out_degrees(node_ids)
    _, foreign_neighbors = aug_g.out_edges(node_ids)
    for_nbr_num = aug_g.out_degrees(node_ids)
    all_neighbors = [set(t.tolist())
                     for t in all_neighbors.split(all_nbr_num.tolist())]
    foreign_neighbors = [set(t.tolist())
                         for t in foreign_neighbors.split(for_nbr_num.tolist())]
    # sample foreign neighbors
    for i, nbrs in enumerate(foreign_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            foreign_neighbors[i] = set(nbrs)
    civil_neighbors = [all_neighbors[i]-foreign_neighbors[i]
                       for i in range(len(all_neighbors))]
    # sample civil neighbors
    for i, nbrs in enumerate(civil_neighbors):
        if len(nbrs) > 10:
            nbrs = random.sample(nbrs, 10)
            civil_neighbors[i] = set(nbrs)
    for_lens = [len(t) for t in foreign_neighbors]
    cv_lens = torch.tensor([len(t)
                           for t in civil_neighbors], dtype=torch.int16)
    zero_indicies = (cv_lens == 0).nonzero().view(-1).tolist()
    cv_lens = cv_lens[cv_lens > 0].tolist()
    foreign_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in foreign_neighbors])
    civil_neighbors = torch.cat(
        [torch.tensor(list(s), dtype=torch.long) for s in civil_neighbors])
    cv_feats = features[civil_neighbors].split(cv_lens)
    cv_feats = [t.mean(dim=0) for t in cv_feats]
    # insert zero vector for zero-length neighbors
    if len(zero_indicies) > 0:
        for i in zero_indicies:
            cv_feats.insert(i, torch.zeros_like(features[0]))
    for_feats = features[foreign_neighbors].split(for_lens)
    for_feats = [t.mean(dim=0) for t in for_feats]
    return torch.stack(cv_feats, dim=0), torch.stack(for_feats, dim=0)


def graph_augment(g: dgl.DGLGraph, user_ids, user_edges):
    # Augment the graph with the item sequence, deleting co-occurrence edges in the batched sequences
    # generating indicies like: [1,2] [2,3] ... as the co-occurrence rel.
    # indexing edge data using node indicies and delete them
    # for edge weights, delete them from the raw data using indexed edges
    user_ids = user_ids.cpu().numpy()
    node_indicies_a = np.concatenate(
        user_edges.loc[user_ids, "item_edges_a"].to_numpy())
    node_indicies_b = np.concatenate(
        user_edges.loc[user_ids, "item_edges_b"].to_numpy())
    node_indicies_a = torch.from_numpy(
        node_indicies_a).to(g.device)
    node_indicies_b = torch.from_numpy(
        node_indicies_b).to(g.device)
    edge_ids = g.edge_ids(node_indicies_a, node_indicies_b)

    aug_g: dgl.DGLGraph = deepcopy(g)
    # The features for the removed edges will be removed accordingly.
    aug_g.remove_edges(edge_ids)

    return aug_g


def graph_dropout(g: dgl.DGLGraph, keep_prob):
    # Firstly mask selected edge values, returns the true values along with the masked graph.
    origin_edge_w = g.edata['w']

    drop_size = int((1-keep_prob) * g.num_edges())
    random_index = torch.randint(
        0, g.num_edges(), (drop_size,), device=g.device)
    mask = torch.zeros(g.num_edges(), dtype=torch.uint8,
                       device=g.device).bool()
    mask[random_index] = True
    g.edata['w'].masked_fill_(mask, 0)

    return origin_edge_w, g


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_prob=0.7):
        super(GCN, self).__init__()
        self.dropout_prob = dropout_prob
        self.layer = GraphConv(in_dim, out_dim, weight=False,
                               bias=False, allow_zero_in_degree=False)

    def forward(self, graph, feature):
        graph = dgl.add_self_loop(graph)
        origin_w, graph = graph_dropout(graph, 1-self.dropout_prob)
        embs = [feature]
        for i in range(2):
            feature = self.layer(graph, feature, edge_weight=graph.edata['w'])
            F.dropout(feature, p=0.2, training=self.training)
            embs.append(feature)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        # recover edge weight
        graph.edata['w'] = origin_w
        return final_emb


class _DCRec(nn.Module):

    def __init__(self, config, n_users, n_items, max_seq_length, external_data):
        super(_DCRec, self).__init__()

        self.config = config
        self.n_items = n_items
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.max_seq_length = max_seq_length
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['embedding_size']  # same as embedding_size
        # the dimensionality in feed-forward layer
        self.inner_size = config['inner_size']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        # load dataset info
        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(
            self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.LayerNorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # Contrastive Learning
        self.contrastive_learning_layer = CLLayer(self.hidden_size, tau=config['cl_temp'])

        # Fusion Attn
        self.attn_weights = nn.Parameter(
            torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)

        # Global Graph Learning
        self.item_adjgraph = external_data["adj_graph"].to(self.device)
        self.user_edges = external_data["user_edges"]
        self.item_simgraph = external_data["sim_graph"].to(self.device)
        self.graph_dropout = config["graph_dropout_prob"]

        self.adj_graph_test = external_data["adj_graph_test"].to(self.device)
        self.sim_graph_test = external_data["sim_graph_test"].to(self.device)

        self.gcn = GCN(self.hidden_size, self.hidden_size, self.graph_dropout)

        self.layernorm = nn.LayerNorm(
            self.hidden_size, eps=self.layer_norm_eps)

        self.loss_fct = nn.CrossEntropyLoss()

        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' be CE!")

        # parameters initialization
        self.apply(self._init_weights)

    def _subgraph_agreement(self, aug_g, raw_output_all, raw_output_seq, valid_items_flatten):
        # here it firstly removes items of the sequence in the cooccurrence graph, and then performs the gnn aggregation, and finally calculates the item-wise agreement score.
        aug_output_seq = self.gcn_forward(g=aug_g)[valid_items_flatten]
        civil_nbr_ro, foreign_nbr_ro = graph_dual_neighbor_readout(
            self.item_adjgraph, aug_g, valid_items_flatten, raw_output_all)

        view1_sim = F.cosine_similarity(
            raw_output_seq, aug_output_seq, eps=1e-12)
        view2_sim = F.cosine_similarity(
            raw_output_seq, foreign_nbr_ro, eps=1e-12)
        view3_sim = F.cosine_similarity(
            civil_nbr_ro, foreign_nbr_ro, eps=1e-12)
        agreement = (view1_sim+view2_sim+view3_sim)/3
        agreement = torch.sigmoid(agreement)
        agreement = (agreement - agreement.min()) / \
            (agreement.max() - agreement.min())
        agreement = (self.config["weight_mean"] / agreement.mean()) * agreement
        return agreement

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq, task_label=False):
        """Generate bidirectional attention mask for multi-head attention."""
        if task_label:
            label_pos = torch.ones((item_seq.size(0), 1), device=self.device)
            item_seq = torch.cat((label_pos, item_seq), dim=1)
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(
            1).unsqueeze(2)  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def _padding_sequence(self, sequence, max_length):
        # 0在后面的mask, 和原版BERT4Rec不同.
        pad_len = max_length - len(sequence)
        sequence = sequence + [0] * pad_len
        return sequence

    def gcn_forward(self, g=None):
        item_emb = self.item_embedding.weight
        item_emb = self.dropout(item_emb)
        light_out = self.gcn(g, item_emb)
        return self.layernorm(light_out+item_emb)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        # print(gather_index)
        output_tensor = output.gather(dim=1, index=gather_index)
        # print(output_tensor)
        return output_tensor.squeeze(1)

    def forward(self, item_seq, item_seq_len, return_all=False):
        # if type(item_seq_len) == int:
        #     item_seq_len = torch.tensor(item_seq_len)
        position_ids = torch.arange(item_seq.size(
            1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        if return_all:
            return output
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, bat_users, bat_item_seq, bat_item_pos, L):
        if self.config["graphcl_enable"]:
            return self.calculate_loss_graphcl(bat_users, bat_item_seq, bat_item_pos, L)

        item_seq = bat_item_seq
        last_item = bat_item_pos
        masked_item_seq, pos_items = self.reconstruct_train_data(
                item_seq, last_item=last_item)
        seq_output = self.forward(masked_item_seq)

        masked_index = (masked_item_seq == self.mask_token)
        # [mask_num, H]
        seq_output = seq_output[masked_index]
        # [item_num, H]
        # [item_num H]
        test_item_emb = self.item_embedding.weight[:self.n_items]
        # [mask_num, item_num]
        logits = torch.mm(seq_output, test_item_emb.transpose(0, 1))

        loss = self.loss_fct(logits, pos_items)

        if torch.isnan(loss):
            print(masked_item_seq.tolist())
            print(masked_index.tolist())
            input()
        return loss

    def calculate_loss_graphcl(self, bat_users, bat_item_seq, bat_item_pos, L):
        user_ids = bat_users
        item_seq = bat_item_seq
        pos_items = bat_item_pos
        item_seq_len = L
        last_items_indices = torch.tensor([i * self.max_seq_length + j for i, j in enumerate(
            item_seq_len - 1)], dtype=torch.long, device=item_seq.device).view(-1)
        # print(item_seq_len)
        # print(last_items_indices)
        # only the last one
        last_items_flatten = item_seq.view(-1)[last_items_indices]
        # print(item_seq[0])
        # print(item_seq[1])
        # print(last_items_flatten)
        valid_items_flatten = last_items_flatten
        valid_items_indices = last_items_indices
        # print(valid_items_flatten)
        # graph view
        masked_g = self.item_adjgraph
        aug_g = graph_augment(self.item_adjgraph, user_ids, self.user_edges)
        iadj_graph_output_raw = self.gcn_forward(masked_g)
        isim_graph_output_raw = self.gcn_forward(self.item_simgraph)
        iadj_graph_output_seq = iadj_graph_output_raw[valid_items_flatten]
        isim_graph_output_seq = isim_graph_output_raw[valid_items_flatten]

        seq_output = self.forward(item_seq, item_seq_len, return_all=False)
        aug_seq_output = self.forward(item_seq, item_seq_len, return_all=True).view(
            -1, self.config["embedding_size"])[valid_items_indices]
        # First-stage CL, providing CL weights
        # CL weights from augmentation
        mainstream_weights = self._subgraph_agreement(
            aug_g, iadj_graph_output_raw, iadj_graph_output_seq, valid_items_flatten)
        # filtering those len=1, set weight=0.5
        mainstream_weights[item_seq_len == 1] = 0.5

        expected_weights_distribution = torch.normal(self.config["weight_mean"], 0.1, size=mainstream_weights.size()).to(self.device)
        # kl_loss = self.config["kl_weight"] * cal_kl(expected_weights_distribution.sort()[0], mainstream_weights.sort()[0])

        # apply log_softmax for input
        kl_loss = self.config["kl_weight"] * cal_kl_1(expected_weights_distribution.sort()[0], mainstream_weights.sort()[0])
        # if torch.isnan(kl_loss):
        #     logging.info("kl_loss: {}".format(kl_loss))
        #     logging.info("mainstream_weights: {}".format(
        #         mainstream_weights.cpu().tolist()))
        #     logging.info("expected_weights_distribution: {}".format(
        #         expected_weights_distribution.cpu().tolist()))
        #     raise ValueError("kl loss is nan")

        personlization_weights = mainstream_weights.max() - mainstream_weights

        # contrastive learning
        if self.config["cl_ablation"] == "full":
            cl_loss_adj = self.contrastive_learning_layer.vanilla_loss(
                aug_seq_output, iadj_graph_output_seq)
            cl_loss_a2s = self.contrastive_learning_layer.vanilla_loss(
                iadj_graph_output_seq, isim_graph_output_seq)
            cl_loss = (self.config["graphcl_coefficient"] * (mainstream_weights *
                       cl_loss_adj + personlization_weights * cl_loss_a2s)).mean()
            # if torch.isnan(cl_loss):
            #     logging.error("cl_loss_adj: {}".format(cl_loss_adj.cpu().tolist()))
            #     logging.error("cl_loss_a2s: {}".format(cl_loss_a2s.cpu().tolist()))
            #     logging.error("mainstream_weights: {}".format(mainstream_weights.cpu().tolist()))
            #     logging.error("personlization_weights: {}".format(personlization_weights.cpu().tolist()))
            #     logging.error("cl loss is nan")
            #     raise ValueError("cl loss is nan")
        # Fusion After CL
        if self.config["graph_view_fusion"]:
            # 3, N_mask, dim
            mixed_x = torch.stack(
                (seq_output, iadj_graph_output_raw[last_items_flatten], isim_graph_output_raw[last_items_flatten]), dim=0)
            weights = (torch.matmul(
                mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
            # 3, N_mask, 1
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            seq_output = (mixed_x*score).sum(0)
        # [item_num, H]
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits+1e-8, pos_items)


        # if torch.isnan(loss):
        #     logging.error("cl_loss: {}".format(cl_loss))
        #     logging.error("loss is nan")
        return loss, cl_loss, kl_loss

    def fast_predict(self, bat_item_seq, L):
        item_seq = bat_item_seq
        item_seq_len = L
        # test_item = interaction["item_id_with_negs"]
        seq_output = self.forward(item_seq, item_seq_len)
        item_seq_len_2 = [self.max_seq_length for _ in range(len(item_seq))]
        item_seq_len_2 = np.array(item_seq_len_2)
        item_seq_len_2 = torch.from_numpy(item_seq_len_2).long().to(self.device)
        if self.config["graph_view_fusion"]:
            last_items_flatten = torch.gather(
                item_seq, 1, (item_seq_len - 1).unsqueeze(1)).squeeze()
            # print(item_seq)
            # print(last_items_flatten)
            # graph view
            masked_g = self.adj_graph_test
            iadj_graph_output_raw = self.gcn_forward(masked_g)
            iadj_graph_output_seq = iadj_graph_output_raw[last_items_flatten]
            isim_graph_output_seq = self.gcn_forward(self.sim_graph_test)[
                last_items_flatten]
            # 3, N_mask, dim
            mixed_x = torch.stack(
                (seq_output, iadj_graph_output_seq, isim_graph_output_seq), dim=0)
            weights = (torch.matmul(
                mixed_x, self.attn_weights.unsqueeze(0))*self.attn).sum(-1)
            # 3, N_mask, 1
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            seq_output = (mixed_x*score).sum(0)

        # test_item_emb = self.item_embedding(test_item)  # [B, num, H]
        test_item_emb = self.item_embedding.weight
        # scores = torch.matmul(seq_output.unsqueeze(
        #     1), test_item_emb.transpose(1, 2)).squeeze()
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores


def build_adj_graph(dataset, phase="train"):
    import dgl
    print("constructing DGL graph...")
    item_adj_dict = defaultdict(list)
    item_edges_of_user = dict()
    users_dict = dataset.to_user_dict()
    # print(users_dict)
    for key, value in users_dict.items():
        item_edges_a, item_edges_b = [], []
        uid = key
        item_seq = value
        # print(item_seq)
        seq_len = len(item_seq)
        # if seq_len >= 50:
        #     seq_len = 50
        #     item_seq = item_seq[-50:]
        # else:
        #     for _ in range(50 - seq_len):
        #         np.insert(item_seq, 0, dataset.num_items)
        for i in range(seq_len):
            if i > 0:
                item_adj_dict[item_seq[i]].append(item_seq[i-1])
                item_adj_dict[item_seq[i-1]].append(item_seq[i])
                item_edges_a.append(item_seq[i])
                item_edges_b.append(item_seq[i-1])
            if i+1 < seq_len:
                item_adj_dict[item_seq[i]].append(item_seq[i+1])
                item_adj_dict[item_seq[i+1]].append(item_seq[i])
                item_edges_a.append(item_seq[i])
                item_edges_b.append(item_seq[i+1])
        item_edges_of_user[uid] = (np.asarray(item_edges_a, dtype=np.int64), np.asarray(item_edges_b, dtype=np.int64))
    item_edges_of_user = pd.DataFrame.from_dict(item_edges_of_user, orient='index', columns=['item_edges_a', 'item_edges_b'])
    # item_edges_of_user.to_pickle(user_edges_file)
    cols = []
    rows = []
    values = []
    for item in item_adj_dict:
        adj = item_adj_dict[item]
        adj_count = Counter(adj)

        rows.extend([item]*len(adj_count))
        cols.extend(adj_count.keys())
        values.extend(adj_count.values())

    adj_mat = csr_matrix((values, (rows, cols)), shape=(
        dataset.num_items + 1, dataset.num_items + 1))
    adj_mat = adj_mat.tolil()
    adj_mat.setdiag(np.ones((dataset.num_items + 1,)))
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()

    g = dgl.from_scipy(norm_adj, 'w', idtype=torch.int64)
    g.edata['w'] = g.edata['w'].float()
    return g, item_edges_of_user

def build_sim_graph(dataset, k, phase="train"):
    import dgl
    print("building isim graph...")
    row = []
    col = []
    users_dict = dataset.to_user_dict()
    for key, value in users_dict.items():
        uid = key
        item_seq = value
        seq_len = len(item_seq)
        # if seq_len >= 50:
        #     seq_len = 50
        #     item_seq = item_seq[-50:]
        # else:
        #     for _ in range(50-seq_len):
        #         np.insert(item_seq, 0, dataset.num_items)
        # item_seq = pad_sequences(list(item_seq), value=dataset.num_items - 1, max_len=50, padding='pre',
        #                          truncating='pre')
        # item_seq = list(item_seq)
        col.extend(item_seq)
        row.extend([uid]*seq_len)
    row = np.array(row)
    col = np.array(col)
    # n_users, n_items
    cf_graph = csr_matrix(([1]*len(row), (row, col)), shape=(
        dataset.num_users + 1, dataset.num_items + 1), dtype=np.float32)
    similarity = cosine_similarity(cf_graph.transpose())
    # filter topk connections
    sim_items_slices = []
    sim_weights_slices = []
    i = 0
    while i < similarity.shape[0]:
        similarity = similarity[i:, :]
        sim = similarity[:256, :]
        sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
        sim_weights = np.take_along_axis(sim, sim_items, axis=1)
        sim_items_slices.append(sim_items)
        sim_weights_slices.append(sim_weights)
        i = i + 256
    sim = similarity[256:, :]
    sim_items = np.argpartition(sim, -(k+1), axis=1)[:, -(k+1):]
    sim_weights = np.take_along_axis(sim, sim_items, axis=1)
    sim_items_slices.append(sim_items)
    sim_weights_slices.append(sim_weights)

    sim_items = np.concatenate(sim_items_slices, axis=0)
    sim_weights = np.concatenate(sim_weights_slices, axis=0)
    row = []
    col = []
    for i in range(len(sim_items)):
        row.extend([i]*len(sim_items[i]))
        col.extend(sim_items[i])
    values = sim_weights / sim_weights.sum(axis=1, keepdims=True)
    values = np.nan_to_num(values).flatten()
    adj_mat = csr_matrix((values, (row, col)), shape=(
        dataset.num_items + 1, dataset.num_items + 1))
    g = dgl.from_scipy(adj_mat, 'w')
    g.edata['w'] = g.edata['w'].float()
    return g


class DCRec(AbstractRecommender):
    def __init__(self, config):
        super(DCRec, self).__init__(config)
        self.lr = config["lr"]
        self.reg = config["reg"]
        # self.seq_L = config["L"]
        self.max_len = config["max_len"]
        self.seq_T = config["T"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        adj_graph, user_edges = build_adj_graph(self.dataset.train_data)
        adj_graph_test, _ = build_adj_graph(self.dataset.test_data, "test")
        sim_graph = build_sim_graph(self.dataset.train_data, 4)
        sim_graph_test = build_sim_graph(self.dataset.test_data, 4, "test")
        external_data = {
            "adj_graph": adj_graph,
            "sim_graph": sim_graph,
            "user_edges": user_edges,
            "adj_graph_test": adj_graph_test,
            "sim_graph_test": sim_graph_test
        }
        self.pad_idx = self.num_items
        self.num_items += 1
        self.user_pos_train = self.dataset.train_data.to_user_dict(by_time=True)
        self.user_truncated_seq = self.dataset.train_data.to_truncated_seq_dict(None,
                                                                                pad_value=self.pad_idx,
                                                                                padding='pre', truncating='pre')
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.user_pos_test = self.dataset.test_data.to_user_dict(by_time=True)
        self.dcrec = _DCRec(config, self.num_users, self.num_items, self.max_len, external_data).to(self.device)
        self.optimizer = torch.optim.Adam(self.dcrec.parameters(), weight_decay=self.reg, lr=self.lr)
        # self.optimizer = torch.optim.Adam(self.dcrec.parameters(), lr=self.lr)


    def train_model(self):
        data_iter = TimeOrderPairwiseSampler(self.dataset.train_data, len_seqs=self.max_len,
                                             len_next=self.seq_T, pad=self.pad_idx,
                                             num_neg=self.seq_T,
                                             batch_size=self.batch_size,
                                             shuffle=True, drop_last=True)
        self.logger.info(self.evaluator.metrics_info())
        training_start_time = time.time()
        for epoch in range(self.epochs):
            # now_training = time.time()
            self.dcrec.train()
            for bat_users, bat_item_seqs, bat_pos_items, bat_neg_items in data_iter:
                bat_items_seq_len = []
                bat_item_seqs_2 = []
                for line in bat_item_seqs:
                    # print(line)
                    seq_len = 0
                    for i in line:
                        if i != self.num_items-1:
                            seq_len += 1
                        else:
                            line = np.delete(line, 0)
                            line = np.insert(line, self.max_len-1, self.num_items-1)
                    bat_items_seq_len.append(seq_len)
                    bat_item_seqs_2.append(line)
                bat_item_seqs_2 = np.array(bat_item_seqs_2)
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_item_seqs_2 = torch.from_numpy(bat_item_seqs_2).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_items_seq_len = torch.from_numpy(np.array(bat_items_seq_len)).long().to(self.device)

                loss1, loss2, loss3 = self.dcrec.calculate_loss(bat_users, bat_item_seqs_2, bat_pos_items, bat_items_seq_len)
                loss = loss1 + loss3 + loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # training_t = time.time() - now_training
            # print("training time：", training_t)
            # now_testing = time.time()
            result = self.evaluate_model()
            # testing_t = time.time() - now_testing
            # print("testing time：", testing_t)
            # self.logger.info("epoch %d:\t%s" % (epoch, result))
            self.logger.info("[iter %d:time:%f]" % (epoch, time.time() - training_start_time))



    def evaluate_model(self):
        self.dcrec.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        bat_seq = [self.user_pos_train[u] for u in users]
        # bat_seq = [self.user_truncated_seq[u] for u in users]
        # print(test_item)
        bat_seq_len = []
        for line in bat_seq:
            if len(line) > self.max_len:
                bat_seq_len.append(self.max_len)
            else:
                bat_seq_len.append(len(line))
        bat_seq = pad_sequences(bat_seq, value=self.num_items-1, max_len=self.max_len, padding='post', truncating='pre')
        # print(bat_seq)
        bat_seq = torch.from_numpy(np.asarray(bat_seq)).long().to(self.device)
        bat_seq_len = np.array(bat_seq_len)
        bat_seq_len = torch.from_numpy(bat_seq_len).long().to(self.device)
        all_ratings = self.dcrec.fast_predict(bat_seq, bat_seq_len)
        return all_ratings.cpu().detach().numpy()


