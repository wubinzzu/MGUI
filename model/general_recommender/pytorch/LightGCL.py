"""
LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation
"""

import torch
import torch.nn as nn
import torch.utils
from model.base import AbstractRecommender
import numpy as np
from data import PairwiseSampler
from util.pytorch import sp_mat_to_sp_tensor


def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)


class _LightGCL(nn.Module):
    def __init__(self, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, device):
        super(_LightGCL,self).__init__()
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u,d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i,d)))

        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l+1)
        self.E_i_list = [None] * (l+1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l+1)
        self.Z_i_list = [None] * (l+1)
        self.G_u_list = [None] * (l+1)
        self.G_i_list = [None] * (l+1)
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)

        self.E_u = None
        self.E_i = None

        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        self.device = device

    def forward(self, uids, iids, pos, neg, test=False):
        if test==True:  # testing phase
            preds = self.E_u[uids] @ self.E_i.T
            mask = self.train_csr[uids.cpu().numpy()].toarray()
            mask = torch.Tensor(mask).cuda(torch.device(self.device))
            preds = preds * (1-mask) - 1e8 * mask
            # predictions = preds.argsort(descending=True)
            return preds  # predictions
        else:  # training phase
            for layer in range(1,self.l+1):
                # GNN propagation
                self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout), self.E_i_list[layer-1]))
                self.Z_i_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm,self.dropout).transpose(0,1), self.E_u_list[layer-1]))

                # svd_adj propagation
                vt_ei = self.vt @ self.E_i_list[layer-1]
                self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
                ut_eu = self.ut @ self.E_u_list[layer-1]
                self.G_i_list[layer] = (self.v_mul_s @ ut_eu)

                # aggregate
                self.E_u_list[layer] = self.Z_u_list[layer]
                self.E_i_list[layer] = self.Z_i_list[layer]

            self.G_u = sum(self.G_u_list)
            self.G_i = sum(self.G_i_list)

            # aggregate across layers
            self.E_u = sum(self.E_u_list)
            self.E_i = sum(self.E_i_list)

            # cl loss
            G_u_norm = self.G_u
            E_u_norm = self.E_u
            G_i_norm = self.G_i
            E_i_norm = self.E_i
            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / self.temp,-5.0,5.0)).mean()
            loss_s = -pos_score + neg_score

            # bpr loss
            u_emb = self.E_u[uids]
            pos_emb = self.E_i[pos]
            neg_emb = self.E_i[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_r = -(pos_scores - neg_scores).sigmoid().log().mean()

            # reg loss
            loss_reg = 0
            for param in self.parameters():
                loss_reg += param.norm(2).square()
            loss_reg *= self.lambda_2

            # total loss
            loss = loss_r + self.lambda_1 * loss_s + loss_reg
            #print('loss',loss.item(),'loss_r',loss_r.item(),'loss_s',loss_s.item())
            return loss, loss_r, self.lambda_1 * loss_s


class LightGCL(AbstractRecommender):
    def __init__(self, config):
        super(LightGCL, self).__init__(config)
        self.lr = config['lr']
        self.d = config['d']
        self.l = config['gnn_layer']
        self.temp = config['temp']
        self.batch_size = config['batch_size']
        self.epoch_no = config['epoch']
        self.max_samp = 40
        self.lambda_1 = config['lambda1']
        self.lambda_2 = config['lambda2']
        self.dropout = config['dropout']
        self.decay = config['decay']
        self.svd_q = config['q']

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        train = self.dataset.train_data.to_csr_matrix().tocoo()
        train.data[:] = 1.0
        train_csr = (train != 0).astype(np.float32)

        # normalizing the adj matrix
        rowD = np.array(train.sum(1)).squeeze()
        colD = np.array(train.sum(0)).squeeze()
        for i in range(len(train.data)):
            train.data[i] = train.data[i] / pow(rowD[train.row[i]] * colD[train.col[i]], 0.5)
        # train = normalize_adj_matrix(train, norm_method="symmetric")
        adj_norm = sp_mat_to_sp_tensor(train)
        adj_norm = adj_norm.coalesce().cuda(self.device)
        print('Adj matrix normalized.')

        # perform svd reconstruction
        adj = sp_mat_to_sp_tensor(train).coalesce().cuda(self.device)
        print('Performing SVD...')
        svd_u, s, svd_v = torch.svd_lowrank(adj, q=self.svd_q)
        u_mul_s = svd_u @ (torch.diag(s))
        v_mul_s = svd_v @ (torch.diag(s))
        del s
        print('SVD done.')

        self.model = _LightGCL(adj_norm.shape[0], adj_norm.shape[1], self.d, u_mul_s, v_mul_s, svd_u.T, svd_v.T,
                               train_csr, adj_norm, self.l, self.temp, self.lambda_1, self.lambda_2,
                               self.dropout, self.device)
        self.model.cuda(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=0, lr=self.lr)

    def train_model(self):
        data_iter = PairwiseSampler(self.dataset.train_data, num_neg=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epoch_no):
            for uids, pos, neg in data_iter:
                uids = torch.from_numpy(uids).long().to(self.device)
                pos = torch.from_numpy(pos).long().to(self.device)
                neg = torch.from_numpy(neg).long().to(self.device)
                iids = torch.cat([pos, neg], dim=0)

                # feed
                self.optimizer.zero_grad()
                loss, loss_r, loss_s = self.model(uids, iids, pos, neg)
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()
            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def evaluate_model(self):
        self.model.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = np.array(users)
        test_uids_input = torch.LongTensor(users).cuda(self.device)
        predictions = self.model(test_uids_input, None, None, None, test=True)
        return predictions.cpu().detach().numpy()
