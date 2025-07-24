import torch
import torch.nn as nn
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_layers, adj_matrix):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.norm_adj_matrix = self._normalize_adj(adj_matrix)
        
    def _normalize_adj(self, adj_matrix):
        adj_matrix = adj_matrix.tocoo()
        indices = np.vstack((adj_matrix.row, adj_matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(adj_matrix.data)
        shape = adj_matrix.shape
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, users, items):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj_matrix, all_emb)
            embs.append(all_emb)
        light_out = torch.mean(torch.stack(embs, dim=0), dim=0)
        users_emb = light_out[:self.n_users]
        items_emb = light_out[self.n_users:]
        user_vecs = users_emb[users]
        item_vecs = items_emb[items]
        return torch.sum(user_vecs * item_vecs, dim=1)

    def calculate_loss(self, users, pos_items, neg_items):
        pos_scores = self.forward(users, pos_items)
        neg_scores = self.forward(users, neg_items)
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        l2_reg = torch.norm(self.user_embedding.weight) ** 2 + torch.norm(self.item_embedding.weight) ** 2
        return loss + 1e-4 * l2_reg

    def predict(self, users):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj_matrix, all_emb)
            embs.append(all_emb)
        light_out = torch.mean(torch.stack(embs, dim=0), dim=0)
        users_emb = light_out[:self.n_users]
        items_emb = light_out[self.n_users:]
        user_vecs = users_emb[users]
        return torch.matmul(user_vecs, items_emb.t())
    
def ndcg_at_k(test_data, r, k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def recall_at_k(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    return recall

def mi_bf(predictions, labels):
    eps = 1e-10
    p1 = np.mean(predictions)
    p2 = np.mean(labels)
    p12 = np.mean(predictions * labels)
    mi = p12 * np.log((p12 + eps) / (p1 * p2 + eps)) + \
         (1 - p12) * np.log(((1 - p12) + eps) / ((1 - p1) * (1 - p2) + eps))
    return mi

def mi_ng(predictions, labels):
    eps = 1e-10
    p1 = np.mean(predictions)
    p2 = np.mean(labels)
    p12 = np.mean(predictions * labels)
    h1 = -p1 * np.log(p1 + eps) - (1 - p1) * np.log(1 - p1 + eps)
    h2 = -p2 * np.log(p2 + eps) - (1 - p2) * np.log(1 - p2 + eps)
    mi = p12 * np.log((p12 + eps) / (p1 * p2 + eps)) + \
         (1 - p12) * np.log(((1 - p12) + eps) / ((1 - p1) * (1 - p2) + eps))
    nmi = mi / (np.sqrt(h1 * h2) + eps)
    return nmi