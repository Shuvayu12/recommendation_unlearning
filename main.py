import os
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from models.lightgcn import LightGCN, ndcg_at_k, recall_at_k, mi_bf, mi_ng
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class MovielensDataset(Dataset):
    def __init__(self, train_data, num_items, num_negatives=4):
        self.train_data = train_data
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.users = train_data['UserIdx'].values
        self.items = train_data['ItemIdx'].values
        self.user_item_set = set(zip(train_data['UserIdx'], train_data['ItemIdx']))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        neg_items = []
        for _ in range(self.num_negatives):
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if (user, neg_item) not in self.user_item_set:
                    break
            neg_items.append(neg_item)
        return user, pos_item, neg_items

def create_adj_matrix(train_data, num_users, num_items):
    user_items = csr_matrix((np.ones_like(train_data['UserIdx']), 
                           (train_data['UserIdx'], train_data['ItemIdx'])),
                           shape=(num_users, num_items))
    n_nodes = num_users + num_items
    adj_mat = csr_matrix((n_nodes, n_nodes))
    adj_mat[:num_users, num_users:] = user_items
    adj_mat[num_users:, :num_users] = user_items.T
    return adj_mat

def train_lightgcn():
    print("Loading data...")
    train = pd.read_csv('data/movielens_data/train.csv')
    test = pd.read_csv('data/movielens_data/test.csv')
    
    num_users = train['UserIdx'].max() + 1
    num_items = train['ItemIdx'].max() + 1
    adj_matrix = create_adj_matrix(train, num_users, num_items)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightGCN(
        n_users=num_users,
        n_items=num_items,
        embedding_dim=64,
        n_layers=3,
        adj_matrix=adj_matrix
    ).to(device)

    dataset = MovielensDataset(train, num_items)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting training...")
    model.train()
    for epoch in range(50):
        total_loss = 0
        for users, pos_items, neg_items in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
            users = users.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items[0].to(device)  # Take first negative sample
            
            loss = model.calculate_loss(users, pos_items, neg_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_users = torch.LongTensor(test['UserIdx'].values).to(device)
                test_items = test['ItemIdx'].values
                predictions = model.predict(test_users).cpu().numpy()
                k = 10  
                indices = np.argsort(-predictions, axis=1)[:, :k]
                r = np.zeros_like(predictions)
                for i, items in enumerate(indices):
                    r[i, items] = 1
                
                test_data = [[item] for item in test_items]
                ndcg = ndcg_at_k(test_data, r, k)
                recall = recall_at_k(test_data, r, k)

                binary_preds = (predictions > predictions.mean()).astype(float)
                binary_truth = np.zeros((len(test_users), num_items))
                for i, item in enumerate(test_items):
                    binary_truth[i, item] = 1
                
                mibf = mi_bf(binary_preds, binary_truth)
                ming = mi_ng(binary_preds, binary_truth)
                
                print(f'Metrics @ {k}:')
                print(f'NDCG: {ndcg:.4f}')
                print(f'Recall: {recall:.4f}')
                print(f'MI-BF: {mibf:.4f}')
                print(f'MI-NG: {ming:.4f}')
            model.train()

if __name__ == '__main__':
    train_lightgcn()
