#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from copy import deepcopy
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData
from torch.nn import Dropout
from torch.optim import AdamW, SGD
from torch_geometric.nn import HeteroConv, Linear, GATConv, GCNConv, HANConv, HGTConv, SAGEConv
import torch.nn.functional as F
from torch_geometric.loader import ImbalancedSampler, NeighborLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math
from collections import Counter
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import wandb


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
prefix = '/usr/local/src/BlockSci/Notebooks/Ransomware - Mario i Albert/Heterogeneous'

def get_parameters():
    print("Type of seed addresses:\n---------------------------------")
    print("1. Licit\n2. Illicit\n3. Licit and Illicit (50/50)")
    seed_options = {"1": "licit", "2": "illicit", "3": "licit and illicit"}
    seed = seed_options[input("Option: ")]

    print("\nDirection of the expansion:\n---------------------------------")
    print("1. Forward Backward\n2. All over")
    direction_options = {"1": "fw bw", "2": "all over"}
    direction = direction_options[input("Option: ")]

    print("\nApproach of the expansion:\n---------------------------------")
    print("1. Transaction-based\n2. Address-based")
    graph_options = {"1": " tx", "2": " addr"}
    graph = graph_options[input("Option: ")]

    if direction == "fw bw":
        print("\nTransaction addresses proportion of the expansion:\n---------------------------------")
        print("1. Whole\n2. Dedicated")
        address_options = {"1": " whole", "2": " dedicated"}
        address = address_options[input("Option: ")]

        if address == " whole" and graph == " addr":
            print("\nSide addresses direction of the expansion:\n---------------------------------")
            print("1. None\n2. Same\n3. Opposite")
            side_direction_options = {"1": " none", "2": " same", "3": " opposite"}
            side_direction = side_direction_options[input("Option: ")]
        else:
            side_direction = ""
    else:
        address = ""
        side_direction = ""

    exp_alg = f'{direction}{graph}{address}{side_direction}'

    print("\nLimit mode:\n---------------------------------")
    print("1. Random node\n2. Random hop\n3. None")
    limit_mode_options = {"1": "random node", "2": "random hop", "3": ""}
    limit_mode = limit_mode_options[input("Option: ")]

    if limit_mode != "":
        print("\nLimit value:\n---------------------------------")
        limit = int(input("Option: "))
    else:
        limit = "no"

    if exp_alg.startswith('fw bw'):
        print("\nNumber of forward hops:\n---------------------------------")
        for_hops = int(input("Option: "))

        print("\nNumber of backward hops:\n---------------------------------")
        back_hops = int(input("Option: "))
        hops = for_hops
    else:
        print("\nNumber of hops:\n---------------------------------")
        hops = int(input("Option: "))
        for_hops = hops
        back_hops = hops

    print("\nNumber of train samples:\n---------------------------------")
    train_samples = int(input("Option: "))

    print("\nNumber of validation samples:\n---------------------------------")
    val_samples = int(input("Option: "))

    print("\nNumber of test samples:\n---------------------------------")
    test_samples = int(input("Option: "))
    print()

    space = "" if limit_mode == "" else " "

    data_path = f'{prefix}/{seed}/{train_samples}-{val_samples}-{test_samples} {exp_alg} {hops} hops {limit}{space}{limit_mode} limit/'

    return space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, prefix, data_path, seed

space, exp_alg, limit_mode, limit, for_hops, back_hops, hops, train_samples, val_samples, test_samples, prefix, data_path, seed = get_parameters()
arch = input("Choose an architecture (RGCN, GAT, HAN, HGT): ").upper() # 'gat', 'rgcn', 'han', 'hgt'
use_wandb = input("\nUse wandb? (yes/no): ").strip().lower() == "yes"


# In[4]:


print("\nReading train transactions...")
train_txs_df   = pd.read_csv(data_path + 'train/tx_feats.csv')
print("Reading train addresses...")
train_addrs_df   = pd.read_csv(data_path + 'train/addr_feats.csv')
print("Reading train inputs...")
train_inputs_df   = pd.read_csv(data_path + 'train/input_feats.csv')
print("Reading train outputs...")
train_outputs_df   = pd.read_csv(data_path + 'train/output_feats.csv')

print("\nReading validation transactions...")
val_txs_df   = pd.read_csv(data_path + 'val/tx_feats.csv')
print("Reading validation addresses...")
val_addrs_df   = pd.read_csv(data_path + 'val/addr_feats.csv')
print("Reading validation inputs...")
val_inputs_df   = pd.read_csv(data_path + 'val/input_feats.csv')
print("Reading validation outputs...")
val_outputs_df   = pd.read_csv(data_path + 'val/output_feats.csv')

print("\nReading test transactions...")
test_txs_df   = pd.read_csv(data_path + 'test/tx_feats.csv')
print("Reading test addresses...")
test_addrs_df   = pd.read_csv(data_path + 'test/addr_feats.csv')
print("Reading test inputs...")
test_inputs_df   = pd.read_csv(data_path + 'test/input_feats.csv')
print("Reading test outputs...")
test_outputs_df   = pd.read_csv(data_path + 'test/output_feats.csv')


# In[ ]:


"""def build_graph(txs_df, addrs_df, inputs_df, outputs_df, tx_mapping, addr_mapping):
    data = HeteroData()

    inputs_df['tx_idx'] = inputs_df['tx_hash'].map(tx_mapping).astype(int)
    inputs_df['addr_idx'] = inputs_df['addr_str'].map(addr_mapping).astype(int)

    inputs_df['spent_tx_idx'] = inputs_df['spent_tx_hash'].map(tx_mapping)
    missing_spent_tx = inputs_df['spent_tx_idx'].isna()
    inputs_df.loc[missing_spent_tx, 'spent_output_index'] = -1

    inputs_df['spent_tx_idx'] = inputs_df['spent_tx_idx'].fillna(-1).astype(int)
    inputs_df['spent_output_index'] = inputs_df['spent_output_index'].astype(int)

    outputs_df['tx_idx'] = outputs_df['tx_hash'].map(tx_mapping).astype(int)
    outputs_df['addr_idx'] = outputs_df['addr_str'].map(addr_mapping).astype(int)

    # Addresses
    data['tx'].x = torch.tensor(txs_df[['block_height', 'fee', 'locktime', 'total_size', 'version']].values, dtype=torch.float)
    bool_features = torch.tensor(txs_df[['is_coinbase']].astype(float).values, dtype=torch.float)
    data['tx'].x = torch.cat([data['tx'].x, bool_features], dim=1)
    data['tx'].id = txs_df.index.values

    # Transactions
    addrs_df['full_type'], _ = pd.factorize(addrs_df['full_type'])
    data['addr'].x = torch.tensor(addrs_df[['full_type']].values, dtype=torch.float)
    data['addr'].id = addrs_df.index.values
    data['addr'].y = torch.tensor(addrs_df['class'].values, dtype=torch.long)

    # Inputs
    data['addr', 'input', 'tx'].edge_index = torch.tensor(inputs_df[['addr_idx', 'tx_idx']].values.T, dtype=torch.long)
    input_attrs = torch.tensor(inputs_df[['age', 'sequence_num', 'value', 'spent_tx_idx', 'spent_output_index']].values, dtype=torch.float)
    data['addr', 'input', 'tx'].edge_attr = input_attrs

    # Outputs
    data['tx', 'output', 'addr'].edge_index = torch.tensor(outputs_df[['tx_idx', 'addr_idx']].values.T, dtype=torch.long)
    output_attrs = torch.tensor(outputs_df[['index', 'value']].values, dtype=torch.float)
    bool_spent = torch.tensor(outputs_df[['is_spent']].astype(float).values, dtype=torch.float)
    data['tx', 'output', 'addr'].edge_attr = torch.cat([output_attrs, bool_spent], dim=1)

    return data

train_tx_mapping = pd.Series(train_txs_df.index.values, index=train_txs_df['hash']).to_dict()
train_addr_mapping = pd.Series(train_addrs_df.index.values, index=train_addrs_df['addr_str']).to_dict()

val_tx_mapping    = pd.Series(val_txs_df.index.values, index=val_txs_df['hash']).to_dict()
val_addr_mapping  = pd.Series(val_addrs_df.index.values, index=val_addrs_df['addr_str']).to_dict()

test_tx_mapping = pd.Series(test_txs_df.index.values, index=test_txs_df['hash']).to_dict()
test_addr_mapping = pd.Series(test_addrs_df.index.values, index=test_addrs_df['addr_str']).to_dict()

for split, tx_map, addr_map in [
    ('train', train_tx_mapping,  train_addr_mapping),
    ('val',   val_tx_mapping,    val_addr_mapping),
    ('test',  test_tx_mapping,   test_addr_mapping),
]:
    with open(data_path + f"{split}/tx_mapping.json",  "w") as f:
        json.dump(tx_map, f)
    with open(data_path + f"{split}/addr_mapping.json","w") as f:
        json.dump(addr_map, f)

train_data = build_graph(
    train_txs_df, train_addrs_df, train_inputs_df, train_outputs_df,
    train_tx_mapping, train_addr_mapping
)
val_data = build_graph(
    val_txs_df, val_addrs_df, val_inputs_df, val_outputs_df,
    val_tx_mapping, val_addr_mapping
)
test_data = build_graph(
    test_txs_df, test_addrs_df, test_inputs_df, test_outputs_df,
    test_tx_mapping, test_addr_mapping
)"""


# In[5]:


"""torch.save(train_data, data_path + 'train/graph.pth')
torch.save(val_data,   data_path + 'val/graph.pth')
torch.save(test_data,  data_path + 'test/graph.pth')"""
train_data = torch.load(data_path + 'train/graph.pth')
val_data = torch.load(data_path + 'val/graph.pth')
test_data = torch.load(data_path + 'test/graph.pth')


# In[6]:


def remove_nan_values(data):
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            data[node_type].x = data[node_type].x[~torch.isnan(data[node_type].x).any(dim=1)]
        if hasattr(data[node_type], 'y'):
            data[node_type].y = data[node_type].y[~torch.isnan(data[node_type].y)]
    return data

clean_train_data = remove_nan_values(train_data)
if train_data != clean_train_data:
    train_data = clean_train_data
else:
    print("There is no null element in training data")

clean_val_data = remove_nan_values(val_data)
if val_data is not clean_val_data:
    val_data = clean_val_data
else:
    print("There is no null element in validation data")

clean_test_data = remove_nan_values(test_data)
if test_data != clean_test_data:
    test_data = clean_test_data
else:
    print("There is no null element in testing data")


# In[7]:


def clip_outliers(data, z_threshold=3):
    for node_type in data.node_types:
        x = data[node_type].x
        for col in range(x.shape[1]):
            if x[:, col].dtype == torch.float:
                values = x[:, col].numpy()
                mean, std = values.mean(), values.std()
                upper_limit = mean + z_threshold * std
                lower_limit = mean - z_threshold * std
                values_clipped = np.clip(values, lower_limit, upper_limit)
                x[:, col] = torch.tensor(values_clipped, dtype=torch.float)
        data[node_type].x = x

    for edge_type in data.edge_types:
        edge_attr = data[edge_type].edge_attr
        for col in range(edge_attr.shape[1]):
            if edge_attr[:, col].dtype == torch.float:
                values = edge_attr[:, col].numpy()
                mean, std = values.mean(), values.std()
                upper_limit = mean + z_threshold * std
                lower_limit = mean - z_threshold * std
                values_clipped = np.clip(values, lower_limit, upper_limit)
                edge_attr[:, col] = torch.tensor(values_clipped, dtype=torch.float)
        data[edge_type].edge_attr = edge_attr
    return data

train_data = clip_outliers(train_data)
val_data = clip_outliers(val_data)
test_data = clip_outliers(test_data)


# In[8]:


def norm_attr(train_data, val_data, test_data, element_type, element_name, attribute_indices):
    for idx in attribute_indices:
        scaler = MinMaxScaler()

        if element_type == 'node':
            train_data[element_name].x[:, idx] = torch.tensor(
                scaler.fit_transform(train_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            val_data[element_name].x[:, idx] = torch.tensor(
                scaler.fit_transform(val_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            test_data[element_name].x[:, idx] = torch.tensor(
                scaler.transform(test_data[element_name].x[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

        elif element_type == 'edge':
            train_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.fit_transform(train_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            val_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.fit_transform(val_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

            test_data[element_name].edge_attr[:, idx] = torch.tensor(
                scaler.transform(test_data[element_name].edge_attr[:, idx].unsqueeze(1).numpy()).flatten(),
                dtype=torch.float
            )

norm_attr(train_data, val_data, test_data, element_type='node', element_name='tx', attribute_indices=[0, 1, 2, 3])  # block_height, fee, locktime, total_size
norm_attr(train_data, val_data, test_data, element_type='edge', element_name=('addr', 'input', 'tx'), attribute_indices=[0,  2, 3])  # block, index, sequence_num, value
norm_attr(train_data, val_data, test_data, element_type='edge', element_name=('tx', 'output', 'addr'), attribute_indices=[0, 1])  # block, index, value


# In[9]:


def _add_dropout(x_dict, dropout):
    return {k: dropout(v) for k, v in x_dict.items()}

class GATHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_dims, dropout_p=0.2):
        super().__init__()
        self.dropout = Dropout(p=dropout_p)
        self.conv1 = HeteroConv({
            ('addr','input','tx'): GATConv((-1,-1), hidden_channels, heads=4,
                                          edge_dim=edge_dims['input'], add_self_loops=False),
            ('tx','output','addr'): GATConv((-1,-1), hidden_channels, heads=4,
                                           edge_dim=edge_dims['output'], add_self_loops=False)
        }, aggr='mean')
        self.conv2 = HeteroConv({
            ('addr','input','tx'): GATConv((-1,-1), hidden_channels, heads=4,
                                          edge_dim=edge_dims['input'], add_self_loops=False),
            ('tx','output','addr'): GATConv((-1,-1), hidden_channels, heads=4,
                                           edge_dim=edge_dims['output'], add_self_loops=False)
        }, aggr='mean')
        self.conv3 = HeteroConv({
            ('addr','input','tx'): GATConv((-1,-1), hidden_channels, heads=4,
                                          edge_dim=edge_dims['input'], add_self_loops=False),
            ('tx','output','addr'): GATConv((-1,-1), hidden_channels, heads=4,
                                           edge_dim=edge_dims['output'], add_self_loops=False)
        }, aggr='mean')
        self.lin = Linear(hidden_channels*4, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for conv in (self.conv1, self.conv2, self.conv3):
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = _add_dropout(x_dict, self.dropout)
        return self.lin(x_dict['addr'])

class RGCNHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_p=0.2):
        super().__init__()
        self.dropout = Dropout(p=dropout_p)
        # Cada capa amb instàncies noves de SAGEConv per evitar reuse de pesos
        self.conv1 = HeteroConv({
            ('addr','input','tx'): SAGEConv((-1,-1), hidden_channels),
            ('tx','output','addr'): SAGEConv((-1,-1), hidden_channels)
        }, aggr='mean')
        self.conv2 = HeteroConv({
            ('addr','input','tx'): SAGEConv((-1,-1), hidden_channels),
            ('tx','output','addr'): SAGEConv((-1,-1), hidden_channels)
        }, aggr='mean')
        self.conv3 = HeteroConv({
            ('addr','input','tx'): SAGEConv((-1,-1), hidden_channels),
            ('tx','output','addr'): SAGEConv((-1,-1), hidden_channels)
        }, aggr='mean')
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for conv in (self.conv1, self.conv2, self.conv3):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = _add_dropout(x_dict, self.dropout)
        return self.lin(x_dict['addr'])

class HANHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, heads=4, dropout_p=0.2):
        super().__init__()
        self.dropout = Dropout(p=dropout_p)
        self.conv1 = HANConv(in_channels=-1, out_channels=hidden_channels,
                             metadata=metadata, heads=heads)
        self.conv2 = HANConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             metadata=metadata, heads=heads)
        self.conv3 = HANConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             metadata=metadata, heads=heads)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for conv in (self.conv1, self.conv2, self.conv3):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = _add_dropout(x_dict, self.dropout)
        return self.lin(x_dict['addr'])

class HGTHeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, heads=4, dropout_p=0.2):
        super().__init__()
        self.dropout = Dropout(p=dropout_p)
        self.conv1 = HGTConv(in_channels=-1, out_channels=hidden_channels,
                             metadata=metadata, heads=heads)
        self.conv2 = HGTConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             metadata=metadata, heads=heads)
        self.conv3 = HGTConv(in_channels=hidden_channels, out_channels=hidden_channels,
                             metadata=metadata, heads=heads)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for conv in (self.conv1, self.conv2, self.conv3):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = _add_dropout(x_dict, self.dropout)
        return self.lin(x_dict['addr'])


# In[21]:


class_names=['Licit','Illicit']
# Hiperparámetros generales
hidden_channels = 32
out_channels = 2

# Dimensiones de atributos de arista
edge_dims = {
    'input': train_data[('addr','input','tx')].edge_attr.size(1),
    'output': train_data[('tx','output','addr')].edge_attr.size(1)
}

# Metadata del grafo (tipos de nodo/edge)
metadata = train_data.metadata()

# Selección de modelo según arquitectura
if arch == 'GAT':
    model = GATHeteroGNN(hidden_channels, out_channels, edge_dims).to(device)
elif arch == 'RGCN':
    model = RGCNHeteroGNN(hidden_channels, out_channels).to(device)
elif arch == 'HAN':
    model = HANHeteroGNN(hidden_channels, out_channels, metadata).to(device)
elif arch == 'HGT':
    model = HGTHeteroGNN(hidden_channels, out_channels, metadata).to(device)
else:
    raise ValueError(f"Arquitectura desconocida: {arch}")

# Cálculo de porcentajes de clases en entrenamiento
licit_percentage   = (train_data['addr'].y == 0).sum().item() / train_data['addr'].y.size(0)
illicit_percentage = (train_data['addr'].y == 1).sum().item() / train_data['addr'].y.size(0)

print(f"\nTrain data licit percentage: {licit_percentage:.4f}")
print(f"Train data illicit percentage: {illicit_percentage:.4f}\n")

# Etiquetas en device
y_train = train_data['addr'].y.to(device)
y_val   = val_data['addr'].y.to(device)
y_test  = test_data['addr'].y.to(device)

# Máscaras o índices
if hasattr(train_data['addr'], 'train_mask'):
    idx_train = train_data['addr'].train_mask.nonzero(as_tuple=True)[0]
else:
    idx_train = torch.arange(y_train.size(0), device=device)

if hasattr(train_data['addr'], 'val_mask'):
    idx_val = train_data['addr'].val_mask.nonzero(as_tuple=True)[0]
else:
    idx_val = torch.arange(y_val.size(0), device=device)

# Para test usamos todos los nodos (grafo separado)
idx_test = torch.arange(y_test.size(0), device=device)

# Cálculo de pesos de clases para CrossEntropyLoss
N_pos = int((y_train == 1).sum())
N_neg = int((y_train == 0).sum())
weights = torch.tensor([1/N_neg, 1/N_pos], dtype=torch.float32)

criterion = torch.nn.CrossEntropyLoss(weight=weights)
print(f"Class-weights  w0 = {weights[0]:.4e}  w1 = {weights[1]:.4e}")

# Optimizadores y scheduler
lr_adam = 6e-3
lr_sgd  = 1e-4

opt1 = AdamW(model.parameters(), lr=lr_adam, weight_decay=3e-4)
opt2 = SGD(model.parameters(), lr=lr_sgd, momentum=0.9)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt2, mode='max', factor=0.5, patience=10, verbose=True
)

# Épocas
epochs = 100


if use_wandb:
    run = wandb.init(
        name=f"{arch} {seed} {train_samples}-{val_samples}-{test_samples} {exp_alg} {hops} hops {limit} {limit_mode}",
        project="bitcoin-ransomware-patterns-detection",
        config={
            "lr_adam": lr_adam,
            "lr_sgd": lr_sgd,
            "epochs": epochs,
            "architecture": f"3layers-{arch}-Hetero",
            "hidden_channels": hidden_channels,
            "out_channels": out_channels,
            "optimizer": "Adam+SDG",
            "loss_function": "CrossEntropyLoss",
            "dataset": {
                "seed": seed,
                "train_samples": train_samples,
                "val_samples": val_samples,
                "test_samples": test_samples,
                "expansion_algorithm": exp_alg,
                "hops": hops,
                "limit_mode": limit_mode,
                "limit_value": limit,
                "forward_hops": for_hops if exp_alg.startswith("fw bw") else None,
                "backward_hops": back_hops if exp_alg.startswith("fw bw") else None
            }
        }
    )


# In[22]:


best_f1, wait, f1_val, patience, warmup_epochs = 0.0, 0, 0.0000, 20, int(0.3*epochs)

# Mover datos a GPU si está disponible
train_data = train_data.to(device)
val_data   = val_data.to(device)

train_losses, val_losses, test_losses = [], [], []

pbar = tqdm(range(1, epochs + 1), desc="Epoch")

for epoch in pbar:
    model.train()
    # Elegir optimizador (warmup con Adam, luego SGD)
    opt = opt1 if epoch <= warmup_epochs else opt2
    opt.zero_grad()

    # Forward + backward en train_data
    out_train = model(train_data)
    loss = criterion(out_train[idx_train], y_train[idx_train])
    loss.backward()
    opt.step()
    train_losses.append(loss.item())

    preds_train = out_train.argmax(dim=1).cpu()
    true_train  = y_train.cpu()
    acc_train   = accuracy_score(true_train[idx_train], preds_train[idx_train])
    prec_train  = precision_score(true_train[idx_train], preds_train[idx_train], zero_division=0)
    rec_train   = recall_score(true_train[idx_train], preds_train[idx_train], zero_division=0)
    f1_train    = f1_score(true_train[idx_train], preds_train[idx_train], zero_division=0)
    cm_train    = confusion_matrix(true_train, preds_train)

    pbar.set_postfix(loss=f"{loss.item():.4f}", valf1=f"{f1_val:.4}")

    # Validación
    model.eval()
    with torch.no_grad():
        out_val = model(val_data)
        val_loss = criterion(out_val[idx_val], y_val[idx_val])
        preds_val = out_val.argmax(dim=1).cpu()
        true_val  = y_val.cpu()
        acc_val   = accuracy_score(true_val, preds_val)
        prec_val  = precision_score(true_val, preds_val, zero_division=0)
        rec_val   = recall_score(true_val, preds_val, zero_division=0)
        f1_val    = f1_score(true_val, preds_val, zero_division=0)
        val_losses.append(val_loss.item())

    # Test (para seguimiento)
    with torch.no_grad():
        out_test = model(test_data)
        test_loss= criterion(out_test[idx_test], y_test.to(device)[idx_test])
        preds_test = out_test.argmax(dim=1).cpu()
        true_test  = y_test.cpu()
        acc_test   = accuracy_score(true_test, preds_test)
        prec_test  = precision_score(true_test, preds_test, zero_division=0)
        rec_test   = recall_score(true_test, preds_test, zero_division=0)
        f1_test    = f1_score(true_test, preds_test, zero_division=0)
        cm_test    = confusion_matrix(true_test, preds_test)
        test_losses.append(test_loss.item())
        
    if epoch % 2 == 0:
        cm_train_df = pd.DataFrame(
            cm_train,
            index=[f"True: {c}" for c in class_names],
            columns=[f"Pred: {c}" for c in class_names]
            )
        cm_test_df = pd.DataFrame(
            cm_test,
            index=[f"True: {c}" for c in class_names],
            columns=[f"Pred: {c}" for c in class_names]
            )

        print("\n------------------------------------------------------\n")
        print(f"Epoch: {epoch}")
        print(f"  Train Loss: {loss.item():.4f}    Test Loss: {test_loss.item():.4f}\n")

        print("  (Train) Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(prec_train, rec_train, f1_train))
        print("  Train Confusion Matrix:")
        print(cm_train_df, "\n")   # Esto ya muestra filas y columnas etiquetadas

        print("  (Test)  Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(prec_test, rec_test, f1_test))
        print("  Test Confusion Matrix:")
        print(cm_test_df)
        
    # Registro en wandb si se habilitó
    if use_wandb:
        wandb.log({
            'epoch': epoch,
            'train/loss': loss.item(),
            'train/accuracy': acc_train,
            'train/precision': prec_train,
            'train/recall': rec_train,
            'train/f1': f1_train,
            'train/confusion_matrix': wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_train[idx_train].numpy(),
                preds=preds_train[idx_train].numpy(),
                class_names=class_names
            ),
            'test/accuracy': acc_test,
            'test/loss': test_loss.item(),
            'test/precision': prec_test,
            'test/recall': rec_test,
            'test/f1': f1_test,
            'test/confusion_matrix': wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_test.numpy(),
                preds=preds_test.numpy(),
                class_names=class_names
            )
        }, step=epoch)

        wandb.log({
            'val/accuracy': acc_val,
            'val/loss': val_loss.item(),
            'val/precision': prec_val,
            'val/recall': rec_val,
            'val/f1': f1_val,
            'val/confusion_matrix': wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_val.numpy(),
                preds=preds_val.numpy(),
                class_names=class_names
            )
        }, step=epoch)

    # Scheduler (solo para SGD)
    if epoch > warmup_epochs:
        scheduler.step(f1_val)

    # Early stopping
    if f1_val > best_f1:
        best_f1, wait = f1_val, 0
        torch.save(model.state_dict(), os.path.join(data_path, 'best_model.pth'))
    else:
        wait += 1
        if wait >= patience:
            print(f"\nEarly stopping en epoch {epoch}")
            break

# Guardar modelo final
torch.save(model.state_dict(), os.path.join(data_path, f'{arch}model.pth'))

if use_wandb:
    artifact_name = f"{arch}"
    artifact = wandb.Artifact(name=artifact_name, type="model")
    artifact.add_file(os.path.join(data_path, f'{arch}model.pth'))
    run.log_artifact(artifact)
    run.finish()


# Cell 11: Función de evaluación e impresión de métricas finales

def evaluate_model(model, data, split_name, use_wandb=False):
    """
    Evalúa `model` sobre el HeteroData `data`, imprime métricas
    y muestra la matriz de confusión para el split indicado.
    """
    model.eval()
    with torch.no_grad():
        out = model(data)
        y_true = data['addr'].y
        y_pred = out.argmax(dim=1)

    acc   = accuracy_score(y_true.cpu(), y_pred.cpu())
    rec   = recall_score(  y_true.cpu(), y_pred.cpu(), pos_label=1)
    prec  = precision_score(y_true.cpu(), y_pred.cpu(), pos_label=1)
    f1    = f1_score(      y_true.cpu(), y_pred.cpu(), pos_label=1)
    cm    = confusion_matrix(y_true.cpu(), y_pred.cpu())

    print(f'\n— {split_name} —')
    print(f'  Accuracy : {acc:.4f}')
    print(f'  Recall   : {rec:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  F1 score : {f1:.4f}')

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix ({split_name})')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return out

# Evaluar el modelo en cada split
out_train = evaluate_model(model, train_data,      "Train",      use_wandb)
out_val   = evaluate_model(model, val_data,        "Validation", use_wandb)
out_test  = evaluate_model(model, test_data,       "Test",       use_wandb)
