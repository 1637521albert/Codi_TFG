import os
import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from torch_geometric.data import HeteroData
from tqdm import tqdm
import itertools
from collections import Counter, defaultdict
BASE_DIR = Path('/usr/local/src/BlockSci/Notebooks/Ransomware - Mario i Albert/TFGA/')
DATA_DIR = BASE_DIR / 'illicit'
RADIUS = 6 #en el meu cas o 4 o 6

def choose_dataset():
    """
    Muestra un menú con los 4 datasets disponibles y devuelve
    el subdirectorio elegido (solo el nombre, sin la ruta DATA_DIR).
    """
    options = [
        "3000-1000-1000 all over tx 3 hops 3 random node limit/",
        "3000-1000-1000 fw bw addr dedicated 3 hops 100000 random hop limit/",
        "3000-1000-1000 fw bw addr whole same 3 hops 20000 random hop limit/",
        "3000-1000-1000 fw bw tx whole 3 hops 10 random node limit/"
    ]

    print("\nElige el dataset a usar:\n---------------------------------")
    for i, name in enumerate(options, start=1):
        print(f"  {i}. {name}")
    print()

    while True:
        choice = input("Opción (1–4): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(options):
                selected = options[idx - 1]
                break
        print("Entrada no válida. Por favor, elige un número entre 1 y 4.")

    print(f"\nDataset seleccionado: {selected}\n")
    return selected

dataset_name = choose_dataset()
DATA_DIR = BASE_DIR / 'illicit' / dataset_name / "train"
EGO_DIR   = BASE_DIR / 'ego-nets' / dataset_name
RES_DIR = DATA_DIR / "res_pipeline"


# In[25]:


graph_path = DATA_DIR / 'graph.pth'
full_graph = torch.load(graph_path)

addr_ids    = np.array(full_graph['addr'].id)  # bitcoin addresses
addr_labels = np.array(full_graph['addr'].y)   # 0=licit,1=il·licit

with open(DATA_DIR / "addr_mapping.json") as f:
    addr2idx = json.load(f)  # BTC address -> idx intern
idx2addr = {v: k for k, v in addr2idx.items()}

with open(DATA_DIR / "tx_mapping.json") as f:
    tx2idx = json.load(f)    # tx-hash -> idx intern
idx2tx = { idx: tx for tx, idx in tx2idx.items() }

y_tensor = full_graph['addr'].y  # shape=(num_addr,)

# 3) Construeix addr_classes:
addr_classes = {
    idx2addr[idx]: int(y_tensor[idx].item())
    for idx in range(full_graph['addr'].num_nodes)
}
print('Loaded data')


# In[26]:


def inspect_heterodata(data: HeteroData):
    print("=== HeteroData Structure ===\n")

    # 1) Nodos y sus atributos
    print("→ Node types and feature shapes:")
    for node_type in data.node_types:
        storage = data[node_type]
        print(f" • Node type '{node_type}' (num_nodes={storage.num_nodes}):")
        for key in storage.keys():               # OJO: ahora llamamos a keys()
            tensor = storage[key]
            print(f"     – {key}: shape {tuple(tensor.shape)}")
    print()

    # 2) Aristas y sus atributos
    print("→ Edge types and attributes:")
    for edge_type in data.edge_types:
        storage = data[edge_type]
        print(f" • Edge type {edge_type}:")
        for key in storage.keys():
            tensor = storage[key]
            print(f"     – {key}: shape {tuple(tensor.shape)}")
    print()

    # 3) Resumen rápido
    total_nodes = sum(data[nt].num_nodes for nt in data.node_types)
    total_edges = sum(data[et].edge_index.size(1) for et in data.edge_types)
    print(f"Total node types: {len(data.node_types)} → {data.node_types}")
    print(f"Total edge types: {len(data.edge_types)} → {data.edge_types}")
    print(f"Total nodes across all types: {total_nodes}")
    print(f"Total edges across all types: {total_edges}")


# Crida la funció
inspect_heterodata(full_graph)


# In[4]:


def summarize_and_degree(data):

    print("="*72, "\nRESUMEN DEL GRAFO (HeteroData)\n", "="*72, sep="")
    # -------- Nodos ----------
    for ntype in data.node_types:
        n_sto = data[ntype]
        xshape = tuple(n_sto.x.shape) if 'x' in n_sto else None
        others = [k for k in n_sto.keys() if k not in ('x',)]
        print(f"[{ntype}]  N={n_sto.num_nodes:,}   x.shape={xshape}   otros={others}")

    # -------- Aristas ----------
    for et in data.edge_types:
        e_sto = data[et]
        src, rel, dst = et
        n_e = e_sto.edge_index.shape[1]
        ashape = tuple(e_sto.edge_attr.shape) if 'edge_attr' in e_sto else None
        print(f"({src}) -[{rel}]-> ({dst})   E={n_e:,}   edge_attr.shape={ashape}")

    # -------- In/Out degree por dirección ----------
    #   addr -> tx   (inputs)   → out-degree de addr
    #   tx   -> addr (outputs)  → in-degree  de addr
    global in_deg, out_deg # Make in_deg and out_deg globally accessible

    # Calculate out-degree for each address:
    # This counts how many times each address appears as a source in 'addr' -> 'tx' edges.
    out_deg = np.bincount(
        data['addr', 'input', 'tx'].edge_index[0].numpy(),
        minlength=data['addr'].num_nodes
    )

    # Calculate in-degree for each address:
    # This counts how many times each address appears as a destination in 'tx' -> 'addr' edges.
    # The variable 'in_deg' will be a NumPy array where in_deg[i] is the in-degree of address with index i.
    in_deg  = np.bincount(
        data['tx', 'output', 'addr'].edge_index[1].numpy(),
        minlength=data['addr'].num_nodes
    )

    # Create a DataFrame summarizing degrees for each address index.
    deg_df = pd.DataFrame({
        'in_degree':  in_deg,
        'out_degree': out_deg
    })

    return data, deg_df



hetero_graph, degree_df = summarize_and_degree(full_graph)

# degree_df tiene in/out-degree de cada addr
print("\nPrimeras 5 direcciones con sus grados:")


# In[27]:


def hetero_to_nx(data, addr_map=None, addr_classes=None, tx_map=None):
    G = nx.DiGraph()

    # 1) Preparar addr_map si no ve donat
    if addr_map is None:
        addr_map = { str(i): data['addr'].id[i].item()
                     for i in range(data['addr'].num_nodes) }

    # 2) Preparar addr_classes si no ve donat
    if addr_classes is None:
        addr_classes = { addr_map[str(i)]: int(data['addr'].y[i].item())
                         for i in range(data['addr'].num_nodes) }

    # 3) Preparar tx_map si no ve donat
    if tx_map is None:
        # Suposem que data['tx'].id existeix com a tensor de hashes
        tx_map = { str(i): data['tx'].id[i].item()
                   for i in range(data['tx'].num_nodes) }

    # Afegir nodes addr (amb global_idx)
    for idx in range(data['addr'].num_nodes):
        G.add_node(f'addr_{idx}',
                  type='addr',
                  global_idx=idx,                        # <-- aquí
                  btc_address=idx2addr[idx],
                  illicit=int(data['addr'].y[idx].item()))

    for idx in range(data['tx'].num_nodes):
        G.add_node(f'tx_{idx}',
                  type='tx',
                  global_idx=idx,                        # <-- i aquí
                  tx_hash=idx2tx[idx])

    # --- Arestes input: addr -> tx ---
    src, dst = data['addr', 'input', 'tx'].edge_index
    for a, t in zip(src.tolist(), dst.tolist()):
        G.add_edge(f'addr_{a}', f'tx_{t}', type='input')

    # --- Arestes output: tx -> addr ---
    src, dst = data['tx', 'output', 'addr'].edge_index
    for t, a in zip(src.tolist(), dst.tolist()):
        G.add_edge(f'tx_{t}', f'addr_{a}', type='output')

    return G

gp_path = RES_DIR / 'graph_lcc.gpickle'

"""if gp_path.exists():
    G_lcc = nx.read_gpickle(gp_path)
    print(f"✅ Grafo cargado desde {gp_path}")
    n_lcc_nodes = G_lcc.number_of_nodes()
    n_illicit_lcc = sum(d['illicit'] for _, d in G_lcc.nodes(data=True) if d['type']=='addr')
    n_licit_lcc   = sum(1 - d['illicit'] for _, d in G_lcc.nodes(data=True) if d['type']=='addr')
    print(f"\nBefore LCC: Nodes={G.number_of_nodes()}, Addr={n_addr_orig} (il·lícits={n_illicit}, lícits={n_licit}), Edges={G.number_of_edges()}")
    print(f"LCC: Nodes={n_lcc_nodes}, Addr={(n_illicit_lcc + n_licit_lcc)} (il·lícits={n_illicit_lcc}, lícits={n_licit_lcc}), Edges={G_lcc.number_of_edges()}")

else: """
G = hetero_to_nx(full_graph, idx2addr, addr_classes, idx2tx)
print(f"G original: Nodes={G.number_of_nodes()}, Edges={G.number_of_edges()}")

# 2. Estadístiques abans de LCC
n_addr_orig = sum(1 for _, d in G.nodes(data=True) if d['type']=='addr')
n_illicit  = sum(d['illicit'] for _, d in G.nodes(data=True) if d['type']=='addr')
n_licit    = n_addr_orig - n_illicit
print(f"Before LCC: Nodes={G.number_of_nodes()}, Addr={n_addr_orig} (il·lícits={n_illicit}, lícits={n_licit}), Edges={G.number_of_edges()}")

# 3. Troba la component feblement conexa més gran
lcc_nodes = max(nx.weakly_connected_components(G), key=len)

# 4. Extreu el subgrau de la LCC
G_lcc = G.subgraph(lcc_nodes).copy()

# 5. Estadístiques després de LCC
n_lcc_nodes = G_lcc.number_of_nodes()
n_illicit_lcc = sum(d['illicit'] for _, d in G_lcc.nodes(data=True) if d['type']=='addr')
n_licit_lcc   = sum(1 - d['illicit'] for _, d in G_lcc.nodes(data=True) if d['type']=='addr')
print(f"After LCC: Nodes={n_lcc_nodes}, Addr={(n_illicit_lcc + n_licit_lcc)} (il·lícits={n_illicit_lcc}, lícits={n_licit_lcc}), Edges={G_lcc.number_of_edges()}")

# 6. Desa el subgrau LCC per inspecció posterior
nx.write_gpickle(G_lcc, RES_DIR / 'graph_lcc.gpickle')
print("Component guardada!")


# In[28]:


def nx_to_hetero_lcc_preserve_ids(G_lcc):
    data = HeteroData()

    # 1) Recollim els node‐keys i el seu ordre local
    addr_nodes = [n for n,d in G_lcc.nodes(data=True) if d['type']=='addr']
    tx_nodes   = [n for n,d in G_lcc.nodes(data=True) if d['type']=='tx']
    addr2loc   = {n:i for i,n in enumerate(addr_nodes)}
    tx2loc     = {n:i for i,n in enumerate(tx_nodes)}

    data['addr'].num_nodes = len(addr_nodes)
    data['tx'].num_nodes   = len(tx_nodes)

    # 2) Build data['addr'].id i data['addr'].y directament de G_lcc
    id_list = []
    y_list  = []
    for n in addr_nodes:
        # n és "addr_1234": extraiem l'índex global
        gl_idx = int(n.split('_', 1)[1])
        id_list.append(gl_idx)
        y_list.append(G_lcc.nodes[n]['illicit'])
    data['addr'].id = torch.tensor(id_list, dtype=torch.long)
    data['addr'].y  = torch.tensor(y_list,  dtype=torch.long)    
    

    # 3) Build data['tx'].id
    tx_id_list = [int(n.split('_',1)[1]) for n in tx_nodes]
    data['tx'].id = torch.tensor(tx_id_list, dtype=torch.long)

    # 4) Edges input (addr → tx)
    src_in, dst_in = [], []
    for u,v,d in G_lcc.edges(data=True):
        if d['type']=='input':
            src_in.append(addr2loc[u])
            dst_in.append(tx2loc[v])
    data['addr','input','tx'].edge_index = torch.tensor([src_in, dst_in], dtype=torch.long)

    # 5) Edges output (tx → addr)
    src_out, dst_out = [], []
    for u,v,d in G_lcc.edges(data=True):
        if d['type']=='output':
            src_out.append(tx2loc[u])
            dst_out.append(addr2loc[v])
    data['tx','output','addr'].edge_index = torch.tensor([src_out, dst_out], dtype=torch.long)

    return data

# ────────────────────────────────────────────────────
# Ús:
# ja tinc carregat full_graph i G_lcc (networkx)
# idx2addr, addr_classes, idx2tx definits com:
#   idx2addr = {0:'1A1z…', 1:'3J98…', …}
#   addr_classes = {'1A1z…':0, …}
#   idx2tx = {0:'tx_hash1', …}

hd_path = RES_DIR / "graph_lcc_data.pt"
graph_lcc = nx_to_hetero_lcc_preserve_ids(G_lcc)
torch.save(graph_lcc, hd_path)
print(f"✅ HeteroData LCC desat a {hd_path}")
print("Hi ha il·licits segons y:", graph_lcc['addr'].y.sum().item())


# In[7]:


def draw_egonet(G, centre = None, layout = "spring", figsize=(8, 6), node_size=300, edge_alpha=0.6, seed = 42):
    """
    Mostra a pantalla un egonet d'adreces ('addr_*') i transaccions ('tx_*').

    Parameters
    ----------
    G : nx.DiGraph
        Grafo amb nodes que tenen attr "type" = {'addr','tx'}
        i arestes amb attr "type" = {'input','output'}.

    centre : str | None
        Node central del qual has fet BFS; es destaca amb un anell.

    layout : {'spring','kamada','bipartite'}
        Algoritme de col·locació.

    figsize : tuple
        Mides (polzades) de la figura.

    Altres paràmetres visuals opcions.
    """
    # ── 1. Posicions ---------------------------------------------------
    if layout == "spring":
        pos = nx.spring_layout(G, seed=seed)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "bipartite":
        # adreces a dalt, tx a baix
        addr = [n for n, d in G.nodes(data=True) if d["type"] == "addr"]
        tx   = [n for n, d in G.nodes(data=True) if d["type"] == "tx"]
        pos = dict()
        pos.update((n, (0, i)) for i, n in enumerate(addr))
        pos.update((n, (1, i)) for i, n in enumerate(tx))
    else:
        raise ValueError("layout ha de ser 'spring', 'kamada' o 'bipartite'")

    # ── 2. Separar nodes segons tipus ---------------------------------
    addr_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "addr"]
    tx_nodes   = [n for n, d in G.nodes(data=True) if d["type"] == "tx"]

    # ── 3. Dibuix ------------------------------------------------------
    plt.figure(figsize=figsize)
    # nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=addr_nodes,
        node_shape="o",
        node_color="tab:blue",
        node_size=node_size,
        label="addr"
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=tx_nodes,
        node_shape="s",
        node_color="tab:orange",
        node_size=node_size * 0.9,
        label="tx"
    )
    # centre (cercle exterior)
    if centre is not None and centre in G:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[centre],
            node_size=node_size * 1.4,
            node_color="none",
            edgecolors="red",
            linewidths=2.0,
        )

    # edges (colors per tipus)
    inp_edges  = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "input"]
    out_edges  = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "output"]

    nx.draw_networkx_edges(G, pos, edgelist=inp_edges,
                           arrows=True, alpha=edge_alpha,
                           edge_color="tab:green", label="input")
    nx.draw_networkx_edges(G, pos, edgelist=out_edges,
                           arrows=True, alpha=edge_alpha,
                           edge_color="tab:red",   label="output")

    # opcional: etiquetes
    nx.draw_networkx_labels(G, pos, font_size=8)

    # ── 4. Llegenda i eixos -------------------------------------------
    plt.legend(scatterpoints=1, fontsize=8)
    plt.axis("off")
    plt.title("Egonet centre: {}".format(centre if centre else "—"))
    plt.tight_layout()
    plt.show()


# In[40]:


SCHEMAS = {
    ('input','output')                    : 'star_in',
    ('output','input')                    : 'star_out',
    '2star_in'   : '2star_in',
    '2star_out'  : '2star_out',
    ('input','output','output*')          : 'split_2+',
    ('output','input','input*')           : 'merge_2+',
    'peeling_2'  : 'peeling_2',
    'peeling_3'  : 'peeling_3',
}

import json

with open(DATA_DIR / "pagerank.json")    as f: pagerank_js    = json.load(f)
with open(DATA_DIR / "clustering.json")  as f: clustering_js  = json.load(f)
with open(DATA_DIR / "abs_betweenness.json") as f: betweenness_js = json.load(f)
with open(DATA_DIR / "eccentricity.json")    as f: eccentricity_js= json.load(f)

N_ADDR = graph_lcc['addr'].num_nodes
cent = {
    'pagerank'    : { idx: pagerank_js    .get(idx2addr[idx], 0.0) for idx in range(N_ADDR) },
    'clustering'  : { idx: clustering_js  .get(idx2addr[idx], 0.0) for idx in range(N_ADDR) },
    'betweenness' : { idx: betweenness_js .get(idx2addr[idx], 0.0) for idx in range(N_ADDR) },
    'eccentricity': { idx: eccentricity_js.get(idx2addr[idx],   0  ) for idx in range(N_ADDR) },
}


def build_csr(src, dst, N):
    order = np.argsort(src)
    src_s, dst_s = src[order], dst[order]
    counts = np.bincount(src_s, minlength=N)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    return dst_s.astype(np.int32), indptr.astype(np.int32)

def bfs_addr_tx(centre_idx, nbr_at, ptr_at, nbr_at_rev, ptr_at_rev,
                nbr_ta, ptr_ta, nbr_ta_rev, ptr_ta_rev):
    vis = {'addr':{centre_idx}, 'tx':set()}
    frontier = [('addr', centre_idx)]
    for _ in range(RADIUS):
        nxt = []
        for typ,idx in frontier:
            if typ=='addr':
                neigh = np.unique(np.concatenate([
                    nbr_at[ptr_at[idx]:ptr_at[idx+1]],
                    nbr_at_rev[ptr_at_rev[idx]:ptr_at_rev[idx+1]]]))
                for t in neigh:
                    if t not in vis['tx']:
                        vis['tx'].add(int(t))
                        nxt.append(('tx', int(t)))
            else:
                neigh = np.unique(np.concatenate([
                    nbr_ta[ptr_ta[idx]:ptr_ta[idx+1]],
                    nbr_ta_rev[ptr_ta_rev[idx]:ptr_ta_rev[idx+1]]]))
                for a in neigh:
                    if a not in vis['addr']:
                        vis['addr'].add(int(a))
                        nxt.append(('addr', int(a)))
        if not nxt: break
        frontier = nxt
    return vis

def build_egonet_index(vis, centre_idx, sa, ta, sb, tb):
    G_ego = nx.DiGraph()
    # nodes
    for a in vis['addr']:
        G_ego.add_node(a, type='addr')
    for t in vis['tx']:
        G_ego.add_node(t, type='tx')
    # edges input
    mask_in  = np.isin(sa, list(vis['addr'])) & np.isin(ta, list(vis['tx']))
    for a, t in zip(sa[mask_in], ta[mask_in]):
        G_ego.add_edge(a, t, type='input')
    # edges output
    mask_out = np.isin(sb, list(vis['tx'])) & np.isin(tb, list(vis['addr']))
    for t, a in zip(sb[mask_out], tb[mask_out]):
        G_ego.add_edge(t, a, type='output')
    return G_ego

def count_peeling_chains(ego, centre_idx, addr_ids, depth = 2):
    centre = addr_ids[centre_idx]
    count = 0

    def dfs(a_node: str, remaining: int):
        nonlocal count
        if remaining == 0:
            count += 1
            return
        # A -> T inputs
        for _, t_node, d in ego.out_edges(a_node, data=True):
            if d['type'] != 'input': 
                continue
            # només un input i branching
            if ego.in_degree(t_node) != 1 or ego.out_degree(t_node) < 2:
                continue
            # tria l’output que segueix la cadena
            for _, a_next, d2 in ego.out_edges(t_node, data=True):
                if d2['type'] != 'output': 
                    continue
                if a_next == a_node:   # no seguim la branca de canvi
                    continue
                dfs(a_next, remaining - 1)

    dfs(centre, depth)
    return count

def count_metapaths_and_features(ego, centre_idx, cent):
    centre = addr_ids[centre_idx]
    cnt = Counter()
    # star_in / star_out
    for t in ego.successors(centre):
        et = ego[centre][t]['type']
        for a in ego.successors(t):
            if a==centre and ego[t][a]['type']==('output' if et=='input' else 'input'):
                cnt['star_in' if et=='input' else 'star_out'] += 1
    
    # 2-stars
    for n,d in ego.nodes(data=True):
        if d['type']!='tx': continue
        if ego.out_degree(n)>=2 and centre in ego.predecessors(n):
            cnt['2star_out'] += 1
        if ego.in_degree(n)>=2 and centre in ego.successors(n):
            cnt['2star_in'] += 1
    # split_2+ / merge_2+
    for t in ego.successors(centre):
        if ego[centre][t]['type']=='input' and ego.out_degree(t)>=2:
            cnt['split_2+'] += 1
    for t in ego.predecessors(centre):
        if ego[t][centre]['type']=='output' and ego.in_degree(t)>=2:
            cnt['merge_2+'] += 1

    feats = { name: cnt.get(name, 0) for name in SCHEMAS.values() }
    # graus
    feats['in_deg']  = ego.in_degree(centre)
    feats['out_deg'] = ego.out_degree(centre)
    
    # peelings
    feats['peeling_2'] = count_peeling_chains(ego, centre_idx, addr_ids, depth=2)
    feats['peeling_3'] = count_peeling_chains(ego, centre_idx, addr_ids, depth=3)
    
    # centralitats
    for key in ['pagerank','clustering','betweenness','eccentricity']:
        feats[key] = cent[key].get(centre_idx, 0.0)
    return feats

def process_single_from_idx(centre_idx, label, sa, ta, sb, tb, ptrs, cent, RADIUS, EGO_DIR):
    # 1) BFS i egonet (usar només índexs)
    vis = bfs_addr_tx(
        centre_idx, *ptrs
    )
    ego = build_egonet_index(vis, centre_idx, sa, ta, sb, tb)

    # 2) Filtra per connexitat i diàmetre exactes
    ego_und = ego.to_undirected()
    #print("Start dim")
    """if nx.diameter(ego_und, usebounds = True) != RADIUS:
        return None"""
    #print("End dim")
    # 3) Desa l’egonet si vols
    path = EGO_DIR / f'egonets_{label}_{RADIUS}' / f"{centre_idx}.gpickle"
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gpickle(ego, path)

    # 4) Comptar metapaths i extreure features
    feats = count_metapaths_and_features(ego, centre_idx, cent)
    row = {'centre_idx': centre_idx, 'label': label, **feats}
    return row

def build_dataset_from_hetero(full_graph, cent, RADIUS, EGO_DIR):
    # 1) Extreu edge_index i labels directament
    sa, ta = full_graph['addr','input','tx'].edge_index.cpu().numpy()
    sb, tb = full_graph['tx','output','addr'].edge_index.cpu().numpy()
    labels = full_graph['addr'].y.cpu().numpy()    # 0=licit,1=il·licit
    N_ADDR = full_graph['addr'].num_nodes

    # 2) Construeix CSR per al BFS
    nbr_at,     ptr_at     = build_csr(sa, ta, N_ADDR)   # addr→tx
    nbr_at_rev, ptr_at_rev = build_csr(tb, sb, N_ADDR)   # rev(tx→addr)
    nbr_ta,     ptr_ta     = build_csr(sb, tb, full_graph['tx'].num_nodes)  # tx→addr
    nbr_ta_rev, ptr_ta_rev = build_csr(ta, sa, full_graph['tx'].num_nodes)  # rev(addr→tx)
    ptrs = (nbr_at, ptr_at,
        nbr_at_rev, ptr_at_rev,
        nbr_ta, ptr_ta,
        nbr_ta_rev, ptr_ta_rev)

    # 3) Defineix pools d’índexs
    illicit_idxs = [i for i, lab in enumerate(labels) if lab == 1]
    licit_idxs   = [i for i, lab in enumerate(labels) if lab == 0]

    rows = []
    # 4) Processa il·lícits
    for centre_idx in tqdm(illicit_idxs, desc="Il·lícits"):
        r = process_single_from_idx(
            centre_idx, 1,
            sa, ta, sb, tb, ptrs, cent, RADIUS, EGO_DIR
        )
        if r: rows.append(r)

    # 5) Processa licits (mateixa mida que il·lícits)
    n_licit_idxs = random.sample(licit_idxs, len(illicit_idxs)*3)
    target = len(rows)
    for centre_idx in tqdm(n_licit_idxs, desc="Lícits"):
        if sum(r['label']==0 for r in rows) >= target:
            break
        r = process_single_from_idx(
            centre_idx, 0,
            sa, ta, sb, tb, ptrs, cent, RADIUS, EGO_DIR
        )
        if r: rows.append(r)

    return rows


# In[41]:


from tqdm import tqdm
import random 
full_graph = graph_lcc
print(f"✅ HeteroData LCC carregada: {graph_lcc}")
print("Il·lícits al nou hetero:", graph_lcc['addr'].y.sum().item())
rows = build_dataset_from_hetero(
    full_graph = graph_lcc,
    cent=cent,
    RADIUS=RADIUS,
    EGO_DIR=EGO_DIR)
print("Dataset generat:", len(rows), "samples")

# 6️⃣ DataFrame i afegir btc_address només al final
df = pd.DataFrame(rows).set_index('centre_idx')
df['btc_address'] = df.index.map(idx2addr)
df.to_csv(RES_DIR/f"full_dataset_{RADIUS}_1:3.csv", index=False, encoding='utf-8')
print("✅ Dataset guardat")