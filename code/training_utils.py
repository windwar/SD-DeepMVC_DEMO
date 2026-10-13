import numpy as np
import random
import torch
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph

def capture_rng_state():
    state = dict(python=random.getstate(), numpy=np.random.get_state(),
                 torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state

def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])

def cpu_snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value

def load_checkpoint(path, device='cpu'):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def structural_components(data, neighbor_count, delta):
    n = len(data)

    iter(torch.utils.data.DataLoader(data, batch_size=n, shuffle=False, drop_last=False))
    consensus = sparse.csr_matrix((n, n), dtype=np.int32)
    for view in data.data_views:
        values = view.detach().cpu().numpy().astype(np.float64, copy=False)
        a = kneighbors_graph(values, neighbor_count, mode='connectivity', include_self=True).astype(np.int32)
        shared = a @ a.T
        support = a.maximum(a.T)
        filtered = (shared >= delta).multiply(support).astype(np.int32)
        consensus = consensus + filtered
    majority = consensus > (len(data.data_views) / 2)
    _, components = connected_components(majority, directed=False)
    return components.astype(np.int32, copy=False)

def normalized_laplacian(u):
    if not np.array_equal(u, u.T):
        u = (u + u.T) / 2
        np.fill_diagonal(u, 1)
    degrees = np.sum(u, axis=1)
    degrees[degrees == 0] = 1e-6
    inverse = 1 / np.sqrt(degrees)
    return np.eye(len(u)) - (inverse[:, None] * u) * inverse[None, :]

def structural_loss(u, features):
    lap = torch.as_tensor(normalized_laplacian(u), dtype=torch.float32, device=features.device)
    return torch.sum(features * (lap @ features))

def epoch_features_and_labels(model, data, batch_size):
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False)
    features = []
    with torch.no_grad():
        for inputs, _, _ in loader:
            embeddings = [encoder(view) for encoder, view in zip(model.encoders, inputs)]
            features.append(model.fusion(embeddings))
    common = torch.cat(features)
    labels = model.ldpsc_clustering(common.cpu())
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=common.device).reshape(-1)
    if len(labels_tensor) != len(data) or not torch.isfinite(common).all() or (labels_tensor < 0).any():
        raise ValueError('Invalid epoch features or LDPSC labels')
    model.ldpsc_labels = labels
    model.dpcpsedo_labels = labels
    model.psedo_labels = labels_tensor.clone()

    iter(torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=False))
    return common

def exact_knn_graph(features, k=5):
    n, dim = features.shape
    if not 1 <= k <= n:
        raise ValueError('k must be between 1 and sample count')
    block = max(1, min(32, 8 * 1024**2 // max(1, n * dim * features.element_size())))
    columns = []
    for start in range(0, n, block):
        distance = ((features[None, :, :] - features[start:start + block, None, :]) ** 2).sum(dim=2)
        columns.append(torch.topk(distance, k, dim=1, largest=False).indices[:, 1:])
    cols = torch.cat(columns).cpu().numpy().reshape(-1)
    rows = np.repeat(np.arange(n), k - 1)
    directed = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, n))
    return directed.maximum(directed.T).tocsr()

def evaluate_final(model, data):
    from metric import evaluate
    model.eval()
    loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=False, drop_last=False)
    features, labels = [], []
    with torch.no_grad():
        for inputs, truth, _ in loader:
            embeddings = [encoder(view) for encoder, view in zip(model.encoders, inputs)]
            features.append(model.fusion(embeddings))
            labels.append(truth.cpu().numpy())
    consensus = torch.cat(features)
    prediction = np.asarray(model.ldpsc_clustering(consensus.cpu()))
    truth = np.concatenate(labels)
    nmi, ari, acc, pur, f_score = evaluate(truth.copy(), prediction)
    return (acc, nmi, pur, ari, f_score), prediction
