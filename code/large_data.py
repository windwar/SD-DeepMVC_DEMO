import math
import hashlib
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import torch
from torch.utils.data import Dataset

from network import Network


def prepare_view(raw, normalization):
    view = np.asarray(raw, dtype=np.float32, order='C')
    if normalization == 'minmax':
        view -= view.min(axis=0)
        scale = view.max(axis=0)
        scale[scale == 0] = 1.0
        view /= scale
        return view
    raise ValueError(normalization)


class LargeMultiviewData(Dataset):
    def __init__(self, name, dataset_dir, device, normalization='minmax'):
        path = Path(dataset_dir) / (name + '.mat')
        if not path.is_file():
            raise FileNotFoundError(path)
        mat = sio.loadmat(str(path))
        labels = np.asarray(mat['Y']).reshape(-1).astype(np.int64, copy=False)
        views = []
        for index, cell in enumerate(mat['X'].ravel()):
            array = np.asarray(cell)
            if array.ndim != 2:
                raise ValueError(f'{name} view {index}: {array.shape}')
            if array.shape[0] != labels.size:
                if array.shape[1] != labels.size:
                    raise ValueError(f'{name} view {index}: {array.shape}')
                array = array.T
            views.append(prepare_view(array, normalization))
        self.cpu_views = views
        self.data_views = [torch.from_numpy(view).to(device) for view in views]
        self.labels = labels
        self.num_views = len(views)

    def __len__(self):
        return self.labels.size

    def __getitem__(self, index):
        return [view[index] for view in self.data_views], self.labels[index], index


def make_network(dataset, cfg, device):
    input_sizes = np.asarray([view.shape[1] for view in dataset.data_views], dtype=int)
    classes = np.unique(dataset.labels).size
    model = Network(dataset.num_views, len(dataset), classes, input_sizes, cfg['feature_dim']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    return model, optimizer


def ensure_self_included(neighbors, n, k):
    output = np.empty((n, k), dtype=np.int32)
    for row in range(n):
        values = [row]
        seen = {row}
        for value in neighbors[row]:
            value = int(value)
            if 0 <= value < n and value not in seen:
                values.append(value)
                seen.add(value)
                if len(values) == k:
                    break
        if len(values) != k:
            raise RuntimeError(f'KNN row {row}: {len(values)}')
        output[row] = values
    return output


def faiss_knn(view, k, backend, threads, ivf_nprobe, hnsw_ef_search):
    import faiss

    faiss.omp_set_num_threads(max(1, threads))
    n, dimension = view.shape
    search_k = min(n, k + 8)
    if backend == 'faiss-hnsw':
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = max(80, 4 * k)
        index.hnsw.efSearch = max(hnsw_ef_search, 4 * k)
        actual = 'faiss-hnsw-cpu'
    elif backend == 'faiss-ivf':
        nlist = max(16, min(n // 20, int(4 * math.sqrt(n))))
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        train_count = min(n, max(10000, 40 * nlist))
        if train_count < n:
            rng = np.random.RandomState(2026)
            training = np.ascontiguousarray(view[rng.choice(n, size=train_count, replace=False)])
        else:
            training = view
        index.train(training)
        index.nprobe = min(nlist, max(1, ivf_nprobe))
        actual = f'faiss-ivf-cpu(nlist={nlist},nprobe={index.nprobe})'
    else:
        raise ValueError(backend)
    index.add(view)
    _, indices = index.search(view, search_k)
    return ensure_self_included(indices, n, k), actual


def choose_backend(requested, n):
    if requested != 'auto':
        return requested
    return 'faiss-ivf' if n >= 20000 else 'faiss-hnsw'


def knn_for_view(view, k, backend, threads, ivf_nprobe, hnsw_ef_search):
    chosen = choose_backend(backend, view.shape[0])
    return faiss_knn(view, k, chosen, threads, ivf_nprobe, hnsw_ef_search)


def candidate_common_counts(neighbors, neighbor_count, chunk_size):
    n = neighbors.shape[0]
    active = neighbors[:, :neighbor_count]
    sources = np.repeat(np.arange(n, dtype=np.int64), neighbor_count - 1)
    targets = active[:, 1:neighbor_count].reshape(-1).astype(np.int64, copy=False)
    low = np.minimum(sources, targets)
    high = np.maximum(sources, targets)
    codes = np.unique(low * np.int64(n) + high)
    sources = codes // n
    targets = codes % n
    common = np.empty(codes.size, dtype=np.uint16 if neighbor_count > 255 else np.uint8)
    for start in range(0, codes.size, chunk_size):
        stop = min(codes.size, start + chunk_size)
        left = active[sources[start:stop]]
        right = active[targets[start:stop]]
        common[start:stop] = (left[:, :, None] == right[:, None, :]).sum(axis=(1, 2), dtype=np.int32)
    return codes, common


def component_labels(n, view_edges, majority):
    nonempty = [edges for edges in view_edges if edges.size]
    if not nonempty:
        return np.arange(n, dtype=np.int32)
    combined = np.concatenate(nonempty)
    codes, counts = np.unique(combined, return_counts=True)
    edge_codes = codes[counts >= majority]
    if not edge_codes.size:
        return np.arange(n, dtype=np.int32)
    rows = (edge_codes // n).astype(np.int32, copy=False)
    cols = (edge_codes % n).astype(np.int32, copy=False)
    data = np.ones(edge_codes.size * 2, dtype=np.uint8)
    graph = sparse.coo_matrix((data, (np.concatenate([rows, cols]), np.concatenate([cols, rows]))), shape=(n, n)).tocsr()
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int32, copy=False)


def static_component_labels(dataset_name, dataset, cfg, args, output_dir):
    cache_key = (dataset_name, len(dataset), cfg['_structural_neighbors'], cfg['delta'],
                 args.normalization, args.static_knn_backend)
    cache_id = hashlib.sha256(repr(cache_key).encode('utf-8')).hexdigest()[:16]
    cache = output_dir / 'graph_cache' / f'{dataset_name}_{cache_id}.npy'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        return np.load(cache), 0.0, 'cached'
    started = time.perf_counter()
    view_edges = []
    backends = []
    for index, view in enumerate(dataset.cpu_views):
        print(f'Static graph view {index + 1}/{dataset.num_views}', flush=True)
        neighbors, actual = knn_for_view(view, cfg['_structural_neighbors'], args.static_knn_backend, args.cpu_threads, args.ivf_nprobe, args.hnsw_ef_search)
        codes, counts = candidate_common_counts(neighbors, cfg['_structural_neighbors'], args.edge_chunk_size)
        view_edges.append(codes[counts >= cfg['delta']])
        backends.append(actual)
    labels = component_labels(len(dataset), view_edges, dataset.num_views // 2 + 1)
    np.save(cache, labels)
    return labels, time.perf_counter() - started, ';'.join(backends)


def sparse_batch(matrix, indices, device, dtype):
    return torch.as_tensor(matrix[indices][:, indices].toarray(), dtype=dtype, device=device)
