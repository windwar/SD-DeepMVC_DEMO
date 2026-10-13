import numpy as np
import torch
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

import run_experiments as core
import large_data as large
from three_state_loss import ThreeStateLoss
from training_utils import epoch_features_and_labels, exact_knn_graph, structural_loss

class _HardForwardSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, original_loss, scores, score_gradient):
        ctx.save_for_backward(score_gradient)
        return original_loss.clone()

    @staticmethod
    def backward(ctx, upstream):
        gradient, = ctx.saved_tensors
        return upstream, upstream * gradient, None

class LearnableFusionLoss(ThreeStateLoss):
    def __init__(self, temperature_f):
        super().__init__(temperature_f)
        self.fusion_logits = torch.nn.Parameter(torch.zeros(4))
        self.frozen_weights = None

    def weights(self):
        return torch.softmax(self.fusion_logits, dim=0)

    def begin_epoch(self, num_views):
        super().begin_epoch(num_views)
        if hasattr(self, 'fusion_logits'):
            self.frozen_weights = self.weights().detach().clone()
        self._pair_cache = None


    def fuse(self, components, equal_weight_scores):
        if self.frozen_weights is None:
            raise RuntimeError('Call begin_epoch before fusing scores')

        delta = (self.frozen_weights - 0.25)[:, None, None]
        hard_scores = equal_weight_scores + (delta * components).sum(dim=0)
        current = self.weights()

        return hard_scores + ((current - current.detach())[:, None, None] * components).sum(dim=0)

    def feature_loss(self, zi, z, w, d, y_pse, density_quantile=None, combine_mode='tight'):
        if combine_mode != 'tight':
            raise ValueError('Only strict three-state mining is supported')
        if self.calls % self.num_views == 0:
            self._pair_cache = self.masks(w, d.detach(), y_pse, density_quantile)
        positive, negative, uncertain, eye = self._pair_cache
        self.calls += 1
        valid = negative.any(dim=1)
        if valid.any():
            logits = (torch.nn.functional.normalize(zi, dim=1)
                      @ torch.nn.functional.normalize(z, dim=1).t()) / self.temperature_f
            logits = logits[valid]
            pos, neg = positive[valid], negative[valid]
            logden = logits.masked_fill(~(pos | neg), -float('inf')).logsumexp(dim=1)
            count = pos.sum(dim=1)
            row_loss = (logden[:, None] - logits).masked_fill(~pos, 0).sum(dim=1) / count
            original = row_loss.mean()
        else:
            original = (zi.sum() + z.sum()) * 0.0
        gradient = torch.zeros_like(d)
        if valid.any():

            with torch.no_grad():
                logits = logits.detach()
                logden = logden.detach()
                per_positive = logden[:, None] - logits
                row_loss = row_loss.detach()
                exp_ratio = torch.exp(logits - logden[:, None])
                dp = (per_positive - row_loss[:, None]) / count[:, None] + exp_ratio
                dn = exp_ratio
                gate_p = (w.bool() & y_pse.bool() & ~eye)[valid]
                gate_n = ((~w.bool()) & (~y_pse.bool()) & ~eye)[valid]
                gradient[valid] = (dp * gate_p - dn * gate_n) / valid.sum()
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError('Nonfinite identity-STE fusion gradient')
        return _HardForwardSTE.apply(original, d, gradient), positive, negative

    def record_gradient(self):
        grad = self.fusion_logits.grad
        if grad is None or not torch.isfinite(grad).all():
            raise FloatingPointError('Fusion logits did not receive a finite gradient')


def component_graphs(common_z, args, use_sparse):
    features = np.ascontiguousarray(common_z.detach().cpu().numpy(), dtype=np.float32)
    n = len(features)
    if use_sparse:
        k = 15
        indices, backend = large.knn_for_view(features, k + 1, args.dynamic_knn_backend,
                                               args.cpu_threads, args.ivf_nprobe, args.hnsw_ef_search)
    else:
        k = min(15, max(3, n - 2))
        knn = NearestNeighbors(n_neighbors=k + 1).fit(features)
        _, indices = knn.kneighbors(features)
        backend = 'dense'
    neighbors = indices[:, 1:k + 1].astype(np.int32, copy=False)
    rows = np.repeat(np.arange(n, dtype=np.int32), k)
    cols = neighbors.reshape(-1)
    adjacency = sparse.csr_matrix((np.ones(rows.size, dtype=np.float32), (rows, cols)), shape=(n, n))
    rank_values = np.asarray([0.7 ** r for r in range(k)], dtype=np.float32)
    rank = sparse.csr_matrix((np.tile(rank_values, n), (rows, cols)), shape=(n, n))
    shared_counts = adjacency @ adjacency.T
    shared = shared_counts.astype(np.float32) * (1.0 / k)
    reciprocal = adjacency.multiply(adjacency.T)
    symmetric = rank + rank.T
    if use_sparse:

        baseline = (rank + shared_counts.astype(np.float32) * (1.0 / k)
                    + reciprocal + rank + rank.T) * 0.25
        baseline.setdiag(float(rank_values.sum(dtype=np.float32)))
        baseline = baseline.tocsr()
        wrows = np.repeat(np.arange(n, dtype=np.int32), 4)
        wcols = neighbors[:, :4].reshape(-1)
        directed = sparse.csr_matrix((np.ones(wrows.size, dtype=np.uint8), (wrows, wcols)), shape=(n, n))
        w = directed.maximum(directed.T).tocsr()
    else:

        baseline = ((rank.astype(np.float64) + shared_counts.astype(np.float64) / k
                     + reciprocal + symmetric) / 4).astype(np.float32).tolil()

        diagonal = np.concatenate([rank[start:start + 256].toarray().sum(axis=1, dtype=np.float32)
                                   for start in range(0, n, 256)])
        baseline.setdiag(diagonal)
        baseline = baseline.tocsr()
        w = exact_knn_graph(common_z, k=5)
    components = []
    for term in [rank, shared, reciprocal, symmetric]:
        term = term.copy().tocsr()
        if np.any(term.diagonal()):
            term.setdiag(0)
            term.eliminate_zeros()
        components.append(term)
    return w, baseline, components, backend

def finetune_epoch(model, data, loss, cfg, graphs, optimizer, args, use_sparse):
    model.train()
    loader, num_views, _, _ = core.get_multiview_data(data, cfg['batch_size'])
    criterion = torch.nn.MSELoss()
    common = epoch_features_and_labels(model, data, cfg['batch_size'])
    w, baseline, terms, backend = component_graphs(common, args, use_sparse)
    totals = []
    for inputs, _, sample_idx in loader:
        idx = sample_idx.detach().cpu().numpy().astype(np.int64, copy=False)
        device = inputs[0].device
        batch_w = large.sparse_batch(w, idx, device, torch.int32)
        batch_base = large.sparse_batch(baseline, idx, device, torch.float32)
        components = torch.stack([large.sparse_batch(term, idx, device, torch.float32) for term in terms])
        scores = loss.fuse(components, batch_base)
        values = graphs[idx]
        batch_u = (values[:, None] == values[None, :]).astype(
            np.float32 if use_sparse else np.float64, copy=False)
        labels = model.psedo_labels[sample_idx.to(model.psedo_labels.device)]
        equal = labels[:, None] == labels[None, :]
        reconstructions, embeddings = model(inputs)
        consensus = model.fusion(embeddings)
        losses = [cfg['alpha'] * structural_loss(batch_u, consensus)]
        for view in range(num_views):
            losses.append(criterion(inputs[view], reconstructions[view]))
            feature, _, _ = loss.feature_loss(embeddings[view], consensus, batch_w, scores, equal)
            losses.append(cfg['beta'] * feature)
        objective = sum(losses)
        optimizer.zero_grad()
        objective.backward()
        loss.record_gradient()
        optimizer.step()
        totals.append(objective.detach())

    values = torch.stack(totals).cpu().tolist()
    total = sum(values)
    return total, backend
