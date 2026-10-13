import torch
import torch.nn as nn
from LDP_SC import LDP_SC


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500), nn.ReLU(),
            nn.Linear(500, 500), nn.ReLU(),
            nn.Linear(500, 2000), nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000), nn.ReLU(),
            nn.Linear(2000, 500), nn.ReLU(),
            nn.Linear(500, 500), nn.ReLU(),
            nn.Linear(500, input_dim),
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, num_views, num_samples, num_clusters, input_size, feature_dim):
        super().__init__()
        self.encoders = nn.ModuleList([Encoder(input_size[i], feature_dim) for i in range(num_views)])
        self.decoders = nn.ModuleList([Decoder(input_size[i], feature_dim) for i in range(num_views)])
        self.num_views = num_views
        self.num_samples = num_samples
        self.num_clusters = num_clusters
        self.psedo_labels = torch.zeros(num_samples, dtype=torch.long, device='cuda')
        self.dpcpsedo_labels = torch.zeros(num_samples, dtype=torch.long, device='cuda')
        self.ldpsc_labels = torch.zeros(num_samples, dtype=torch.long, device='cuda')
        self.weights = nn.Parameter(torch.full((num_views,), 1 / num_views))

    def forward(self, xs):
        zs = [encoder(x) for encoder, x in zip(self.encoders, xs)]
        xrs = [decoder(z) for decoder, z in zip(self.decoders, zs)]
        return xrs, zs

    def get_weights(self):
        weights = torch.softmax(self.weights, dim=0)
        return weights / weights.sum()

    def fusion(self, zs):
        return torch.stack([z * weight for z, weight in zip(zs, self.get_weights())]).sum(dim=0)

    def ldpsc_clustering(self, feature):
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()
        return LDP_SC(feature, cluster_num=self.num_clusters, k=5)
