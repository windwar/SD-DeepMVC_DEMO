from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import Dataset
import hdf5storage


class MultiviewData(Dataset):
    def __init__(self, name, device, path=None):
        root = Path(path) if path is not None else Path(__file__).resolve().parent.parent / 'datasets'
        files = {
            'MSRC': 'MSRC.mat',
            'Notting-Hill': 'Notting-Hill.mat',
            'Caltech7': 'Caltech-7.mat',
            '100Leaves': '100Leaves.mat',
            'COIL20': 'COIL20.mat',
            'uci-digit': 'uci-digit.mat',
        }
        if name not in files:
            raise ValueError(name)
        mat = hdf5storage.loadmat(str(root / files[name]))
        if name == 'MSRC':
            views = [mat[f'msr{i}'] for i in range(1, 6)]
            labels = mat['truth']
        elif name == 'uci-digit':
            views = [mat['mfeat_fac'], mat['mfeat_fou'], mat['mfeat_kar']]
            labels = mat['truth']
        elif name in {'Caltech7', 'Notting-Hill'}:
            views = list(mat['fea'][0])
            labels = mat['gt']
        else:
            views = list(mat['X'][0])
            labels = mat['Y']
        scaler = MinMaxScaler()
        self.data_views = [torch.from_numpy(scaler.fit_transform(np.asarray(view, dtype=np.float32))).to(device) for view in views]
        self.labels = np.asarray(labels).reshape(-1).astype(np.int32)
        self.num_views = len(self.data_views)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return [view[index] for view in self.data_views], self.labels[index], index


def get_multiview_data(mv_data, batch_size):
    loader = torch.utils.data.DataLoader(mv_data, batch_size=batch_size, shuffle=True, drop_last=False)
    return loader, len(mv_data.data_views), len(mv_data.labels), len(np.unique(mv_data.labels))
