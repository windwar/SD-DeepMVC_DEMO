import numpy as np
import torch

from dataloader import get_multiview_data


def pre_train(model, mv_data, batch_size, epochs, optimizer):
    loader, num_views, _, _ = get_multiview_data(mv_data, batch_size)
    losses = np.zeros(epochs + 1, dtype=np.float64)
    criterion = torch.nn.MSELoss()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for views, _, _ in loader:
            reconstructions, _ = model(views)
            loss = sum(criterion(views[i], reconstructions[i]) for i in range(num_views))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        losses[epoch] = total
        if epoch % 100 == 0 or epoch == epochs:
            print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total))
    return losses
