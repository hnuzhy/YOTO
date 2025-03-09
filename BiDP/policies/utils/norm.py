import numpy as np
import torch


class Normalizer(object):
    def __init__(self, data, symmetric=False, indices=None):
        if isinstance(data, dict):
            # load from existing data statistics
            self.stats = data
        elif symmetric:
            # just scaling applied in normalization, no bias
            # perform the same normalization in groups
            if indices is None:
                indices = np.arange(data.shape[-1])[None]

            self.stats = {
                "min": torch.zeros([data.shape[-1]]).to(data.device),
                "max": torch.ones([data.shape[-1]]).to(data.device),
            }
            for group in indices:
                max_abs = torch.abs(data[:, group]).max(0)[0].detach()
                limits = torch.ones_like(max_abs) * torch.max(max_abs)
                self.stats["max"][group] = limits
        else:
            mask = torch.zeros([data.shape[-1]]).to(data.device)
            if indices is not None:
                mask[indices.flatten()] += 1
            else:
                mask += 1
            self.stats = {
                "min": data.min(0)[0].detach() * mask,
                "max": data.max(0)[0].detach() * mask + 1.0 * (1 - mask),
            }

    def normalize(self, data):
        nd = len(data.shape)
        target_shape = (1,) * (nd - 1) + (data.shape[-1],)
        dmin = self.stats["min"].reshape(target_shape)
        dmax = self.stats["max"].reshape(target_shape)
        return (data - dmin) / (dmax - dmin + 1e-12)

    def unnormalize(self, data):
        nd = len(data.shape)
        target_shape = (1,) * (nd - 1) + (data.shape[-1],)
        dmin = self.stats["min"].reshape(target_shape)
        dmax = self.stats["max"].reshape(target_shape)
        return data * (dmax - dmin) + dmin

    def state_dict(self):
        return self.stats

    def load_state_dict(self, state_dict):
        self.stats = state_dict
