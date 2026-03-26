import torch

def get_order(saliency):
    B, H, W = saliency.shape
    flat = saliency.view(B, -1)
    return torch.argsort(flat, dim=1, descending=True)
