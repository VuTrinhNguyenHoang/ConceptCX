import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

class ViTCXGenerator:
    def __init__(self, image_size=224, delta=0.1):
        self.image_size = image_size
        self.delta = delta

    def build_mvit(self, features, grid_size):
        B, N, D = features.shape
        gh, gw = grid_size

        maps = features.permute(0, 2, 1).reshape(B, D, gh, gw)

        maps = F.interpolate(
            maps,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )

        maps_min = maps.amin(dim=(-2, -1), keepdim=True)
        maps_max = maps.amax(dim=(-2, -1), keepdim=True)
        maps = (maps - maps_min) / (maps_max - maps_min + 1e-8)

        return maps

    def cluster_masks(self, masks):
        D, H, W = masks.shape

        flat = masks.view(D, -1)
        flat = F.normalize(flat, dim=1).cpu().numpy()

        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=self.delta
        )

        labels = clustering.fit_predict(flat)

        clusters = []
        for k in range(labels.max() + 1):
            idx = (labels == k)
            clusters.append(masks[idx].mean(dim=0))

        return torch.stack(clusters, dim=0)

    def __call__(self, features, grid_size):
        mvit = self.build_mvit(features, grid_size)

        batch_masks = []
        for i in range(mvit.shape[0]):
            mcx = self.cluster_masks(mvit[i])
            batch_masks.append(mcx)

        return batch_masks
