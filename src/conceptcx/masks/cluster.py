import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering


def _default_num_clusters(num_masks):
    if num_masks <= 0:
        raise ValueError("num_masks must be positive.")
    return min(num_masks, max(4, num_masks // 4))


def _fit_agglomerative(distance_matrix, n_clusters):
    kwargs = {
        "n_clusters": n_clusters,
        "linkage": "average",
    }

    try:
        clustering = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        clustering = AgglomerativeClustering(affinity="precomputed", **kwargs)

    return clustering.fit_predict(distance_matrix)


class PerImageMaskClusterer:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters

    def _resolve_num_clusters(self, num_masks):
        if self.n_clusters is None:
            return _default_num_clusters(num_masks)

        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        return min(self.n_clusters, num_masks)

    def __call__(self, masks):
        if masks.ndim != 4:
            raise ValueError(f"Expected masks with shape [B, K, H, W], got {tuple(masks.shape)}")

        B, K, _, _ = masks.shape
        target_clusters = self._resolve_num_clusters(K)
        if target_clusters == K:
            return masks

        flat = masks.flatten(2)
        flat = F.normalize(flat, dim=-1)
        similarity = torch.bmm(flat, flat.transpose(1, 2))

        merged_masks = []
        for b in range(B):
            distance = (1.0 - similarity[b]).clamp_min(0).cpu().numpy()
            labels = _fit_agglomerative(distance, target_clusters)
            labels = torch.from_numpy(labels).to(device=masks.device)

            merged = []
            for cluster_id in range(target_clusters):
                cluster_masks = masks[b][labels == cluster_id]
                merged.append(cluster_masks.mean(dim=0))

            merged_masks.append(torch.stack(merged, dim=0))

        return torch.stack(merged_masks, dim=0)
