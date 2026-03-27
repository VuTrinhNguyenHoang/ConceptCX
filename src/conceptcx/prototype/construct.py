import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


_EPS = 1e-12


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Expected (images, labels) batch.")
        return batch[0], batch[1]

    if isinstance(batch, dict):
        return batch["images"], batch["labels"]

    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _tensor_stats(tensors):
    if not tensors:
        return float("nan"), float("nan")

    values = torch.cat(tensors).float()
    return float(values.mean().item()), float(values.std(unbiased=False).item())


def _select_top_mid_indices(norms, patches_per_image, top_ratio=0.8, mid_range=(0.25, 0.75)):
    if not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be in (0, 1].")

    mid_low, mid_high = mid_range
    if not 0 <= mid_low < mid_high <= 1:
        raise ValueError("mid_range must satisfy 0 <= low < high <= 1.")

    batch_size, num_patches = norms.shape
    num_selected = min(int(patches_per_image), num_patches)
    if num_selected <= 0:
        raise ValueError("patches_per_image must be positive.")

    num_top = min(num_selected, max(1, int(round(num_selected * top_ratio))))
    num_mid = num_selected - num_top

    order = norms.argsort(dim=1, descending=True)
    top_idx = order[:, :num_top]

    if num_mid == 0:
        return top_idx, top_idx, None

    start = min(max(int(num_patches * mid_low), 0), num_patches - 1)
    end = min(max(int(num_patches * mid_high), start + 1), num_patches)

    mid_list = []
    for b in range(batch_size):
        taken = torch.zeros(num_patches, dtype=torch.bool, device=norms.device)
        taken[top_idx[b]] = True

        candidates = order[b, start:end]
        candidates = candidates[~taken[candidates]]

        if candidates.numel() < num_mid:
            remaining = order[b][~taken[order[b]]]
            candidates = remaining

        if candidates.numel() == 0:
            candidates = order[b, :1]

        if candidates.numel() >= num_mid:
            choice = candidates[torch.randperm(candidates.numel(), device=candidates.device)[:num_mid]]
        else:
            choice = candidates[torch.arange(num_mid, device=candidates.device) % candidates.numel()]

        mid_list.append(choice)

    mid_idx = torch.stack(mid_list, dim=0)
    selected_idx = torch.cat([top_idx, mid_idx], dim=1)
    return selected_idx, top_idx, mid_idx


@torch.no_grad()
def collect_features(
    extractor,
    loader,
    device,
    patches_per_image=20,
    top_ratio=0.8,
    mid_range=(0.25, 0.75),
):
    features = []
    patch_labels = []

    all_norms = []
    selected_norms = []
    top_norms = []
    mid_norms = []

    num_images = 0
    total_top = 0
    total_mid = 0

    for batch in loader:
        images, labels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats, _ = extractor(images)
        batch_size, _, channels = feats.shape

        patch_norms = feats.norm(dim=-1)
        selected_idx, top_idx, mid_idx = _select_top_mid_indices(
            patch_norms,
            patches_per_image=patches_per_image,
            top_ratio=top_ratio,
            mid_range=mid_range,
        )

        selected = torch.gather(
            feats,
            dim=1,
            index=selected_idx.unsqueeze(-1).expand(batch_size, selected_idx.shape[1], channels),
        )

        features.append(selected.reshape(-1, channels).cpu())
        patch_labels.append(labels[:, None].expand(batch_size, selected_idx.shape[1]).reshape(-1).cpu())

        all_norms.append(patch_norms.reshape(-1).cpu())
        selected_norms.append(torch.gather(patch_norms, 1, selected_idx).reshape(-1).cpu())
        top_norms.append(torch.gather(patch_norms, 1, top_idx).reshape(-1).cpu())

        total_top += batch_size * top_idx.shape[1]
        if mid_idx is not None:
            mid_norms.append(torch.gather(patch_norms, 1, mid_idx).reshape(-1).cpu())
            total_mid += batch_size * mid_idx.shape[1]

        num_images += batch_size

    x = torch.cat(features, dim=0).float()
    y = torch.cat(patch_labels, dim=0).long()

    all_norm_mean, all_norm_std = _tensor_stats(all_norms)
    selected_norm_mean, selected_norm_std = _tensor_stats(selected_norms)
    top_norm_mean, top_norm_std = _tensor_stats(top_norms)
    mid_norm_mean, mid_norm_std = _tensor_stats(mid_norms)

    total_selected = max(total_top + total_mid, 1)
    stats = {
        "num_images": int(num_images),
        "num_features": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "patches_per_image": float(x.shape[0] / max(num_images, 1)),
        "sampling_top_ratio": float(total_top / total_selected),
        "sampling_mid_ratio": float(total_mid / total_selected),
        "mid_quantile_low": float(mid_range[0]),
        "mid_quantile_high": float(mid_range[1]),
        "all_patch_norm_mean": all_norm_mean,
        "all_patch_norm_std": all_norm_std,
        "selected_norm_mean": selected_norm_mean,
        "selected_norm_std": selected_norm_std,
        "top_norm_mean": top_norm_mean,
        "top_norm_std": top_norm_std,
        "mid_norm_mean": mid_norm_mean,
        "mid_norm_std": mid_norm_std,
    }

    return x, y, stats


@torch.no_grad()
def compute_prototype_metrics(X, P, tau, labels=None, silhouette_samples=5000):
    x = X.to(P.device)
    x_norm = x.norm(dim=-1)

    x_unit = F.normalize(x, dim=-1)
    p_unit = F.normalize(P, dim=-1)

    sim = x_unit @ p_unit.T
    prob = F.softmax(tau * sim, dim=1)
    hard = sim.argmax(dim=1)

    top1 = prob.max(dim=1).values
    entropy = -(prob * (prob + _EPS).log()).sum(dim=1)

    if sim.shape[1] > 1:
        top2 = torch.topk(sim, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = torch.zeros_like(top1)

    binc = torch.bincount(hard, minlength=p_unit.shape[0]).float()
    used = binc > 0
    hard_mass = binc / binc.sum().clamp_min(_EPS)
    soft_mass = prob.mean(dim=0)

    assigned_cos = sim.gather(1, hard[:, None]).squeeze(1)

    proto_sim = p_unit @ p_unit.T
    off_diag_mask = ~torch.eye(p_unit.shape[0], dtype=torch.bool, device=p_unit.device)
    off_diag = proto_sim[off_diag_mask]

    def _effective_count(distribution):
        distribution = distribution[distribution > 0]
        if distribution.numel() == 0:
            return 0.0
        return float(torch.exp(-(distribution * distribution.log()).sum()).item())

    silhouette = float("nan")
    sample_size = min(int(silhouette_samples), x_unit.shape[0])
    if sample_size >= 2:
        sample_idx = torch.randperm(x_unit.shape[0], device=x_unit.device)[:sample_size]
        hard_sample = hard[sample_idx]
        if 1 < hard_sample.unique().numel() < hard_sample.numel():
            silhouette = float(
                silhouette_score(
                    x_unit[sample_idx].cpu().numpy(),
                    hard_sample.cpu().numpy(),
                    metric="cosine",
                )
            )

    metrics = {
        "feature_norm_mean": float(x_norm.mean().item()),
        "feature_norm_std": float(x_norm.std(unbiased=False).item()),
        "prototype_std_mean": float(torch.std(p_unit, dim=0).mean().item()),
        "top1_prob": float(top1.mean().item()),
        "top1_prob_std": float(top1.std(unbiased=False).item()),
        "entropy": float(entropy.mean().item()),
        "entropy_std": float(entropy.std(unbiased=False).item()),
        "margin": float(margin.mean().item()),
        "margin_std": float(margin.std(unbiased=False).item()),
        "cluster_usage": float(used.float().mean().item()),
        "dead_clusters": int((~used).sum().item()),
        "effective_clusters_hard": _effective_count(hard_mass),
        "effective_clusters_soft": _effective_count(soft_mass),
        "soft_mass_top1": float(soft_mass.max().item()),
        "soft_mass_active_1pct": int((soft_mass > 0.01).sum().item()),
        "cluster_size_min": int(binc[used].min().item()) if used.any() else 0,
        "cluster_size_max": int(binc[used].max().item()) if used.any() else 0,
        "cluster_size_mean": float(binc[used].mean().item()) if used.any() else 0.0,
        "cluster_size_std": float(binc[used].std(unbiased=False).item()) if used.any() else 0.0,
        "cluster_size_cv": float(
            binc[used].std(unbiased=False).item() / (binc[used].mean().item() + _EPS)
        ) if used.any() else 0.0,
        "assign_cos_mean": float(assigned_cos.mean().item()),
        "assign_cos_std": float(assigned_cos.std(unbiased=False).item()),
        "proto_sim_mean": float(off_diag.mean().item()) if off_diag.numel() else 0.0,
        "proto_sim_max": float(off_diag.max().item()) if off_diag.numel() else 0.0,
        "proto_sim_p95": float(torch.quantile(off_diag, 0.95).item()) if off_diag.numel() else 0.0,
        "silhouette": silhouette,
    }

    if labels is not None:
        y = labels.to(hard.device).view(-1)
        num_classes = int(y.max().item()) + 1 if y.numel() else 0

        purity_values = []
        label_entropy_values = []
        label_entropy_norm_values = []
        label_cardinality_values = []

        for cluster_id in range(p_unit.shape[0]):
            cluster_idx = torch.nonzero(hard == cluster_id, as_tuple=False).squeeze(1)
            if cluster_idx.numel() == 0:
                continue

            counts = torch.bincount(y[cluster_idx], minlength=num_classes).float()
            counts = counts[counts > 0]
            probs = counts / counts.sum()

            purity = probs.max()
            label_entropy = -(probs * (probs + _EPS).log()).sum()
            if probs.numel() > 1:
                label_entropy_norm = label_entropy / math.log(float(probs.numel()))
            else:
                label_entropy_norm = torch.zeros(1, device=probs.device).squeeze(0)

            purity_values.append(purity)
            label_entropy_values.append(label_entropy)
            label_entropy_norm_values.append(label_entropy_norm)
            label_cardinality_values.append(torch.tensor(float(probs.numel()), device=probs.device))

        if purity_values:
            purity_tensor = torch.stack(purity_values)
            label_entropy_tensor = torch.stack(label_entropy_values)
            label_entropy_norm_tensor = torch.stack(label_entropy_norm_values)
            label_cardinality_tensor = torch.stack(label_cardinality_values)

            metrics.update(
                {
                    "label_purity_mean": float(purity_tensor.mean().item()),
                    "label_purity_min": float(purity_tensor.min().item()),
                    "label_entropy_mean": float(label_entropy_tensor.mean().item()),
                    "label_entropy_norm_mean": float(label_entropy_norm_tensor.mean().item()),
                    "label_cardinality_mean": float(label_cardinality_tensor.mean().item()),
                }
            )

    return metrics


@torch.no_grad()
def collect_class_features(
    extractor,
    loader,
    device,
    patches_per_image=20,
    top_ratio=0.8,
    mid_range=(0.25, 0.75),
):
    class_features = defaultdict(list)
    class_selected_norms = defaultdict(list)
    class_top_norms = defaultdict(list)
    class_mid_norms = defaultdict(list)

    total_images = 0
    total_top = 0
    total_mid = 0

    for batch in loader:
        images, labels = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        feats, _ = extractor(images)
        batch_size, _, channels = feats.shape
        patch_norms = feats.norm(dim=-1)
        selected_idx, top_idx, mid_idx = _select_top_mid_indices(
            patch_norms,
            patches_per_image=patches_per_image,
            top_ratio=top_ratio,
            mid_range=mid_range,
        )

        selected = torch.gather(
            feats,
            dim=1,
            index=selected_idx.unsqueeze(-1).expand(batch_size, selected_idx.shape[1], channels),
        )
        selected_norm = torch.gather(patch_norms, 1, selected_idx)
        top_norm = torch.gather(patch_norms, 1, top_idx)
        mid_norm = torch.gather(patch_norms, 1, mid_idx) if mid_idx is not None else None

        for i in range(batch_size):
            class_idx = int(labels[i].item())

            class_features[class_idx].append(selected[i].detach().cpu())
            class_selected_norms[class_idx].append(selected_norm[i].reshape(-1).detach().cpu())
            class_top_norms[class_idx].append(top_norm[i].reshape(-1).detach().cpu())
            if mid_norm is not None:
                class_mid_norms[class_idx].append(mid_norm[i].reshape(-1).detach().cpu())

            total_top += int(top_idx.shape[1])
            if mid_idx is not None:
                total_mid += int(mid_idx.shape[1])

        total_images += batch_size

    features_by_class = {}
    per_class_stats = []

    for class_idx in sorted(class_features):
        x_class = torch.cat(class_features[class_idx], dim=0).float()
        features_by_class[int(class_idx)] = x_class

        selected_norm_mean, selected_norm_std = _tensor_stats(class_selected_norms[class_idx])
        top_norm_mean, top_norm_std = _tensor_stats(class_top_norms[class_idx])
        mid_norm_mean, mid_norm_std = _tensor_stats(class_mid_norms[class_idx])
        feature_norm = x_class.norm(dim=-1)
        support_images = len(class_features[class_idx])

        per_class_stats.append(
            {
                "class_idx": int(class_idx),
                "support_images": int(support_images),
                "support_features": int(x_class.shape[0]),
                "feature_dim": int(x_class.shape[1]),
                "patches_per_image": float(x_class.shape[0] / support_images),
                "selected_norm_mean": selected_norm_mean,
                "selected_norm_std": selected_norm_std,
                "top_norm_mean": top_norm_mean,
                "top_norm_std": top_norm_std,
                "mid_norm_mean": mid_norm_mean,
                "mid_norm_std": mid_norm_std,
                "feature_norm_mean": float(feature_norm.mean().item()),
                "feature_norm_std": float(feature_norm.std(unbiased=False).item()),
            }
        )

    total_selected = max(total_top + total_mid, 1)
    support_counts = torch.tensor([row["support_images"] for row in per_class_stats], dtype=torch.float32)
    summary = {
        "num_classes_collected": int(len(features_by_class)),
        "total_images": int(total_images),
        "total_support_images": int(sum(row["support_images"] for row in per_class_stats)),
        "total_support_features": int(sum(x.shape[0] for x in features_by_class.values())),
        "sampling_top_ratio": float(total_top / total_selected),
        "sampling_mid_ratio": float(total_mid / total_selected),
        "mid_quantile_low": float(mid_range[0]),
        "mid_quantile_high": float(mid_range[1]),
        "support_images_per_class_mean": float(support_counts.mean().item()),
        "support_images_per_class_min": int(support_counts.min().item()),
        "support_images_per_class_max": int(support_counts.max().item()),
    }

    return features_by_class, per_class_stats, summary


@torch.no_grad()
def build_classwise_prototypes(
    features_by_class,
    distance_threshold=0.1,
    use_medoid=True,
):
    prototypes_by_class = {}
    per_class_metrics = []

    concepts_per_class = []
    total_support_features = 0

    for class_idx in sorted(features_by_class):
        x = features_by_class[class_idx].float()
        x_unit = F.normalize(x, dim=-1)
        total_support_features += int(x_unit.shape[0])

        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(distance_threshold),
            metric="cosine",
            linkage="average",
            compute_full_tree=True,
        )
        labels = torch.from_numpy(agg.fit_predict(x_unit.cpu().numpy())).long()

        unique_labels = labels.unique(sorted=True)
        cluster_entries = []

        for cluster_id in unique_labels.tolist():
            member_idx = torch.nonzero(labels == cluster_id, as_tuple=False).squeeze(1)
            members = x_unit[member_idx]
            size = int(member_idx.numel())

            center = F.normalize(members.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            if use_medoid:
                sims_to_center = members @ center
                best_local_idx = int(sims_to_center.argmax().item())
                prototype = members[best_local_idx]
            else:
                prototype = center

            member_sims = members @ prototype
            cluster_entries.append(
                {
                    "cluster_id": int(cluster_id),
                    "size": size,
                    "prototype": prototype,
                    "member_sim_mean": float(member_sims.mean().item()),
                    "member_sim_std": float(member_sims.std(unbiased=False).item()),
                }
            )

        cluster_entries.sort(key=lambda entry: (-entry["size"], entry["cluster_id"]))
        prototypes = torch.stack([entry["prototype"] for entry in cluster_entries], dim=0)
        prototypes = F.normalize(prototypes, dim=-1)
        prototypes_by_class[int(class_idx)] = prototypes.cpu()

        cluster_sizes = torch.tensor([entry["size"] for entry in cluster_entries], dtype=torch.float32)
        cluster_sim_mean = torch.tensor([entry["member_sim_mean"] for entry in cluster_entries], dtype=torch.float32)
        cluster_sim_std = torch.tensor([entry["member_sim_std"] for entry in cluster_entries], dtype=torch.float32)

        if prototypes.shape[0] > 1:
            proto_sim = prototypes @ prototypes.T
            off_diag_mask = ~torch.eye(prototypes.shape[0], dtype=torch.bool)
            off_diag = proto_sim[off_diag_mask]
            proto_sim_mean = float(off_diag.mean().item())
            proto_sim_max = float(off_diag.max().item())
            proto_sim_p95 = float(torch.quantile(off_diag, 0.95).item())
        else:
            proto_sim_mean = 0.0
            proto_sim_max = 0.0
            proto_sim_p95 = 0.0

        concepts_per_class.append(int(prototypes.shape[0]))
        per_class_metrics.append(
            {
                "class_idx": int(class_idx),
                "support_features": int(x_unit.shape[0]),
                "num_concepts": int(prototypes.shape[0]),
                "raw_clusters": int(len(cluster_entries)),
                "singleton_clusters": int((cluster_sizes == 1).sum().item()),
                "cluster_size_min": int(cluster_sizes.min().item()),
                "cluster_size_max": int(cluster_sizes.max().item()),
                "cluster_size_mean": float(cluster_sizes.mean().item()),
                "cluster_size_std": float(cluster_sizes.std(unbiased=False).item()),
                "cluster_size_cv": float(cluster_sizes.std(unbiased=False).item() / (cluster_sizes.mean().item() + _EPS)),
                "intra_cluster_sim_mean": float(cluster_sim_mean.mean().item()),
                "intra_cluster_sim_std": float(cluster_sim_std.mean().item()),
                "proto_sim_mean": proto_sim_mean,
                "proto_sim_max": proto_sim_max,
                "proto_sim_p95": proto_sim_p95,
                "distance_threshold": float(distance_threshold),
                "use_medoid": bool(use_medoid),
            }
        )

    concepts_tensor = torch.tensor(concepts_per_class, dtype=torch.float32) if concepts_per_class else torch.zeros(0)
    summary = {
        "num_classes_built": int(len(prototypes_by_class)),
        "total_support_features": int(total_support_features),
        "total_concepts": int(concepts_tensor.sum().item()) if concepts_tensor.numel() else 0,
        "concepts_per_class_mean": float(concepts_tensor.mean().item()) if concepts_tensor.numel() else 0.0,
        "concepts_per_class_min": int(concepts_tensor.min().item()) if concepts_tensor.numel() else 0,
        "concepts_per_class_max": int(concepts_tensor.max().item()) if concepts_tensor.numel() else 0,
        "concepts_per_class_std": float(concepts_tensor.std(unbiased=False).item()) if concepts_tensor.numel() else 0.0,
        "distance_threshold": float(distance_threshold),
        "use_medoid": bool(use_medoid),
    }

    return prototypes_by_class, per_class_metrics, summary
