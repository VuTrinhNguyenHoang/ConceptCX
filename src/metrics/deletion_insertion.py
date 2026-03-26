import torch
from tqdm.auto import tqdm

from .utils import get_order


def _make_black_baseline(images):
    # Black canvas in ImageNet-normalized space.
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device, dtype=images.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device, dtype=images.dtype)
    baseline = ((0.0 - mean) / std).view(1, images.shape[1], 1, 1)
    return baseline.expand_as(images)


def _forward_chunk(model, x, labels, chunk_size):
    scores = []
    for x_chunk, y_chunk in zip(torch.split(x, chunk_size), torch.split(labels, chunk_size)):
        logits = model(x_chunk)
        probs = torch.softmax(logits, dim=1)
        score = probs.gather(1, y_chunk[:, None]).squeeze(1)
        scores.append(score)
    return torch.cat(scores, dim=0)


def deletion_insertion_auc(
    model,
    loader,
    saliency_cache,
    device,
    steps=20,
    chunk_size=64,
    baseline_mode="black",
):
    model.eval()

    deletion_all = []
    insertion_all = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing Del/Ins AUC"):
            if batch is None:
                continue

            images = batch["images"].to(device)
            B, C, H, W = images.shape

            logits = model(images)
            labels = logits.argmax(dim=1)

            # ===== load saliency =====
            sal = torch.stack([saliency_cache[i] for i in batch["idx"]]).to(device)  # [B,H,W]
            order = get_order(sal)  # [B, HW]

            flat_img = images.reshape(B, C, -1)

            if baseline_mode == "black":
                baseline = _make_black_baseline(images)
            elif baseline_mode == "mean":
                baseline = images.mean(dim=(2, 3), keepdim=True).expand_as(images)
            else:
                raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")

            flat_base = baseline.reshape(B, C, -1)

            del_curve = []
            ins_curve = []

            for s in range(steps + 1):
                k = int((s / steps) * flat_img.shape[-1])

                x_del = flat_img.clone()
                x_ins = flat_base.clone()

                mask = torch.zeros_like(flat_img[:, 0, :], dtype=torch.bool)  # [B, HW]

                for b in range(B):
                    mask[b, order[b, :k]] = True

                mask = mask.unsqueeze(1).expand(-1, C, -1)  # [B, C, HW]

                # deletion: remove important pixels
                x_del[mask] = flat_base[mask]

                # insertion: add important pixels
                x_ins[mask] = flat_img[mask]

                x_del = x_del.reshape(B, C, H, W)
                x_ins = x_ins.reshape(B, C, H, W)

                del_score = _forward_chunk(model, x_del, labels, chunk_size)
                ins_score = _forward_chunk(model, x_ins, labels, chunk_size)

                del_curve.append(del_score)
                ins_curve.append(ins_score)

            del_curve = torch.stack(del_curve, dim=1)  # [B, steps+1]
            ins_curve = torch.stack(ins_curve, dim=1)

            # ===== normalize curve (CỰC QUAN TRỌNG) =====
            del_curve = del_curve / (del_curve[:, :1] + 1e-8)
            ins_curve = ins_curve / (ins_curve[:, -1:] + 1e-8)

            deletion_all.append(del_curve.cpu())
            insertion_all.append(ins_curve.cpu())

    deletion_all = torch.cat(deletion_all, dim=0)
    insertion_all = torch.cat(insertion_all, dim=0)

    dx = 1.0 / steps
    deletion_auc = torch.trapz(deletion_all, dx=dx, dim=1).mean().item()
    insertion_auc = torch.trapz(insertion_all, dx=dx, dim=1).mean().item()

    return deletion_auc, insertion_auc
