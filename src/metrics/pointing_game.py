import torch

def pointing_game(saliency, bboxes):
    B, H, W = saliency.shape
    hits = 0

    for i in range(B):
        heat = saliency[i]
        idx = torch.argmax(heat).item()
        y, x = divmod(idx, W)

        boxes = bboxes[i]   # [N_box, 4]

        hit = False
        for box in boxes:
            xmin, ymin, xmax, ymax = box.tolist()
            if xmin <= x <= xmax and ymin <= y <= ymax:
                hit = True
                break

        if hit:
            hits += 1

    return hits, B

def pointing_game_acc(loader, saliency_cache):
    total_hits = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            sal = torch.stack([saliency_cache[i] for i in batch["idx"]])  # [B,H,W]
            bboxes = batch["bboxes"]

            hits, B = pointing_game(sal.cpu(), bboxes)

            total_hits += hits
            total_samples += B

    return total_hits / total_samples
