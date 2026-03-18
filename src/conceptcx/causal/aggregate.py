import torch

class CoverageAggregator:
    def __call__(self, scores, masks):
        numerator = (scores[:, :, None, None] * masks).sum(dim=1)  # [B, H, W]
        coverage = masks.mean(dim=1)                               # [B, H, W]

        saliency = torch.zeros_like(numerator)
        valid = coverage > 0
        saliency[valid] = numerator[valid] / coverage[valid]

        saliency_min = saliency.amin(dim=(-2, -1), keepdim=True)
        saliency_max = saliency.amax(dim=(-2, -1), keepdim=True)
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

        return saliency

if __name__ == "__main__":
    aggregator = CoverageAggregator()
    scores = torch.randn(2, 5)
    masks = torch.rand(2, 5, 4, 4)

    saliency = aggregator(scores, masks)
    print(saliency.shape)
    print(saliency)
