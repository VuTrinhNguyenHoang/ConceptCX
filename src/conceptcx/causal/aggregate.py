import torch

class CoverageAggregator:
    def __init__(self, score_activation="none", eps=1e-8):
        self.score_activation = score_activation
        self.eps = eps

    def __call__(self, scores, masks):
        if self.score_activation == "relu":
            scores = torch.relu(scores)
        elif self.score_activation != "none":
            raise ValueError(f"Unsupported score_activation: {self.score_activation}")

        numerator = (scores[:, :, None, None] * masks).sum(dim=1)  # [B, H, W]
        coverage = masks.sum(dim=1)                               # [B, H, W]

        saliency = torch.zeros_like(numerator)
        saliency = numerator / (coverage + self.eps)

        saliency_min = saliency.amin(dim=(-2, -1), keepdim=True)
        saliency_max = saliency.amax(dim=(-2, -1), keepdim=True)
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + self.eps)

        return saliency

if __name__ == "__main__":
    aggregator = CoverageAggregator(score_activation="relu")
    scores = torch.randn(2, 5)
    masks = torch.rand(2, 5, 4, 4)

    saliency = aggregator(scores, masks)
    print(saliency.shape)
    print(saliency)
