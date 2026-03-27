import torch
import torch.nn.functional as F

class DebiasedCausalScorer:
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, images, x_masked, x_noise, target_indices):
        B, K, C, H, W = x_masked.shape

        # f(y | X)
        logits = self.model(images)
        probs = F.softmax(logits, dim=1)
        base_scores = probs.gather(1, target_indices[:, None]).squeeze(1)  # [B]

        target_expand = target_indices[:, None].expand(B, K).reshape(B * K)
        # f(y | X * M + eps)
        x_masked = x_masked.reshape(B * K, C, H, W)
        masked_logits = self.model(x_masked)
        masked_probs = F.softmax(masked_logits, dim=1)
        masked_score = masked_probs.gather(1, target_expand[:, None]).squeeze(1)
        masked_score = masked_score.view(B, K)

        # f(y | X + eps)
        x_noise = x_noise.reshape(B * K, C, H, W)
        noise_logits = self.model(x_noise)
        noise_probs = F.softmax(noise_logits, dim=1)
        noise_score = noise_probs.gather(1, target_expand[:, None]).squeeze(1)
        noise_score = noise_score.view(B, K)

        alpha = masked_score + (base_scores[:, None] - noise_score)
        return alpha
    
if __name__ == "__main__":
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            B = x.shape[0]
            return torch.randn(B, 10)

    model = DummyModel()
    scorer = DebiasedCausalScorer(model)
    images = torch.randn(2, 3, 4, 4)
    x_masked = torch.randn(2, 5, 3, 4, 4)
    x_noise = torch.randn(2, 5, 3, 4, 4)
    target_indices = torch.tensor([1, 3])

    scores = scorer(images, x_masked, x_noise, target_indices)
    print(scores.shape)  # Should be [2, 5]
    print(scores)
