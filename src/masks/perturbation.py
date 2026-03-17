import torch

class MaskPerturbation:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
    
    def __call__(self, images, masks):
        B, C, H, W = images.shape
        _, K, _, _ = masks.shape

        images = images.unsqueeze(1)        # [B, 1, C, H, W]
        masks = masks.unsqueeze(2)          # [B, K, 1, H, W]

        noise = torch.randn(
            B, K, C, H, W,
            device=images.device,
            dtype=images.dtype
        ) * self.noise_std
        eps = (1 - masks) * noise

        x_masked = images * masks + eps
        x_noise = images + eps

        return x_masked, x_noise

if __name__ == "__main__":
    perturbation = MaskPerturbation(noise_std=0.1)
    images = torch.randn(2, 3, 4, 4)
    masks = torch.rand(2, 5, 4, 4)
    x_masked, x_noise = perturbation(images, masks)
    print(x_masked.shape)  # Should be [2, 5, 3, 4, 4]
    print(x_noise.shape)   # Should be [2, 5, 3, 4, 4]
