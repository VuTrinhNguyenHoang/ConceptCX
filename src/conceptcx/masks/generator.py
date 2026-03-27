import torch
import torch.nn.functional as F

class MaskGenerator:
    def __init__(self, image_size=224, normalize_mode="minmax", eps=1e-8):
        self.image_size = image_size
        self.normalize_mode = normalize_mode
        self.eps = eps

    def __call__(self, assignments, grid_size):
        B, N, K = assignments.shape
        gh, gw = grid_size

        masks = assignments.permute(0, 2, 1).reshape(B, K, gh, gw)

        masks = F.interpolate(
            masks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False
        )

        if self.normalize_mode == "minmax":
            masks_min = masks.amin(dim=(-2, -1), keepdim=True)
            masks_max = masks.amax(dim=(-2, -1), keepdim=True)
            masks = (masks - masks_min) / (masks_max - masks_min + self.eps)
        elif self.normalize_mode == "mass":
            masks = masks / (masks.sum(dim=1, keepdim=True) + self.eps)
        elif self.normalize_mode == "none":
            pass
        else:
            raise ValueError(f"Unsupported normalize_mode: {self.normalize_mode}")

        return masks
    
if __name__ == "__main__":
    generator = MaskGenerator(image_size=224, normalize_mode="mass")
    assignments = torch.rand(2, 196, 5)
    grid_size = (14, 14)

    masks = generator(assignments, grid_size)
    print(masks.shape)