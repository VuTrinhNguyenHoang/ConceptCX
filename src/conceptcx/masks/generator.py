import torch
import torch.nn.functional as F

class MaskGenerator:
    def __init__(self, image_size=224):
        self.image_size = image_size

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

        masks_min = masks.amin(dim=(-2, -1), keepdim=True)
        masks_max = masks.amax(dim=(-2, -1), keepdim=True)
        masks = (masks - masks_min) / (masks_max - masks_min + 1e-8)

        return masks
    
if __name__ == "__main__":
    generator = MaskGenerator(image_size=224)
    assignments = torch.rand(2, 196, 5)
    grid_size = (14, 14)

    masks = generator(assignments, grid_size)
    print(masks.shape)
