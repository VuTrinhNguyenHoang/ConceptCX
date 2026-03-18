import torch
import timm

from conceptcx.extractors.vit import ViTExtractor
from conceptcx.masks.generator import MaskGenerator
from conceptcx.masks.perturbation import MaskPerturbation
from conceptcx.causal.scorer import DebiasedCausalScorer
from conceptcx.causal.aggregate import CoverageAggregator

from vitcx.vitcx_generator import ViTCXGenerator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # model + extractor
    # --------------------------------------------------
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model = model.to(device).eval()

    extractor = ViTExtractor(model)

    # --------------------------------------------------
    # dummy input images
    # --------------------------------------------------
    B = 2
    images = torch.randn(B, 3, 224, 224).to(device)
    with torch.no_grad():
        target_indices = model(images).argmax(dim=1)
    print(f"Target indices: {target_indices}")

    # --------------------------------------------------
    # feature extraction
    # --------------------------------------------------
    features, grid_size = extractor(images)
    print(f"Extracted features shape: {features.shape}")
    print(f"Grid size: {grid_size}")

    # --------------------------------------------------
    # vitcx generator
    # --------------------------------------------------
    vitcx_generator = ViTCXGenerator(image_size=224, delta=0.1)
    batch_masks = vitcx_generator(features, grid_size=grid_size)

    for i, masks in enumerate(batch_masks):
        print(f"Image {i} - Masks shape: {masks.shape}")

    # --------------------------------------------------
    # modules
    # --------------------------------------------------
    perturbation = MaskPerturbation(noise_std=0.1)
    scorer = DebiasedCausalScorer(model)
    aggregator = CoverageAggregator()

    # --------------------------------------------------
    # compute saliency (loop per image)
    # --------------------------------------------------
    saliency_list = []

    for i in range(B):
        masks_i = batch_masks[i]              # [K_i, H, W]
        K_i = masks_i.shape[0]

        print(f"Image {i}: {K_i} masks")

        masks_i = masks_i.unsqueeze(0).to(device)   # [1, K_i, H, W]

        img_i = images[i:i+1]                       # [1,3,H,W]
        target_i = target_indices[i:i+1]

        # perturb
        x_masked, x_noise = perturbation(img_i, masks_i)

        # score
        scores = scorer(
            img_i,
            x_masked,
            x_noise,
            target_i
        )  # [1, K_i]

        # aggregate
        sal = aggregator(scores, masks_i)   # [1, H, W]

        saliency_list.append(sal)

    saliency = torch.cat(saliency_list, dim=0)

    print("Final saliency:", saliency.shape)

if __name__ == "__main__":
    main()
