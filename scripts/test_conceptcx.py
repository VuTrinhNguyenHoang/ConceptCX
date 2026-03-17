import torch
import timm

from extractors.vit import ViTExtractor
from masks.generator import MaskGenerator
from masks.perturbation import MaskPerturbation
from causal.scorer import DebiasedCausalScorer
from causal.aggregate import CoverageAggregator

from conceptcx.bank import ConceptPrototypes

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
    # build prototype bank from dummy train features
    # --------------------------------------------------
    D = model.num_features
    train_features = torch.randn(2000, D)

    prototypes = ConceptPrototypes(K=8, tau=5, device=device)
    prototypes.fit(train_features)

    print(f"Prototypes shape: {prototypes.prototypes.shape}")

    # --------------------------------------------------
    # dummy input images
    # --------------------------------------------------
    B = 2
    images = torch.randn(B, 3, 224, 224).to(device)
    target_indices = torch.randint(0, 1000, (B,)).to(device)

    # --------------------------------------------------
    # feature extraction
    # --------------------------------------------------
    features, grid_size = extractor(images)
    print(f"Extracted features shape: {features.shape}")
    print(f"Grid size: {grid_size}")

    # --------------------------------------------------
    # concept assignment
    # --------------------------------------------------
    assignments = prototypes(features)
    print(f"Concept assignments shape: {assignments.shape}")

    # --------------------------------------------------
    # mask generation
    # --------------------------------------------------
    mask_generator = MaskGenerator(image_size=224)
    masks = mask_generator(assignments, grid_size=grid_size)
    print(f"Masks shape: {masks.shape}")

    # --------------------------------------------------
    # mask perturbation
    # --------------------------------------------------
    perturbation = MaskPerturbation(noise_std=0.1)
    x_masked, x_noise = perturbation(images, masks)
    print(f"Masked images shape: {x_masked.shape}")
    print(f"Noisy images shape: {x_noise.shape}")

    # --------------------------------------------------
    # causal scoring
    # --------------------------------------------------
    scorer = DebiasedCausalScorer(model)
    scores = scorer(images, x_masked, x_noise, target_indices)
    print(f"Causal scores shape: {scores.shape}")

    # --------------------------------------------------
    # aggregation
    # --------------------------------------------------
    aggregator = CoverageAggregator()
    saliency = aggregator(scores, masks)
    print(f"Saliency shape: {saliency.shape}")
    print(f"Saliency min/max: {saliency.min()}/{saliency.max()}")

if __name__ == "__main__":
    main()
