import torch

class ViTExtractor:
    def __init__(self, model):
        self.model = model.eval()
    
    def __call__(self, images):
        with torch.no_grad():
            features = self.model.forward_features(images)

        features = features[:, 1:, :] # Remove CLS

        B, N, C = features.shape
        g = int(N ** 0.5)

        return features, (g, g)

if __name__ == "__main__":
    from timm import create_model
    model = create_model("vit_base_patch16_224", pretrained=True)
    # print(model)
    extractor = ViTExtractor(model)

    images = torch.randn(2, 3, 224, 224)
    features, (H, W) = extractor(images)

    print(features.shape)
    print(H, W)
