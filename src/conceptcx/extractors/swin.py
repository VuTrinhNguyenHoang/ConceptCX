import torch

class SwinExtractor:
    def __init__(self, model):
        self.model = model.eval()

    def __call__(self, images):
        features = self.model.forward_features(images)

        B, H, W, C = features.shape
        features = features.view(B, H * W, C)

        return features, (H, W)

if __name__ == "__main__":
    from timm import create_model
    model = create_model("swin_base_patch4_window7_224", pretrained=True)
    # print(model)
    extractor = SwinExtractor(model)

    images = torch.randn(2, 3, 224, 224)
    features, (H, W) = extractor(images)

    print(features.shape)
    print(H, W)
