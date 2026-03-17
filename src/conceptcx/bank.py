import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

class ConceptPrototypes:
    def __init__(self, K=64, tau=5, device="cpu", random_state=42):
        self.K = K
        self.tau = tau
        self.prototypes = None
        self.device = device
        self.random_state = random_state

    def fit(self, features):
        features = F.normalize(features, dim=-1)
        features = features.detach().cpu().numpy()

        kmeans = KMeans(
            n_clusters=self.K,
            random_state=self.random_state,
            n_init=10
        )
        kmeans.fit(features)

        self.prototypes = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        self.prototypes = F.normalize(self.prototypes, dim=-1)

    def load(self, path):
        self.prototypes = torch.load(path, map_location=self.device)

    def save(self, path):
        torch.save(self.prototypes.cpu(), path)

    def __call__(self, features):
        features = F.normalize(features, dim=-1)

        sim = torch.einsum(
            "bnd,kd->bnk",
            features,
            self.prototypes
        )

        return F.softmax(self.tau * sim, dim=-1)

if __name__ == "__main__":
    train_features = torch.randn(32, 64)
    prototypes = ConceptPrototypes(K=4, tau=5, device="cpu")
    prototypes.fit(train_features)

    print(prototypes.prototypes.shape)

    features = torch.randn(2, 8, 64)
    sim = prototypes(features)
    print(sim.shape)
    print(sim)
