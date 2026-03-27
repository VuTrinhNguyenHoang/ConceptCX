import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

class ConceptPrototypes:
    def __init__(self, K=64, tau=5, device=torch.device("cpu"), random_state=42, batch_size=4096, max_iter=300):
        self.K = K
        self.tau = tau
        self.prototypes = None
        self.device = device
        self.random_state = random_state
        self.batch_size = batch_size
        self.max_iter = max_iter

    def fit(self, features):
        features = F.normalize(features, dim=-1)
        features = features.detach().cpu().numpy()

        kmeans = MiniBatchKMeans(
            n_clusters=self.K,
            random_state=self.random_state,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            max_no_improvement=30,
            tol=0.0001,
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


class ClassConditionalConceptPrototypes:
    def __init__(self, tau=5, device=torch.device("cpu")):
        self.tau = tau
        self.device = device
        self.prototypes = {}

    def fit(self, prototypes_by_class):
        self.prototypes = {
            int(class_idx): F.normalize(prototypes.float().to(self.device), dim=-1)
            for class_idx, prototypes in prototypes_by_class.items()
        }

    def save(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)

        for class_idx, prototypes in self.prototypes.items():
            class_dir = root / f"class={class_idx:04d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            torch.save(prototypes.detach().cpu(), class_dir / "prototypes.pt")

        with (root / "manifest.json").open("w") as f:
            json.dump(
                {
                    "tau": float(self.tau),
                    "classes": sorted(self.prototypes.keys()),
                },
                f,
                indent=2,
            )

    def load(self, root):
        root = Path(root)
        with (root / "manifest.json").open() as f:
            manifest = json.load(f)

        self.tau = float(manifest["tau"])
        class_list = [int(class_idx) for class_idx in manifest["classes"]]

        self.prototypes = {
            int(class_idx): F.normalize(
                torch.load(path, map_location=self.device).float().to(self.device),
                dim=-1,
            )
            for class_idx in class_list
            for path in [root / f"class={class_idx:04d}" / "prototypes.pt"]
        }

    def __call__(self, features, class_idx, tau=None):
        features = F.normalize(features, dim=-1)
        prototypes = self.prototypes[int(class_idx)]
        sim = torch.einsum("...nd,kd->...nk", features, prototypes)
        return F.softmax(float(self.tau if tau is None else tau) * sim, dim=-1)

if __name__ == "__main__":
    train_features = torch.randn(32, 64)
    prototypes = ConceptPrototypes(K=4, tau=5, device=torch.device("cpu"))
    prototypes.fit(train_features)

    print(prototypes.prototypes.shape)

    features = torch.randn(2, 8, 64)
    sim = prototypes(features)
    print(sim.shape)
    print(sim)
