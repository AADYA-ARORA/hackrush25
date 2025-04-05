import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from train_levit import LeViTNPR, Normalize6Channels, compute_npr
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image)
    npr = compute_npr(tensor)
    image_6ch = torch.cat([tensor, npr], dim=0)
    normalizer = Normalize6Channels(mean=[0.485]*6, std=[0.229]*6)
    image_6ch = normalizer(image_6ch)
    return image_6ch.unsqueeze(0)

@torch.no_grad()
def extract_embeddings(model, image_dir):
    model.eval()
    embeddings = []
    labels = []

    for label in ['fake', 'real']:
        label_path = os.path.join(image_dir, label)
        for fname in tqdm(os.listdir(label_path), desc=f"Extracting from {label}"):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(label_path, fname)
            image_tensor = load_image_tensor(img_path).to(device)

            with torch.no_grad():
                features = model.model.forward_features(image_tensor)
                pooled = features.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(pooled)
            labels.append(label)

    return np.array(embeddings), np.array(labels)

def visualize_boundary(embeddings, labels):
    # Reduce to 2D
    reducer = PCA(n_components=2)  # Try TSNE(n_components=2) for nonlinear
    reduced = reducer.fit_transform(embeddings)

    # Train classifier on 2D
    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = LogisticRegression().fit(reduced, y)

    # Create meshgrid
    x_min, x_max = reduced[:, 0].min() - 1, reduced[:, 0].max() + 1
    y_min, y_max = reduced[:, 1].min() - 1, reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    # Plot decision boundary and points
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    for label in np.unique(labels):
        plt.scatter(reduced[labels == label, 0],
                    reduced[labels == label, 1],
                    label=label, s=40)
    plt.legend()
    plt.title("Decision Boundary on Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.savefig("decision_boundary_plot.png")  # ✅ Save the figure
    print("✅ Saved decision boundary plot as 'decision_boundary_plot.png'")
    plt.show()

def main(model_path, model_name, val_dir):
    model = LeViTNPR(model_name=model_name, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    embeddings, labels = extract_embeddings(model, val_dir)
    visualize_boundary(embeddings, labels)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='levit_192')
    parser.add_argument('--val_dir', type=str, required=True)  # Should contain `real/` and `fake/`
    args = parser.parse_args()

    main(args.model_path, args.model_name, args.val_dir)
