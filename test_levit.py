import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import csv
import numpy as np
from train_levit import LeViTNPR  # Make sure this import is correct


class Normalize6Channels:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def __call__(self, tensor):
        if tensor.size(0) != 6:
            raise ValueError(f"Expected tensor with 6 channels, got {tensor.size(0)}")
        return (tensor - self.mean) / self.std


def compute_npr(image_tensor):
    image_np = image_tensor.numpy()
    h_freq = image_np - np.roll(image_np, 1, axis=1)
    v_freq = image_np - np.roll(image_np, 1, axis=2)
    freq_tensor = np.sqrt(h_freq ** 2 + v_freq ** 2)
    return torch.tensor(freq_tensor, dtype=torch.float32)


class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
        self.normalizer = Normalize6Channels(mean=[0.485] * 6, std=[0.229] * 6)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)

        npr = compute_npr(image)
        image_6ch = torch.cat([image, npr], dim=0)
        image_6ch = self.normalizer(image_6ch)

        filename = os.path.basename(img_path)
        return image_6ch, filename  # return full filename with extension


def parse_args():
    parser = argparse.ArgumentParser(description="LeViT-NPR Model Tester")
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Directory containing test images")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Path to model .pth checkpoint")
    parser.add_argument('--output_csv', type=str, required=True,
                        help="Where to save predictions CSV")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--model_name', type=str, default='levit_192')
    return parser.parse_args()


def test_model(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(args.image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Test dataset loaded.")

    model = LeViTNPR(model_name=args.model_name, pretrained=False, num_classes=1)
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint_path}")

    predictions = []

    with torch.no_grad():
        for inputs, file_names in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.75).long().cpu().numpy()

            for fname, pred in zip(file_names, preds):
                label = "0" if pred == 1 else "1"
                predictions.append((fname, label))

    # Sort by filename
    predictions.sort(key=lambda x: x[0])

    # Write to CSV
    with open(args.output_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "class"])
        for fname, label in predictions:
            writer.writerow([fname, label])

    print(f"Predictions saved to {args.output_csv}")


def main():
    args = parse_args()
    test_model(args)


if __name__ == "__main__":
    main()