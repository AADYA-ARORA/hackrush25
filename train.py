
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
from tqdm import tqdm
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True

class SixChannelNormalizer:
    def __init__(self, mean_vals, std_vals):
        self.mean = torch.tensor(mean_vals).view(6, 1, 1)
        self.std = torch.tensor(std_vals).view(6, 1, 1)

    def __call__(self, img_tensor):
        return (img_tensor - self.mean) / self.std

def extract_npr_features(img_tensor):
    img_np = img_tensor.numpy()
    h_gradient = img_np - np.roll(img_np, 1, axis=1)
    v_gradient = img_np - np.roll(img_np, 1, axis=2)
    grad_magnitude = np.sqrt(h_gradient**2 + v_gradient**2)
    return torch.tensor(grad_magnitude, dtype=torch.float32)

class FaceDataset(Dataset):
    def __init__(self, directory, transform_ops=None):
        self.directory = directory
        self.transform_ops = transform_ops
        self.image_paths = []
        for label, category in enumerate(['real', 'fake']):
            category_path = os.path.join(directory, category)
            for file in os.listdir(category_path):
                if file.endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append((os.path.join(category_path, file), label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        filepath, label = self.image_paths[index]
        img = Image.open(filepath).convert('RGB')
        if self.transform_ops:
            img = self.transform_ops(img)
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        npr = extract_npr_features(img)
        combined = torch.cat([img, npr], dim=0)
        normalizer = SixChannelNormalizer(mean_vals=[0.485]*6, std_vals=[0.229]*6)
        normalized_img = normalizer(combined)
        return normalized_img, torch.tensor(label, dtype=torch.float32)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), targets, reduction='none')
        prob = torch.exp(-bce_loss)
        loss = self.alpha * (1 - prob) ** self.gamma * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class LeViTWithNPR(nn.Module):
    def __init__(self, model_id='levit_192', load_pretrained=True, output_dim=1):
        super().__init__()
        self.backbone = timm.create_model(model_id, pretrained=load_pretrained, num_classes=0)

        for name, mod in self.backbone.named_modules():
            if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
                old_weights = mod.weight.data
                new_input_conv = nn.Conv2d(
                    in_channels=6,
                    out_channels=mod.out_channels,
                    kernel_size=mod.kernel_size,
                    stride=mod.stride,
                    padding=mod.padding,
                    bias=(mod.bias is not None)
                )
                with torch.no_grad():
                    new_input_conv.weight[:, :3] = old_weights
                    new_input_conv.weight[:, 3:] = old_weights.clone()
                    if mod.bias is not None:
                        new_input_conv.bias.copy_(mod.bias)
                parent = self._locate_parent_module(self.backbone, name)
                setattr(parent, name.split('.')[-1], new_input_conv)
                break

        feature_dim = self.backbone.num_features
        self.classifier = nn.Linear(feature_dim, output_dim)

    def _locate_parent_module(self, net, module_name):
        parts = module_name.split('.')
        for part in parts[:-1]:
            net = getattr(net, part)
        return net

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled = features.mean(dim=1)
        return self.classifier(pooled)

class ModelTrainer:
    def __init__(self, network, device, train_data, val_data, optimizer, loss_fn, scheduler, cfg):
        self.network = network
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.cfg = cfg

    def run_training(self):
        best_accuracy = 0.0
        for epoch in range(self.cfg.epochs):
            self.network.train()
            epoch_loss = 0.0
            correct_preds, total_preds = 0, 0
            progress = tqdm(self.train_data, desc=f"Epoch {epoch+1}/{self.cfg.epochs}")
            for batch_imgs, batch_labels in progress:
                batch_imgs, batch_labels = batch_imgs.to(self.device), batch_labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.network(batch_imgs).squeeze()
                loss = self.loss_fn(logits, batch_labels)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                predictions = torch.sigmoid(logits) > 0.5
                correct_preds += (predictions == batch_labels.bool()).sum().item()
                total_preds += batch_labels.size(0)

                progress.set_postfix({
                    'Loss': epoch_loss / (total_preds / self.cfg.batch_size),
                    'Accuracy': 100 * correct_preds / total_preds
                })

            self.scheduler.step()
            val_accuracy = self.evaluate()
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self._save_model()

    def evaluate(self):
        self.network.eval()
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            for val_imgs, val_labels in self.val_data:
                val_imgs, val_labels = val_imgs.to(self.device), val_labels.to(self.device)
                logits = self.network(val_imgs).squeeze()
                preds = torch.sigmoid(logits) > 0.5
                total_correct += (preds == val_labels.bool()).sum().item()
                total_samples += val_labels.size(0)
        accuracy = 100 * total_correct / total_samples
        print(f"Validation Accuracy: {accuracy:.2f}%")
        return accuracy

    def _save_model(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        model_path = os.path.join(self.cfg.save_dir, 'best_model.pth')
        torch.save(self.network.state_dict(), model_path)
        print(f"Best model saved at {model_path}")

    def load_model(self, model_path):
        self.network.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--arch', type=str, default='levit_128')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--gpus', type=str, default='0')
    return parser.parse_args()

def launch():
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_set = FaceDataset(directory=os.path.join(args.data_root, 'train'), transform_ops=preprocess)
    val_set = FaceDataset(directory=os.path.join(args.data_root, 'val'), transform_ops=preprocess)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)

    net = LeViTWithNPR(model_id=args.arch, load_pretrained=args.use_pretrained, output_dim=1).to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    loss_function = BinaryFocalLoss(alpha=0.75, gamma=2, reduction='mean')
    optim_func = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_sched = optim.lr_scheduler.StepLR(optim_func, step_size=10, gamma=0.1)

    engine = ModelTrainer(network=net, device=device, train_data=train_loader, val_data=val_loader,
                          optimizer=optim_func, loss_fn=loss_function, scheduler=lr_sched, cfg=args)

    if args.resume_path:
        engine.load_model(args.resume_path)

    engine.run_training()

if __name__ == '__main__':
    launch()