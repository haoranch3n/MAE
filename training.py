import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from model import *
from utils import setup_seed


def list_files(dataset_path):
    print("Listing files in:", dataset_path)
    images = []
    for root, _, files in sorted(os.walk(dataset_path)):
        for name in sorted(files):
            if name.lower().endswith('.tif'):
                images.append(os.path.join(root, name))
    print(f"Found {len(images)} .tif files.")
    return images


class CustomImageDataset(Dataset):
    """The above class is a custom dataset class for images in PyTorch."""
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = list_files(self.img_dir)
        self.transform =  transforms.Compose([
                            transforms.Resize(252),
                            transforms.CenterCrop(252),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
                        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--fine_tuning_learning_rate', type=float, default=1.5e-5)  # Lower learning rate for fine-tuning
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=50)  # Shorter warm-up period for fine-tuning
    parser.add_argument('--data_path', type=str, default='/data/fundus')
    parser.add_argument('--output_model_path', type=str, default='/model/fundus-vit-t-mae.pt')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to the pre-trained model')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # Placeholder for loading new data
    # Replace the following lines with code to load your new dataset
    # train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataset = CustomImageDataset(args.data_path)

    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)

    writer = SummaryWriter(os.path.join('logs', 'new_data', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the pre-trained model if a path is provided
    if args.pretrained_model_path:
        model = torch.load(args.pretrained_model_path, map_location=device)
        learning_rate = args.fine_tuning_learning_rate  # Use fine-tuning learning rate
    else:
        model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
        learning_rate = args.base_learning_rate  # Use base learning rate for training from scratch

    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        print(f'In epoch {e}, average training loss is {avg_loss}.')

        ''' save model '''
        torch.save(model, args.model_path)
