import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from model import DDColor
from arch_utils.discriminator_arch import DynamicUNetDiscriminator
from dataset import ImageDataset
from losses import L1Loss, PerceptualLoss, VanillaGANLoss, ColorfulnessLoss
from utils.img_utils import tensor_lab2rgb
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import json

# Config
BATCH_SIZE = 12
NUM_EPOCHS = 20
GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 1e-4


def train_one_epoch(
    generator, discriminator, dataloader,
    g_optimizer, d_optimizer,
    l1_criterion, perceptual_criterion,
    gan_criterion, colorfulness_criterion, scaler, device
):
    generator.train()
    discriminator.train()
    running_g_loss = 0.0
    running_g_rec_loss = 0.0
    running_g_percep_loss = 0.0
    running_g_gan_loss = 0.0
    running_g_colorfulness_loss = 0.0

    running_d_loss = 0.0

    loop = tqdm(dataloader, leave=True)

    for gray_rgb, gt_rgb, l, gt_ab in loop:
        gray_rgb = gray_rgb.to(device)
        gt_rgb = gt_rgb.to(device)
        l = l.to(device)
        gt_ab = gt_ab.to(device)

        # Optimize generator
        for param in discriminator.parameters():
            param.requires_grad = False

        g_optimizer.zero_grad()

        with autocast(device_type=device.type):
            output_ab = generator(gray_rgb)

        output_lab = torch.cat([l, output_ab], dim=1)
        output_rgb = tensor_lab2rgb(output_lab)

        with autocast(device_type=device.type):
            fake_g_pred = discriminator(output_rgb)

            rec_loss = l1_criterion(output_ab, gt_ab)
            percep_loss = perceptual_criterion(output_rgb, gt_rgb)
            gan_loss = gan_criterion(fake_g_pred, True, is_disc=False)
            colorfulness_loss = colorfulness_criterion(output_rgb)

            loss_g = rec_loss + percep_loss + gan_loss + colorfulness_loss

        scaler.scale(loss_g).backward()
        scaler.step(g_optimizer)
        scaler.update()


        # Optimize discriminator
        for param in discriminator.parameters():
            param.requires_grad = True

        d_optimizer.zero_grad()

        with autocast(device_type=device.type):
            real_d_pred = discriminator(gt_rgb)
            fake_d_pred = discriminator(output_rgb.detach())
            loss_d = gan_criterion(real_d_pred, True, is_disc=True) + gan_criterion(fake_d_pred, False, is_disc=True)

        scaler.scale(loss_d).backward()
        scaler.step(d_optimizer)
        scaler.update()

        running_g_loss += loss_g.item()
        running_g_rec_loss += rec_loss.item()
        running_g_percep_loss += percep_loss.item()
        running_g_gan_loss += gan_loss.item()
        running_g_colorfulness_loss += colorfulness_loss.item()
        running_d_loss += loss_d.item()

        loop.set_description(f"Total_G: {loss_g.item():.4f}, Rec: {rec_loss.item():.4f}, Percep: {percep_loss.item():.4f}, GAN: {gan_loss.item():.4f}, Color: {colorfulness_loss.item():.4f}, D: {loss_d.item():.4f}")

    epoch_loss = {
        'g_loss': running_g_loss / len(dataloader),
        'g_rec_loss': running_g_rec_loss / len(dataloader),
        'g_percep_loss': running_g_percep_loss / len(dataloader),
        'g_gan_loss': running_g_gan_loss / len(dataloader),
        'g_colorfulness_loss': running_g_colorfulness_loss / len(dataloader),
        'd_loss': running_d_loss / len(dataloader),
    }
    return epoch_loss


def calculate_val_loss(generator, dataloader, l1_criterion, perceptual_criterion, gan_criterion, colorfulness_criterion, device):
    generator.eval()
    running_g_loss = 0.0
    running_g_rec_loss = 0.0
    running_g_percep_loss = 0.0
    running_g_gan_loss = 0.0
    running_g_colorfulness_loss = 0.0

    loop = tqdm(dataloader, leave=True)

    for gray_rgb, gt_rgb, l, gt_ab in loop:
        gray_rgb = gray_rgb.to(device)
        gt_rgb = gt_rgb.to(device)
        l = l.to(device)
        gt_ab = gt_ab.to(device)

        with autocast(device_type=device.type):
            output_ab = generator(gray_rgb)

        output_lab = torch.cat([l, output_ab], dim=1)
        output_rgb = tensor_lab2rgb(output_lab)

        with autocast(device_type=device.type):
            fake_g_pred = discriminator(output_rgb)

            rec_loss = l1_criterion(output_ab, gt_ab)
            percep_loss = perceptual_criterion(output_rgb, gt_rgb)
            gan_loss = gan_criterion(fake_g_pred, True, is_disc=False)  
            colorfulness_loss = colorfulness_criterion(output_rgb)

            loss_g = rec_loss + percep_loss + gan_loss + colorfulness_loss

        running_g_loss += loss_g.item()
        running_g_rec_loss += rec_loss.item()
        running_g_percep_loss += percep_loss.item()
        running_g_gan_loss += gan_loss.item()
        running_g_colorfulness_loss += colorfulness_loss.item()

        loop.set_description(f"Total_G: {loss_g.item():.4f}, Rec: {rec_loss.item():.4f}, Percep: {percep_loss.item():.4f}, GAN: {gan_loss.item():.4f}, Color: {colorfulness_loss.item():.4f}")

    epoch_loss = {
        'g_loss': running_g_loss / len(dataloader),
        'g_rec_loss': running_g_rec_loss / len(dataloader),
        'g_percep_loss': running_g_percep_loss / len(dataloader),
        'g_gan_loss': running_g_gan_loss / len(dataloader),
        'g_colorfulness_loss': running_g_colorfulness_loss / len(dataloader),
    }
    return epoch_loss


def train(model, discriminator, train_loader, val_loader, g_optimizer, d_optimizer, l1_criterion, perceptual_criterion, gan_criterion, colorfulness_criterion, device, num_epochs=10, current_epoch=0):
    start_time = time.time()
    train_losses = []
    val_losses = []

    scaler = GradScaler()

    g_scheduler = StepLR(
        g_optimizer, step_size=10, gamma=0.5
    )
    d_scheduler = StepLR(
        d_optimizer, step_size=10, gamma=0.5
    )

    for epoch in range(current_epoch, num_epochs):
        train_loss = train_one_epoch(model, discriminator, train_loader, g_optimizer, d_optimizer, l1_criterion, perceptual_criterion, gan_criterion, colorfulness_criterion, scaler, device)
        train_losses.append(train_loss)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}')

        g_scheduler.step()
        d_scheduler.step()

        val_loss = calculate_val_loss(model, val_loader, l1_criterion, perceptual_criterion, gan_criterion, colorfulness_criterion, device)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}, Val Loss: {val_loss}')

        save_metrics(train_loss, val_loss, epoch+1)

        if (epoch+1) % 2 == 0:
            save_checkpoint(model, discriminator, g_optimizer, d_optimizer, epoch+1, save_dir="checkpoints")
    
    end_time = time.time()

    return train_losses, end_time - start_time

def save_checkpoint(model, discriminator, g_optimizer, d_optimizer, epoch, save_dir="checkpoints", prefix="ddcolor"):
    """
    Save a full training checkpoint.
    
    Args:
        model: Generator model
        discriminator: Discriminator model
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        epoch: Current epoch number
        save_dir: Directory to save checkpoints
        prefix: Filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "generator_state_dict": model.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optimizer_state_dict": g_optimizer.state_dict(),
        "d_optimizer_state_dict": d_optimizer.state_dict()
    }
    path = os.path.join(save_dir, f"{prefix}_epoch{epoch}.pth")
    torch.save(checkpoint, path)
    print(f"[INFO] Checkpoint saved at {path}")


def save_metrics(train_loss, val_loss, epoch, save_dir="training_metrics"):
    os.makedirs(save_dir, exist_ok=True)
    metrics = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    }
    path = os.path.join(save_dir, f"metrics_epoch{epoch}.json")
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrics saved at {path}")


if __name__ == '__main__':
    train_dataset = ImageDataset('train2017/')
    val_dataset = ImageDataset('val2017/')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    current_checkpoint = None
    current_epoch = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDColor(num_queries=100, num_scales=3, nf=512, num_output_channels=2).to(device)

    if current_checkpoint is not None:
        checkpoint = torch.load(current_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["generator_state_dict"])
    discriminator = DynamicUNetDiscriminator(n_channels=3, nf=64, n_blocks=3).to(device)


    l1_criterion = L1Loss(loss_weight=0.1).to(device)
    perceptual_criterion = PerceptualLoss(layer_weights={'conv1_1': 0.0625, 'conv2_1': 0.125, 'conv3_1': 0.25, 'conv4_1': 0.5, 'conv5_1': 1.0}, perceptual_weight=5.0).to(device)
    gan_criterion = VanillaGANLoss(real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0).to(device)
    colorfulness_criterion = ColorfulnessLoss(loss_weight=0.5).to(device)

    g_optimizer = torch.optim.AdamW(model.parameters(), lr=GENERATOR_LR, betas=(0.9, 0.99), weight_decay=0.01)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=(0.9, 0.99))

    train_losses, train_time = train(model, discriminator, train_loader, val_loader, g_optimizer, d_optimizer, l1_criterion, perceptual_criterion, gan_criterion, colorfulness_criterion, device, num_epochs=NUM_EPOCHS, current_epoch=current_epoch)
    print(f'Total training time: {train_time:.2f}s')

    # Visualize result
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


        

            
