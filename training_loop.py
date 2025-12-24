import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from model import DDColor
from arch_utils.discriminator_arch import PatchDiscriminator
from dataset import ImageDataset
from losses import L1Loss, PerceptualLoss, GANLoss, ColorfulnessLoss
import kornia.color as K
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


# Config
BATCH_SIZE = 12
NUM_EPOCHS = 50
LR = 1e-4


def train_one_epoch(
    model, discriminator, dataloader,
    g_optimizer, d_optimizer,
    l1_criterion, perceptual_criterion,
    gan_criterion, colorfulness_criterion,
    scaler, device
):
    model.train()
    discriminator.train()
    running_loss = 0.0

    loop = tqdm(dataloader, leave=True)

    for L, gt in loop:
        L = L.to(device)
        gt = gt.to(device)

        # Generator forward pass
        with autocast(device_type=device.type):
            pred = model(L)                       # (B,2,H,W) a/b channels
            pred = pred.clamp(-128, 127)
            L_single = L[:, 0:1, :, :]           # (B,1,H,W)

            # Prepare Lab images for discriminator / perceptual loss
            gt_lab = torch.cat([L_single, gt], dim=1) 
            out_lab = torch.cat([L_single, pred], dim=1)

        gt_rgb = K.lab_to_rgb(gt_lab)
        out_rgb = K.lab_to_rgb(out_lab)


        # Update Discriminator
        d_optimizer.zero_grad()
        with autocast(device_type=device.type):
            d_real = discriminator(gt_lab)
            d_fake = discriminator(out_lab.detach())  # detach to avoid G gradients
            D_loss = 0.5 * (gan_criterion(d_real, True, is_disc=True) + gan_criterion(d_fake, False, is_disc=True))

        scaler.scale(D_loss).backward()
        scaler.unscale_(d_optimizer)
        nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
        scaler.step(d_optimizer)
        scaler.update()

        # Update Generator
        g_optimizer.zero_grad()
        with autocast(device_type=device.type):
            # Re-run discriminator on fake images (or reuse d_fake)
            d_fake_for_G = discriminator(out_lab)
            G_adv_loss = gan_criterion(d_fake_for_G, True, is_disc=False)

            # Reconstruction + perceptual + colorfulness losses
            G_rec_loss = l1_criterion(pred, gt)
            G_percep_loss = perceptual_criterion(out_rgb, gt_rgb)
            G_color_loss = colorfulness_criterion(out_rgb)

            G_loss = G_rec_loss + G_percep_loss + G_adv_loss + G_color_loss

        scaler.scale(G_loss).backward()
        scaler.unscale_(g_optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(g_optimizer)
        scaler.update()

        running_loss += G_loss.item() * L.size(0)

        loop.set_description(f"Loss: {G_loss.item():.4f}")


    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def train(model, discriminator, dataloader, g_optimizer, d_optimizer, l1_criterion, perceptual_criterion, gan_criterion, colorfulness_criterion, device, num_epochs=10):
    start_time = time.time()
    train_losses = []

    scaler = GradScaler(device=device.type)  
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, discriminator, dataloader, g_optimizer, d_optimizer, l1_criterion, perceptual_criterion, gan_criterion, colorfulness_criterion, scaler, device)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}')

        if (epoch+1) % 10 == 0:
            save_checkpoint(model, discriminator, g_optimizer, d_optimizer, epoch+1)
    
    end_time = time.time()
    torch.save(model, "models/ddcolor.pt")

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


if __name__ == '__main__':
    dataset = ImageDataset('train2017/')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDColor("convnext-t", num_queries=100, num_scales=3, nf=512, num_channels=2).to(device)
    discriminator = PatchDiscriminator(in_channels=3, nf=64, n_blocks=3).to(device)


    l1_criterion = L1Loss(loss_weight=0.1).to(device)
    perceptual_criterion = PerceptualLoss(loss_weight=5.0).to(device)
    gan_criterion = GANLoss(loss_weight=1.0, real_label_val=1.0, fake_label_val=0.0).to(device)
    colorfulness_criterion = ColorfulnessLoss(loss_weight=0.5).to(device)

    g_optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.01)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.9, 0.99))

    train_losses, train_time = train(model, discriminator, dataloader, g_optimizer, d_optimizer, l1_criterion, perceptual_criterion, gan_criterion, colorfulness_criterion, device, num_epochs=NUM_EPOCHS)
    print(f'Total training time: {train_time:.2f}s')

    # Visualize result
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


        

            
