import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images.

    Parameters:
    - img1, img2: torch.Tensor of shape (B, C, H, W), values in [0, 1] or [0, max_val]
    - max_val: Maximum possible pixel value. Default 1.0 for normalized images

    Returns:
    - psnr: float, average PSNR over the batch
    """
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
    return psnr.item()

def calculate_ssim(img1, img2):
    # Single-scale SSIM
    return ssim(img1, img2, data_range=1.0, size_average=True).item()

def calculate_ms_ssim(img1, img2):
    # Multi-scale SSIM
    return ms_ssim(img1, img2, data_range=1.0, size_average=True).item()


if __name__ == '__main__':
    # example usage
    img1 = torch.rand(10, 3, 256, 256)  # real images
    img2 = torch.rand(10, 3, 256, 256)  # generated images

    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    ms_ssim = calculate_ms_ssim(img1, img2)
    print("PSNR:", psnr)
    print("SSIM:", ssim)
    print("MS-SSIM:", ms_ssim)