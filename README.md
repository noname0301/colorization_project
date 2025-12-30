# DDColor Image Colorization Project

A deep learning project for automatic image colorization using a DDColor model with GAN-based training. The model converts grayscale images to colorized versions using a transformer-based architecture with ConvNeXt encoder and dual decoder.

## Project Overview

This project implements an image colorization system that:
- Uses a DDColor generator with ConvNeXt encoder and transformer-based decoder
- Employs a U-Net discriminator for adversarial training
- Trains on LAB color space for better color representation
- Includes multiple loss functions: L1, Perceptual, GAN, and Colorfulness losses
- Supports training resumption from checkpoints
- Provides comprehensive metrics evaluation (PSNR, SSIM, MS-SSIM, FID, Colorfulness)

## Project Structure

```
.
├── model.py                    # DDColor generator model
├── training_loop.py            # Training script with GAN training loop
├── dataset.py                  # Dataset loader for image data
├── losses.py                   # Loss functions (L1, Perceptual, GAN, Colorfulness)
├── inference.py                # Batch inference script for colorization
├── calculate_metrics.py        # Comprehensive metrics evaluation script
├── requirement.txt             # Python dependencies
├── arch_utils/                 # Architecture utilities
│   ├── convnext.py            # ConvNeXt encoder
│   ├── discriminator_arch.py  # U-Net discriminator
│   ├── transformer_utils.py   # Transformer components
│   ├── unet_utils.py          # U-Net utilities
│   ├── vgg_arch.py            # VGG for perceptual loss
│   └── position_encoding.py   # Positional encoding
├── metrics/                    # Evaluation metrics
│   ├── psnr_ssim.py           # PSNR and SSIM metrics
│   ├── fid.py                 # Fréchet Inception Distance
│   └── colorfulness.py        # Colorfulness metric
├── utils/                      # Utility functions
│   └── img_utils.py           # Image processing utilities
├── train2017/                  # Training dataset directory (configurable)
├── val2017/                    # Validation dataset directory (configurable)
├── val_input_test/             # Test input images (configurable)
├── val_output_test/            # Test output images (configurable)
├── checkpoints/                # Saved model checkpoints
└── training_metrics/           # Training metrics logs
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- PyTorch 2.9.1 with CUDA 13.0 support

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "DL Project"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows**: `venv\Scripts\activate`
- **Linux/Mac**: `source venv/bin/activate`

### 3. Install Dependencies

Install PyTorch with CUDA support first:

```bash
pip install torch==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Then install other dependencies:

```bash
pip install -r requirement.txt
```

**Note**: The `requirement.txt` includes PyTorch with CUDA 13.0. If you have a different CUDA version or want CPU-only, modify the PyTorch installation accordingly.

### 4. Dataset Configuration

**Directory Setup**:

Configure dataset paths in the respective scripts:

**For Training** (in `training_loop.py`):
- `TRAIN_DIR`: Training dataset directory (default: `"train2017/"`)
- `VAL_DIR`: Validation dataset directory (default: `"val2017/"`)

**For Inference** (in `inference.py`):
- `INPUT_DIR`: Input images directory
- `OUTPUT_DIR`: Output images directory

**For Metrics** (in `calculate_metrics.py`):
- `input_dir`: Ground truth images directory
- `output_dir`: Generated images directory

**Dataset Requirements**:
- Image format: JPEG (recommended)
- The dataset loader automatically:
  - Resizes images to 256x256
  - Converts to LAB color space
  - Extracts L channel for input and AB channels for ground truth

## Usage

### Training

To train the model from scratch:

```bash
python training_loop.py
```

**Training Configuration** (edit in `training_loop.py`):
- `BATCH_SIZE`: 12 (default)
- `NUM_EPOCHS`: 20 (default)
- `GENERATOR_LR`: 1e-4
- `DISCRIMINATOR_LR`: 1e-4
- `TRAIN_DIR`: "train2017/" (training dataset directory)
- `VAL_DIR`: "val2017/" (validation dataset directory)

**Resume Training from Checkpoint**:

Edit the following lines in `training_loop.py`:

```python
last_checkpoint = "checkpoints/ddcolor_epoch10.pth"  # Path to checkpoint
last_epoch = 10  # Epoch number to resume from
```

**Advanced Configuration** (in `training_loop.py`):

Loss weights:
- L1 Loss: 0.1
- Perceptual Loss: 5.0 (with layer weights for VGG features)
- GAN Loss: 1.0
- Colorfulness Loss: 0.5

Optimizers:
- Generator: AdamW (lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01)
- Discriminator: Adam (lr=1e-4, betas=(0.9, 0.99))

The training script will:
- Save checkpoints every 2 epochs to `checkpoints/`
- Save training metrics to `training_metrics/`
- Use learning rate scheduler (StepLR with step_size=10, gamma=0.5)
- Display progress with tqdm progress bars

### Inference

To colorize a batch of images:

```bash
python inference.py
```

**Configuration** (edit in `inference.py`):
- `INPUT_DIR`: Input directory containing grayscale images (default: `"val_input_test/"`)
- `OUTPUT_DIR`: Output directory for colorized images (default: `"val_output_test/"`)
- `device`: Set to "cuda" or "cpu" (default: "cpu")
- Checkpoint path: Default is `"checkpoints/ddcolor_epoch20.pth"`

The script will:
- Load the trained model from the checkpoint
- Process all images in the input directory
- Save colorized results to the output directory
- Automatically convert images to LAB color space and back to RGB

### Evaluating Metrics

To calculate comprehensive metrics on colorized images:

```bash
python calculate_metrics.py
```

**Configuration** (edit in `calculate_metrics.py`):
- `input_dir`: Directory with original/ground truth images
- `output_dir`: Directory with colorized images

The script will calculate and display:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MS-SSIM**: Multi-Scale Structural Similarity Index
- **Colorfulness**: Average colorfulness score of generated images
- **FID**: Fréchet Inception Distance between real and generated images

**Example Output**:
```
PSNR: 25.34
SSIM: 0.87
MS-SSIM: 0.92
Colorfulness: 45.67
FID: 12.34
```

## Model Architecture

- **Generator**: DDColor with ConvNeXt-L encoder, transformer-based dual decoder
  - Input: 3-channel grayscale image (L channel repeated)
  - Output: 2-channel AB color prediction
  - Parameters: ~60M (configurable)

- **Discriminator**: Dynamic U-Net discriminator
  - Input: 3-channel RGB image
  - Architecture: 3-block U-Net with 64 base features

## Loss Functions

The training uses a combination of losses:
1. **L1 Loss**: Pixel-wise reconstruction loss
2. **Perceptual Loss**: VGG-based perceptual similarity
3. **GAN Loss**: Adversarial loss for realistic colorization
4. **Colorfulness Loss**: Encourages vibrant colors

## Evaluation Metrics

The project includes comprehensive evaluation metrics accessible via `calculate_metrics.py`:

- **PSNR** (Peak Signal-to-Noise Ratio): Measures pixel-level reconstruction quality
- **SSIM** (Structural Similarity Index): Evaluates structural similarity between images
- **MS-SSIM** (Multi-Scale SSIM): Multi-scale version of SSIM for better perceptual quality assessment
- **FID** (Fréchet Inception Distance): Measures distribution similarity using Inception-v3 features
- **Colorfulness**: Quantifies color vibrancy and saturation of generated images

All metrics are implemented in the `metrics/` directory and can be imported individually.

## Checkpoints

Checkpoints are saved with the following information:
- Generator state dict
- Discriminator state dict
- Generator optimizer state dict
- Discriminator optimizer state dict
- Epoch number

## Workflow Example

Here's a complete workflow from training to evaluation:

1. **Prepare your dataset**:
   ```bash
   # Place images in train2017/ and val2017/
   ```

2. **Train the model**:
   ```bash
   python training_loop.py
   ```

3. **Run inference on test images**:
   ```bash
   python inference.py
   ```

4. **Evaluate the results**:
   ```bash
   python calculate_metrics.py
   ```

## Troubleshooting

### CUDA Out of Memory
- Reduce `BATCH_SIZE` in `training_loop.py` (try 8 or 4)
- Reduce image resolution in `dataset.py`
- Use gradient accumulation for effective larger batch sizes

### Missing Dependencies
- Ensure all packages in `requirement.txt` are installed
- Check PyTorch CUDA compatibility with your GPU
- For FID calculation, ensure `scipy` and `torchvision` are properly installed

### Dataset Issues
- Ensure images are in JPEG format
- Check that input and output directories exist and contain images
- Verify image paths in scripts match your directory structure

### Inference Issues
- Ensure checkpoint file exists at the specified path
- Check that input images are readable by OpenCV
- Verify output directory has write permissions

## Acknowledgments

This project is based on the DDColor architecture for image colorization.

