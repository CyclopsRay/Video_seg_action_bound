# YouCook2 Video Segmentation Pipeline

## Overview

This project implements a deep learning pipeline for automatic video segmentation on the YouCook2 dataset. The system detects segment boundaries in cooking videos, which can be used for tasks like action recognition, video summarization, and recipe step extraction.

## Project Structure

The project is organized as follows:

```
.
├── data/
│   ├── annotations/               # Contains JSON annotation files
│   ├── raw_videos/                # Downloaded YouTube videos
│   └── splits/                    # Train/val/test splits
├── download_youcookii_videos.py   # Script for downloading dataset videos
├── dataset.py                     # Dataset handling and processing
├── model.py                       # TimeSformer model for boundary detection
├── trainer.py                     # Training and validation functions
├── main.py                        # Main entry point and pipeline orchestration
└── utils.py                       # Utility functions
```

## Pipeline Overview

The pipeline consists of the following key components:

1. **Data Collection**: Download videos from YouTube using the `download_youcookii_videos.py` script
2. **Data Processing**: Process videos and annotations with the `YouCook2Dataset` class in `dataset.py`
3. **Model Training**: Train a TimeSformer-based model using the training pipeline in `main.py`
4. **Inference**: Perform boundary prediction on new videos using the trained model

## Main Execution Flow

The pipeline starts with `main.py`, which orchestrates the entire process:

```python
# Example usage:
# python main.py --mode train --data_root /path/to/data --batch_size 4 --num_epochs 30
```

### Command Line Arguments

- `--mode`: `train` or `test` mode
- `--data_root`: Data root directory
- `--batch_size`: Batch size for training and inference
- `--num_epochs`: Number of training epochs
- `--lr`: Learning rate
- `--clip_length`: Number of frames in each video clip
- `--stride`: Stride for sliding window over videos
- `--fps`: Frames per second for video processing
- `--seed`: Random seed for reproducibility
- `--checkpoint`: Path to model checkpoint for testing
- `--output_dir`: Directory for saving results
- `--boundary_threshold`: Threshold for boundary detection
- `--limit_samples`: Limit samples per epoch (for debugging)

## Data Pipeline

### Video Download

The `download_youcookii_videos.py` script:
- Downloads videos from YouTube using the yt-dlp tool
- Organizes videos into appropriate folders based on split (train/val/test)
- Implements retry logic for robust downloading
- Tracks missing videos that couldn't be downloaded

### Dataset Processing

The `YouCook2Dataset` class in `dataset.py`:
- Loads video clips and annotations
- Processes videos into fixed-length clips using a sliding window approach
- Labels clips that contain segment boundaries
- Provides data loaders for training and inference

## Model Architecture

The `TimeSformerBoundary` class in `model.py` implements a modified TimeSformer architecture:
- Uses a pre-trained Vision Transformer (ViT) as the backbone (frozen during training)
- Adds temporal attention modules to model time dependencies across frames
- Implements a boundary detection head that outputs boundary predictions
- Uses clip-level classification for final predictions

Key components:
- **Patch Embedding**: Converts video frames into patches
- **Visual Backbone**: Pre-trained ViT processes spatial features
- **Temporal Attention**: Custom modules for modeling temporal relations
- **Boundary Head**: Final classification layers for boundary prediction

## Training Pipeline

The training process in `trainer.py`:
- Implements a weighted binary cross-entropy loss to handle class imbalance
- Uses different learning rates for different components
- Implements warmup and cosine annealing learning rate scheduling
- Tracks metrics like loss and F1 score
- Saves checkpoints for best models

## Inference and Evaluation

The inference process:
- Loads a trained model from a checkpoint
- Processes videos in sliding windows
- Applies the model to predict boundary probabilities
- Post-processes predictions to obtain final segment boundaries
- Evaluates results using metrics like precision, recall, F1, and IoU

## Utilities

The `utils.py` module provides various utility functions:
- Setting random seeds for reproducibility
- Logging setup
- Video information extraction
- Evaluation metrics calculation
- Visualization tools for model predictions
- Temporal smoothing of boundary predictions
- Result saving and export

## Getting Started

1. First, set up the dataset:
   ```
   python download_youcookii_videos.py
   ```

2. Train the model:
   ```
   python main.py --mode train --data_root /path/to/data --batch_size 4 --num_epochs 30
   ```

3. Test the model:
   ```
   python main.py --mode test --data_root /path/to/data --checkpoint /path/to/checkpoint.pth
   ```

## Requirements

- PyTorch
- timm (for Vision Transformer models)
- OpenCV
- Pandas
- NumPy
- tqdm
- scikit-learn

## Notes

- The code uses a sliding window approach to process videos, which allows handling videos of any length.
- Class imbalance is addressed through weighted loss functions.
- The model freezes the visual backbone to speed up training and prevent overfitting.
- Temporal attention modules are the key components for capturing segment boundaries.