# main.py - Main entry point for the project

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TimeSformerBoundary, FeatureBoundaryDetector
from dataset import YouCook2Dataset, YouCook2FeatureDataset
from trainer import train_model, validate_model
from inference import predict_boundaries
from utils import setup_logger, set_seed

def main():
    parser = argparse.ArgumentParser(description='YouCook2 Video Segmentation')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Train or test mode')
    parser.add_argument('--data_root', type=str, default='/home/leiy4/video-segmentation/data', help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')  # Increased batch size since features are smaller
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--clip_length', type=int, default=16, help='Clip length in frames')
    parser.add_argument('--stride', type=int, default=8, help='Stride for sliding window')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint for testing')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--boundary_threshold', type=float, default=0.5, help='Threshold for boundary detection')
    parser.add_argument('--limit_samples', type=int, default=None, help='Limit number of samples per epoch (for debugging)')
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = setup_logger('youcook2_segmentation')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create datasets using precomputed features
    if args.mode == 'train':
        train_dataset = YouCook2FeatureDataset(
            split='train',
            data_root=args.data_root,
            clip_length=args.clip_length,
            stride=args.stride
        )
        
        # Optionally limit dataset size for faster iterations
        if args.limit_samples:
            logger.info(f"Limiting training dataset to {args.limit_samples} samples")
            train_indices = torch.randperm(len(train_dataset))[:args.limit_samples]
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        val_dataset = YouCook2FeatureDataset(
            split='val',
            data_root=args.data_root,
            clip_length=args.clip_length,
            stride=args.stride
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Initialize model for precomputed features
        # Assuming ResNet-34 features have 512 dimensions
        model = FeatureBoundaryDetector(
            feature_dim=512,  # ResNet-34 feature dimension
            hidden_dim=256,
            num_frames=args.clip_length
        )
        
        # Train model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            output_dir=args.output_dir,
            logger=logger
        )
    
    elif args.mode == 'test':
        test_dataset = YouCook2FeatureDataset(
            split='test',
            data_root=args.data_root,
            clip_length=args.clip_length,
            stride=args.stride,
            is_test=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        # Initialize model
        model = FeatureBoundaryDetector(
            feature_dim=512,  # ResNet-34 feature dimension
            hidden_dim=256,
            num_frames=args.clip_length
        )
        
        # Load checkpoint
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {args.checkpoint}")
        else:
            logger.error("No checkpoint provided for testing")
            return
        
        # Run inference - this function will need to be adapted as well
        predict_boundaries(
            model=model,
            test_loader=test_loader,
            test_dataset=test_dataset,
            boundary_threshold=args.boundary_threshold,
            output_dir=args.output_dir,
            logger=logger
        )

if __name__ == "__main__":
    main()


# Example usage:
# python main.py --mode train --data_root /home/leiy4/video-segmentation/data --batch_size 4 --num_epochs 30