# trainer.py - Training and validation functions

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

def weighted_binary_cross_entropy(predictions, targets, pos_weight=10.0):
    """Binary cross entropy with higher weight for positive samples (boundaries)"""
    # Use built-in BCEWithLogitsLoss for better numerical stability
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=predictions.device))
    
    # If predictions have been passed through sigmoid already, we need to use BCELoss
    if predictions.min() >= 0 and predictions.max() <= 1:
        loss_fn = nn.BCELoss(reduction='none')
        loss = loss_fn(predictions, targets)
        weights = torch.ones_like(targets)
        weights[targets == 1] = pos_weight
        return (weights * loss).mean()
    
    # Otherwise use BCEWithLogitsLoss
    return loss_fn(predictions, targets)

def train_model(model, train_loader, val_loader, num_epochs, lr, output_dir, logger):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create parameter groups for different learning rates
    temporal_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen parameters
        if 'temporal_attn' in name or 'time_embed' in name:
            temporal_params.append(param)
        else:
            classifier_params.append(param)
    
    logger.info(f"Training {len(temporal_params)} temporal parameters and {len(classifier_params)} classifier parameters")
    
    # Use different learning rates
    optimizer = optim.AdamW([
        {'params': temporal_params, 'lr': lr},  # Temporal components need full learning
        {'params': classifier_params, 'lr': lr * 1.5}  # Higher learning rate for simpler components
    ], weight_decay=0.01)
    
    # Add warmup and cosine annealing
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[lr, lr * 1.5],
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1  # 10% warmup
    )
    
    # Calculate class weights to handle imbalance
    pos_samples = 0
    total_samples = 0
    logger.info("Calculating class balance...")
    
    # Sample a small subset for class balance calculation
    for batch in train_loader:
        labels = batch['labels']
        pos_samples += torch.sum(labels).item()
        total_samples += labels.numel()
        
        # Sample at most 1000 batches
        if total_samples > 1000 * train_loader.batch_size:
            break
    
    neg_samples = total_samples - pos_samples
    pos_weight = neg_samples / max(pos_samples, 1)  # Avoid division by zero
    logger.info(f"Class balance - Positive: {pos_samples}, Negative: {neg_samples}, Weight: {pos_weight:.2f}")
    
    # Track metrics
    best_loss = float('inf')
    best_f1 = 0.0
    
    # Define criterion here
    criterion = weighted_binary_cross_entropy
    
    # Training loop
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_f1 = 0.0
        
        # Track predictions for F1 calculation
        all_preds = []
        all_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in progress_bar:
            # Use 'features' key instead of 'frames'
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            
            # Ensure outputs have the same shape as labels
            if outputs.shape != labels.shape:
                outputs = outputs.view_as(labels)
            
            # Compute loss using the calculated pos_weight
            loss = criterion(outputs, labels, pos_weight=pos_weight)
            
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Step the scheduler
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Calculate F1 score
            predictions = (outputs > 0.5).float()
            all_preds.append(predictions.detach().cpu())
            all_targets.append(labels.detach().cpu())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item()
            })
        
        # Calculate epoch F1 score on all predictions
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        train_f1 = calculate_f1(all_preds, all_targets)
        
        # Average loss
        train_loss /= len(train_loader)
        
        # Validation
        val_loss, val_f1 = validate_model(model, val_loader, criterion, device, pos_weight)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
                   f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_loss = val_loss
            
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f"best_model_f1_{val_f1:.4f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'f1': val_f1
            }, checkpoint_path)
            
            logger.info(f"Saved best model with F1: {val_f1:.4f} to {checkpoint_path}")
        
        # Save last model
        checkpoint_path = os.path.join(output_dir, "last_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'f1': val_f1
        }, checkpoint_path)
    
    return model

def validate_model(model, val_loader, criterion, device, pos_weight=10.0):
    """Validate the model"""
    model.eval()
    val_loss = 0.0
    
    # Collect all predictions for F1 calculation
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            # Use 'features' key instead of 'frames'
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Ensure outputs have the same shape as labels
            if outputs.shape != labels.shape:
                outputs = outputs.view_as(labels)
            
            # Compute loss
            loss = criterion(outputs, labels, pos_weight=pos_weight)
            
            # Update metrics
            val_loss += loss.item()
            
            # Collect predictions
            predictions = (outputs > 0.5).float()
            all_preds.append(predictions.cpu())
            all_targets.append(labels.cpu())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item()
            })
    
    # Calculate epoch F1 score
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    val_f1 = calculate_f1(all_preds, all_targets)
    
    # Average loss
    val_loss /= len(val_loader)
    
    return val_loss, val_f1

def calculate_f1(predictions, targets, epsilon=1e-7):
    """Calculate F1 score for boundary detection"""
    # True Positives, False Positives, False Negatives
    tp = torch.sum(predictions * targets).item()
    fp = torch.sum(predictions * (1 - targets)).item()
    fn = torch.sum((1 - predictions) * targets).item()
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # Print detailed stats
    if tp + fp + fn > 0:
        print(f"Predictions: {predictions.sum().item()}/{predictions.numel()}, "
              f"TP: {tp}, FP: {fp}, FN: {fn}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    return f1